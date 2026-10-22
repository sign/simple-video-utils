"""Store rectangular crops and time trims as chapters of one Matroska video, and read, split, cut and join them."""

import io
import json
import math
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import chain, islice
from pathlib import Path

import av
from av.video.reformatter import ColorRange

from simple_video_utils.frames import _rotation_filters
from simple_video_utils.metadata import _open_video
from simple_video_utils.slicing import _encode_clip, _iter_packets

# pixels kept around a padded crop before its 4:4:4 conversion, past the reach of FFmpeg's chroma upsampling filter, so
# the crop's chroma comes out as if the whole frame had been converted
CONTEXT = 16


@dataclass(frozen=True)
class ChapterClip:
    """Source frames [start_frame, end_frame), cropped to (left, top, right, bottom), then resized to resolution."""

    start_frame: int
    end_frame: int
    box: tuple[int, int, int, int]
    payload: dict | None = None
    resolution: tuple[int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        left, top, right, bottom = self.box
        return self.resolution or (right - left, bottom - top)


def list_chapters(src: str | Path) -> Iterator[dict]:
    """Yield PyAV chapter dictionaries (start/end ticks, time_base and metadata)."""
    with _open_video(str(src)) as container:
        yield from container.chapters()


@dataclass(frozen=True)
class Chapter:
    """A chapter of a chaptered file: seconds [start, end) on the file's own timeline, and where it starts on its
    source's (the ``start_frame`` of its ``payload`` tag, at the file's rate or the tag's own ``rate``)."""

    start: Fraction
    end: Fraction
    source: Fraction

    @property
    def source_end(self) -> Fraction:
        return self.source + self.end - self.start

    def overlap(self, start: Fraction | float, end: Fraction | float) -> tuple[Fraction, Fraction] | None:
        """The part of source span [start, end) this chapter holds, on the source's timeline."""
        low, high = max(start, self.source), min(end, self.source_end)
        return (Fraction(low), Fraction(high)) if low < high else None

    def local(self, t: Fraction) -> Fraction:
        """A source time on the file's own timeline."""
        return self.start + t - self.source


Source = str | Path | io.BytesIO


def _open(src: Source) -> av.container.InputContainer:
    if isinstance(src, io.BytesIO):
        src.seek(0)  # read from the start, however often it is passed
        return _open_video(src)
    return _open_video(str(src))


def _chapters_of(container: av.container.InputContainer) -> list[Chapter]:
    rate = container.streams.video[0].average_rate
    found = []
    for chapter in container.chapters():
        spec = json.loads({key.lower(): value for key, value in chapter["metadata"].items()}["payload"])
        found.append(Chapter(chapter["start"] * chapter["time_base"], chapter["end"] * chapter["time_base"],
                             spec["start_frame"] / Fraction(spec.get("rate", rate))))
    return found


def _require_chapters(container: av.container.InputContainer) -> list[Chapter]:
    if not (found := _chapters_of(container)):
        message = "the video has no chapters"
        raise ValueError(message)
    return found


def read_chapters(src: Source) -> list[Chapter]:
    """The chapters of a file, [] for a file without them."""
    with _open(src) as container:
        return _chapters_of(container)


def split_chapters(src: Source) -> Iterator[tuple[Chapter, bytes]]:
    """Yield each chapter as its own Matroska file, packets copied and timestamps from 0, as soon as it is read:
    one chapter's packets are held at a time. Raises ``ValueError`` on a file without chapters.

    Every chapter must start on a keyframe that carries its own headers, as ``write_chapters`` and
    ``join_chapters`` write them, and chapters run back to back from 0."""
    with _open(src) as source:
        found = _require_chapters(source)
        stream = source.streams.video[0]
        index, packets = 0, []
        for packet in source.demux(stream):
            if packet.pts is None or not packet.size:  # Matroska stores no dts for reordered frames
                continue
            t = packet.pts * packet.time_base
            if index + 1 < len(found) and t >= found[index + 1].start:  # every chapter holds a frame
                yield found[index], _remux(stream, packets)
                index, packets = index + 1, []
            if not found[index].start <= t < found[index].end:
                message = f"frame at {float(t):.3f} s is outside every chapter"
                raise ValueError(message)
            if not packets and not packet.is_keyframe:
                message = f"chapter {index} does not start on a keyframe"
                raise ValueError(message)
            packets.append(packet)
        if index != len(found) - 1:
            message = f"{len(found) - 1 - index} chapters hold no frames"
            raise ValueError(message)
        yield found[index], _remux(stream, packets)


def _remux(stream: av.VideoStream, packets: list[av.Packet]) -> bytes:
    out = io.BytesIO()
    with av.open(out, mode="w", format="matroska") as destination:
        track = destination.add_stream_from_template(stream)
        shift = packets[0].pts
        for packet in packets:
            packet.stream = track
            packet.pts, packet.dts = packet.pts - shift, packet.dts - shift if packet.dts is not None else None
            destination.mux(packet)
    return out.getvalue()


def slice_chapters(
    src: str | Path, spans: Sequence[tuple[float, float]], size: int | None = None,
) -> Iterator[list[tuple[Fraction, bytes]]]:
    """Yield, per source-timeline span of a chaptered file, its pieces, one per chapter holding a frame that starts
    in it: (where the piece starts on the source's timeline, those frames re-encoded at ``size``). A span between
    chapters has none;
    ``end`` may be ``math.inf``. Clips are cut as they are yielded. Raises ``ValueError`` on a file without chapters."""
    with _open(src) as container:
        found = _require_chapters(container)
        rate = container.streams.video[0].average_rate
    for a, b in spans:
        pieces = []
        for chapter in found:
            if part := chapter.overlap(_exact(a), _exact(b)):
                # the frames that START in [low, high); reading at mid-frame times dodges float rounding at both ends
                first, last = (math.ceil(chapter.local(t) * rate) for t in part)
                if last > first:  # re-encoded: the file's header describes only its first chapter
                    clip = _encode_clip(str(src), float((first + 0.5) / rate), float((last - 0.5) / rate),
                                        float(rate), size)
                    pieces.append((part[0], clip))
        yield pieces


def _exact(t: float) -> Fraction | float:
    return Fraction(str(t)) if math.isfinite(t) else t


def join_chapters(pieces: Sequence[tuple[Fraction, bytes]]) -> bytes:
    """Clips of one encoder setting (so one set of headers), as ``slice_chapters`` cuts them, as the chapters of one
    Matroska file; each chapter's ``payload`` tag holds where it starts on the source's timeline, to the millisecond."""
    out = io.BytesIO()
    with ExitStack() as stack:
        inputs = [stack.enter_context(_open_video(io.BytesIO(clip))) for _, clip in pieces]
        streams = [container.streams.video[0] for container in inputs]
        if len({bytes(stream.codec_context.extradata or b"") for stream in streams}) != 1:
            message = "clips to join differ in codec headers"
            raise ValueError(message)
        packets = [list(_iter_packets(container, stream)) for container, stream in zip(inputs, streams, strict=True)]
        starts = [min(p.pts for p in group) * stream.time_base for group, stream in zip(packets, streams, strict=True)]
        ends = [max(p.pts + p.duration for p in group) * stream.time_base
                for group, stream in zip(packets, streams, strict=True)]
        chapters, offset = [], Fraction(0)
        for (source, _), first, last in zip(pieces, starts, ends, strict=True):
            spec = {"start_frame": round(source * 1000), "rate": "1000"}
            end = offset + last - first
            chapters.append({"id": len(chapters), "start": round(offset * 1000), "end": round(end * 1000),
                             "time_base": Fraction(1, 1000), "metadata": {"payload": json.dumps(spec)}})
            offset = end
        destination = stack.enter_context(av.open(out, mode="w", format="matroska"))
        destination.set_chapters(chapters)
        track = destination.add_stream_from_template(streams[0])
        for group, stream, first, chapter in zip(packets, streams, starts, chapters, strict=True):
            shift = Fraction(chapter["start"], 1000) - first
            for packet in group:
                # Matroska keeps milliseconds; flooring dts keeps it monotonic and never past pts.
                pts, dts = packet.pts * stream.time_base + shift, packet.dts * stream.time_base + shift
                packet.stream, packet.time_base = track, Fraction(1, 1000)
                packet.pts, packet.dts = round(pts * 1000), math.floor(dts * 1000)
                destination.mux(packet)
    return out.getvalue()


def write_chapters(
    src: str | Path,
    out: str | Path,
    clips: Sequence[ChapterClip],
    *,
    crf: int = 23,
    preset: str = "medium",
    thread_type: str = "AUTO",
) -> None:
    """Write crops and trims consecutively, padding outside pixels black and optionally resizing.

    Output must not exist. Coordinates refer to the displayed frame, rotation applied, as ``video_metadata`` and the
    frame readers report it, and a rotated source's chapters come out upright.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    ``preset`` is the x264 preset, x264's own by default. ``thread_type`` is PyAV's for decoding the source ("AUTO",
    "FRAME", "SLICE" or "NONE"); the frames do not change.
    """
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        # AUTO decodes single-slice H.264 frames in parallel, where FFmpeg's slice threading keeps one core busy
        stream.thread_type = thread_type
        rate = stream.average_rate
        chapters = []
        offset = 0
        for clip in clips:
            width, height = clip.size
            if width % 16 or height % 16:
                message = "H.264 chapter width and height must be multiples of 16"
                raise ValueError(message)
            end = offset + clip.end_frame - clip.start_frame
            chapters.append({
                "id": len(chapters),
                "start": int(offset * 1000 / rate + Fraction(1, 2)),
                "end": int(end * 1000 / rate + Fraction(1, 2)),
                "time_base": Fraction(1, 1000), "metadata": {"payload": json.dumps(asdict(clip))},
            })
            offset = end
        with open(out, "xb") as file, av.open(file, mode="w", format="matroska") as destination:
            destination.set_chapters(chapters)
            width, height = clips[0].size
            # The muxer takes the header (avcC) from the first clip's packets.
            track = destination.add_mux_stream("h264", rate=rate, width=width, height=height)
            decoded = source.decode(stream)
            # PyAV reports the display-matrix rotation on decoded frames only
            first = next(decoded, None)
            rotation = 0 if first is None else first.rotation % 360
            decoded = chain(() if first is None else (first,), decoded)
            source_end = output_start = 0
            for clip in clips:
                if clip.start_frame < source_end:
                    source.seek(0, stream=stream)
                    decoded = source.decode(stream)
                    source_end = 0
                # shortcut: decodes gaps too; add keyframe seeking if long gaps matter.
                frames = islice(decoded, clip.start_frame - source_end, clip.end_frame - source_end)
                _write_clip(stream, track, clip, frames, output_start, crf, preset, rotation=rotation)
                source_end = clip.end_frame
                output_start += clip.end_frame - clip.start_frame


def _neighbourhood(low: int, high: int, size: int) -> tuple[int, int]:
    """[low, high) widened by CONTEXT on each side and out to the 16-pixel grid, within [0, size)."""
    start = min(max(low - CONTEXT, 0) // 16 * 16, size)
    end = min(-(-(high + CONTEXT) // 16) * 16, size)
    return (start, end) if end > start else (0, size)


def _write_clip(
    source: av.VideoStream,
    track: av.VideoStream,
    clip: ChapterClip,
    frames: Iterator[av.VideoFrame],
    start: int,
    crf: int,
    preset: str,
    rotation: int = 0,
) -> None:
    rate = source.average_rate
    graph = av.filter.Graph()
    left, top, right, bottom = clip.box
    # a rotated source is turned upright first, as FFmpeg autorotates it, then cut like a source stored upright
    frame_width, frame_height = source.width, source.height
    if rotation in (90, 270):
        frame_width, frame_height = frame_height, frame_width
    pixel_format = source.format
    shifted_chroma = (left % 2 and pixel_format.chroma_width(2) == 1) or (
        top % 2 and pixel_format.chroma_height(2) == 1
    )
    padding, x, y = [], 0, 0
    outside = left < 0 or top < 0 or right > frame_width or bottom > frame_height
    inside_left, inside_top = max(left, 0), max(top, 0)
    inside_right, inside_bottom = min(right, frame_width), min(bottom, frame_height)
    step_x, step_y = 4 // pixel_format.chroma_width(4), 4 // pixel_format.chroma_height(4)
    on_chroma_grid = not (
        left % step_x or right % step_x or inside_right % step_x
        or top % step_y or bottom % step_y or inside_bottom % step_y
    )
    if outside or shifted_chroma:
        needs_conversion = source.codec_context.color_range == ColorRange.JPEG or not (
            on_chroma_grid and inside_left < inside_right and inside_top < inside_bottom
        )
        if needs_conversion:
            # Keep nearby pixels so conversion preserves chroma at the crop's edges.
            x0, x1 = _neighbourhood(left, right, frame_width)
            y0, y1 = _neighbourhood(top, bottom, frame_height)
        else:
            x0, y0, x1, y1 = inside_left, inside_top, inside_right, inside_bottom
        left, top, right, bottom = left - x0, top - y0, right - x0, bottom - y0
        x, y = max(0, -left), max(0, -top)
        padding = [graph.add("crop", f"{x1 - x0}:{y1 - y0}:{x0}:{y0}:exact=1")]
        if needs_conversion:
            padding.append(graph.add("format", "yuv444p16le"))
        padding.append(graph.add("pad", f"{max(x1 - x0, right) + x}:{max(y1 - y0, bottom) + y}:{x}:{y}:black"))
    width, height = clip.size
    graph.link_nodes(
        graph.add_buffer(template=source),
        # turned in the decoded format, which FFmpeg would otherwise convert whole for the padding first
        *_rotation_filters(graph=graph, rotation=rotation),
        *([graph.add("format", source.pix_fmt)] if rotation else []),
        *padding,
        graph.add("crop", f"{right - left}:{bottom - top}:{left + x}:{top + y}:exact=1"),
        # Avoid implicit matrix or range conversions.
        graph.add("scale", f"{width}:{height}:out_color_matrix={source.codec_context.colorspace}:"
                  f"out_range={source.codec_context.color_range}"),
        graph.add("format", source.pix_fmt),
        graph.add("buffersink"),
    )
    graph.configure()
    # Each size needs its own encoder; H.264 keyframes carry the new dimensions.
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = width, height
    encoder.pix_fmt = source.pix_fmt
    encoder.color_range = source.codec_context.color_range
    encoder.colorspace = source.codec_context.colorspace
    encoder.color_primaries = source.codec_context.color_primaries
    encoder.color_trc = source.codec_context.color_trc
    encoder.time_base, encoder.framerate = 1 / rate, rate
    # Independent encoders must not reorder packets across chapter boundaries.
    encoder.max_b_frames = 0
    encoder.options = {"crf": str(crf), "preset": preset}
    indices = range(start, start + clip.end_frame - clip.start_frame)
    for index, frame in zip(indices, frames, strict=True):
        graph.push(frame)
        cropped = graph.pull()
        cropped.pts = index
        cropped.time_base = 1 / rate
        cropped.pict_type = 0
        for packet in encoder.encode(cropped):
            packet.stream = track
            track.container.mux(packet)
    for packet in encoder.encode():
        packet.stream = track
        track.container.mux(packet)
