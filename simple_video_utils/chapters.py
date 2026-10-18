"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import accumulate, islice
from pathlib import Path

import av

from simple_video_utils.metadata import _open_video

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


def write_chapters(
    src: str | Path,
    out: str | Path,
    clips: Sequence[ChapterClip],
    *,
    crf: int = 23,
    preset: str | None = None,
    on_frame: Callable[[int, av.VideoFrame], None] | None = None,
    thread_type: str = "AUTO",
) -> None:
    """Write crops and trims consecutively, padding outside pixels black and optionally resizing.

    Output must not exist. Coordinates refer to encoded pixels, before rotation.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    ``preset`` is the x264 preset (x264's own default when None). ``on_frame(clip index, frame)`` sees each chapter
    frame as it is encoded, cropped and resized, so a caller can measure chapters without decoding the output.
    ``thread_type`` is PyAV's for decoding the source ("AUTO", "FRAME", "SLICE" or "NONE"); the frames do not change.
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
            source_end = output_start = 0
            for index, clip in enumerate(clips):
                if clip.start_frame < source_end:
                    source.seek(0, stream=stream)
                    decoded = source.decode(stream)
                    source_end = 0
                # shortcut: decodes gaps too; add keyframe seeking if long gaps matter.
                frames = islice(decoded, clip.start_frame - source_end, clip.end_frame - source_end)
                seen = None if on_frame is None else (lambda frame, index=index: on_frame(index, frame))
                _write_clip(stream, track, clip, frames, output_start, crf, preset, seen)
                source_end = clip.end_frame
                output_start += clip.end_frame - clip.start_frame


def keep_chapters(src: str | Path, out: str | Path, keep: Sequence[int]) -> None:
    """Copy the chapters at indices ``keep`` of a ``write_chapters`` file to ``out``, packets as they are, back to back.

    Each chapter is its own encode, opening on a keyframe that carries its parameter sets and with no B-frames, so its
    packets stand alone: the copy decodes to the frames ``write_chapters`` writes for those clips alone, with no second
    decode or encode. Output must not exist.
    """
    keep = sorted(set(keep))
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        rate = stream.average_rate
        payloads = [{k.lower(): v for k, v in c["metadata"].items()}["payload"] for c in source.chapters()]
        specs = [json.loads(payload) for payload in payloads]
        lengths = [spec["end_frame"] - spec["start_frame"] for spec in specs]
        starts = list(accumulate(lengths, initial=0))  # each chapter's first frame in src
        chapters, shift, offset = [], {}, 0
        for index in keep:
            chapters.append({
                "id": len(chapters),
                "start": int(offset * 1000 / rate + Fraction(1, 2)),
                "end": int((offset + lengths[index]) * 1000 / rate + Fraction(1, 2)),
                "time_base": Fraction(1, 1000), "metadata": {"payload": payloads[index]},
            })
            shift[index] = starts[index] - offset
            offset += lengths[index]
        first = specs[keep[0]]
        width, height = first["resolution"] or (first["box"][2] - first["box"][0], first["box"][3] - first["box"][1])
        with open(out, "xb") as file, av.open(file, mode="w", format="matroska") as destination:
            destination.set_chapters(chapters)
            track = destination.add_mux_stream("h264", rate=rate, width=width, height=height)
            for packet in source.demux(stream):
                if packet.pts is None:  # the demuxer's closing empty packet
                    continue
                frame = round(packet.pts * packet.time_base * rate)
                chapter = bisect_right(starts, frame) - 1
                if chapter not in shift:
                    continue
                # Matroska stores length-prefixed NAL units; hand the muxer start codes as the encoder does, so it
                # builds the header from the first kept chapter's own parameter sets
                copy = av.Packet(_annex_b(bytes(packet)))
                copy.pts = frame - shift[chapter]
                copy.dts = round(packet.dts * packet.time_base * rate) - shift[chapter]
                copy.time_base = 1 / rate
                copy.is_keyframe = packet.is_keyframe
                copy.stream = track
                destination.mux(copy)


def _annex_b(data: bytes) -> bytes:
    """Length-prefixed H.264 NAL units, as Matroska stores them, with start codes instead."""
    out, position = bytearray(), 0
    while position < len(data):
        size = int.from_bytes(data[position : position + 4])
        out += b"\x00\x00\x00\x01" + data[position + 4 : position + 4 + size]
        position += 4 + size
    return bytes(out)


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
    preset: str | None = None,
    on_frame: Callable[[av.VideoFrame], None] | None = None,
) -> None:
    rate = source.average_rate
    graph = av.filter.Graph()
    left, top, right, bottom = clip.box
    pixel_format = source.format
    shifted_chroma = (left % 2 and pixel_format.chroma_width(2) == 1) or (
        top % 2 and pixel_format.chroma_height(2) == 1
    )
    padding = []
    if left < 0 or top < 0 or shifted_chroma or right > source.width or bottom > source.height:
        # Cut the box's neighbourhood out first, on the chroma grid, so the conversion and padding below cost the
        # crop's pixels rather than the whole frame's: a 1080p frame took 25 ms per frame through them.
        x0, x1 = _neighbourhood(left, right, source.width)
        y0, y1 = _neighbourhood(top, bottom, source.height)
        left, top, right, bottom = left - x0, top - y0, right - x0, bottom - y0
        x, y = max(0, -left), max(0, -top)
        padding = [
            graph.add("crop", f"{x1 - x0}:{y1 - y0}:{x0}:{y0}:exact=1"),
            # Full-resolution, high-depth chroma preserves odd offsets and source precision.
            graph.add("format", "yuv444p16le"),
            graph.add("pad", f"{max(x1 - x0, right) + x}:{max(y1 - y0, bottom) + y}:{x}:{y}:black"),
        ]
    else:
        x = y = 0
    width, height = clip.size
    graph.link_nodes(
        graph.add_buffer(template=source),
        *padding,
        graph.add("crop", f"{right - left}:{bottom - top}:{left + x}:{top + y}:exact=1"),
        graph.add("scale", f"{width}:{height}"),
        graph.add("format", source.pix_fmt),
        graph.add("buffersink"),
    )
    graph.configure()
    # Each size needs its own encoder; H.264 keyframes carry the new dimensions.
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = width, height
    encoder.pix_fmt = source.pix_fmt
    encoder.time_base, encoder.framerate = 1 / rate, rate
    # Independent encoders must not reorder packets across chapter boundaries.
    encoder.max_b_frames = 0
    encoder.options = {"crf": str(crf), **({"preset": preset} if preset else {})}
    indices = range(start, start + clip.end_frame - clip.start_frame)
    for index, frame in zip(indices, frames, strict=True):
        graph.push(frame)
        cropped = graph.pull()
        if on_frame is not None:
            on_frame(cropped)
        cropped.pts = index
        cropped.time_base = 1 / rate
        cropped.pict_type = 0
        for packet in encoder.encode(cropped):
            packet.stream = track
            track.container.mux(packet)
    for packet in encoder.encode():
        packet.stream = track
        track.container.mux(packet)
