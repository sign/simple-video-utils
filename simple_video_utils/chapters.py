"""Write crops and trims of a video as the chapters of one Matroska file, each frame at its place on the source."""

import json
import os
import shutil
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import chain, islice, pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from av.video.reformatter import ColorRange

from simple_video_utils.video import PACKED, PackedChaptersError, Video, chapter_table, refusing, rotation_filters

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


@refusing("Failed to write chapters", keep=OSError)
def write_chapters(  # noqa: C901 - the checks and the cleanup read best next to the write they guard
    src: str | os.PathLike,
    out: str | os.PathLike,
    clips: Sequence[ChapterClip],
    *,
    crf: int = 23,
    preset: str = "medium",
) -> list[ChapterClip]:
    """Write `clips` as H.264 chapters of the Matroska file `out`, and return the clips written.

    Frame i is written at i / rate, its source time when the source keeps one rate, so the stretches between clips
    hold no frames. Boxes are in displayed pixels
    (rotation applied) and may reach past the frame, which pads with black. Each chapter stores its clip as a JSON
    `payload` tag, and the frame rate beside it. A clip running past the last frame the source decodes is cut there, and
    later ones are dropped. `out` must not exist; a failed write removes it.
    """
    if not clips:
        raise ValueError("write_chapters needs at least one clip")
    if any(left.end_frame > right.start_frame for left, right in pairwise(clips)):
        raise ValueError("clips must be in source order and must not overlap")
    for clip in clips:
        left, top, right, bottom = clip.box
        if not 0 <= clip.start_frame < clip.end_frame or left >= right or top >= bottom:
            raise ValueError(f"clip needs 0 <= start_frame < end_frame and a non-empty box: {clip}")
        if clip.size[0] % 16 or clip.size[1] % 16:
            raise ValueError("H.264 chapter width and height must be multiples of 16")
    with Video(src) as video:
        rate = video.rate
        if not rate:
            raise ValueError("Video has no frame rate information")
        with open(out, "xb") as file:
            try:
                with av.open(file, mode="w", format="matroska") as output:
                    output.set_chapters(_table(clips, rate))
                    width, height = clips[0].size
                    # the muxer takes the stream header (avcC) from the first clip's packets
                    track = output.add_mux_stream("h264", rate=rate, width=width, height=height)
                    video.rewind()  # measuring a missing frame rate reads the whole source
                    decoded = video.container.decode(video.stream)
                    written, position = [], 0
                    for clip in clips:
                        # ponytail: decodes the frames between clips; seek to keyframes if long gaps get slow
                        frames = islice(decoded, clip.start_frame - position, clip.end_frame - position)
                        count = _write_clip(video, track, clip, frames, rate, crf, preset)
                        if count:
                            written.append(replace(clip, end_frame=clip.start_frame + count))
                        if count < clip.end_frame - clip.start_frame:
                            break  # the source ended
                        position = clip.end_frame
                if not written:
                    raise ValueError("every clip starts past the last frame the source decodes")
                if written != list(clips):
                    # Matroska writes chapters ahead of the frames, before the source's real end is known
                    file.close()
                    _replace_chapters(Path(out), _table(written, rate))
            except BaseException:
                # anything, an interrupt included, leaves a truncated file that "xb" would refuse to overwrite on retry
                try:
                    file.close()  # flushes, so it raises again when the device refused the bytes
                finally:
                    Path(out).unlink()
                raise
    return written


def _mark(number: int, start_frame: int, end_frame: int, payload: dict, rate: Fraction) -> dict:
    # each chapter states the rate: the header's comes back from Matroska rounded, and frame numbers drift with it
    start, end = (int(frame * 1000 / rate + Fraction(1, 2)) for frame in (start_frame, end_frame))
    return {"id": number, "start": start, "end": end, "time_base": Fraction(1, 1000),
            "metadata": {"payload": json.dumps(payload), "rate": str(rate)}}


def _table(clips: Sequence[ChapterClip], rate: Fraction) -> list[dict]:
    return [_mark(number, clip.start_frame, clip.end_frame, asdict(clip), rate) for number, clip in enumerate(clips)]


def _replace_chapters(path: Path, chapters: list[dict], moved: Callable[[av.Packet], int] = lambda packet: 0,
                      out: Path | None = None) -> None:
    """Copy the file's packets under a new chapter table, each `moved(packet)` ticks later, over it or to `out`."""
    out = out or path
    # staged beside the output, so the replacement stays on one filesystem
    with TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as directory:
        staged = Path(directory) / out.name
        with av.open(str(path)) as source, av.open(str(staged), mode="w", format="matroska") as output:
            output.set_chapters(chapters)
            track = output.add_stream_from_template(source.streams.video[0])
            for packet in source.demux(video=0):
                if packet.size:  # the empty one only flushes
                    move = moved(packet)
                    packet.pts, packet.stream = packet.pts + move, track
                    if packet.dts is not None:
                        packet.dts += move
                    output.mux(packet)
        staged.replace(out)


def unpack_chapters(path: str | os.PathLike, out: str | os.PathLike | None = None) -> None:
    """Rewrite a Matroska file from before 0.10 where it lies, so that each chapter sits at its source time.

    Transitional. Packets are copied and nothing is encoded, so the frames are the same and it takes milliseconds.
    A file that already keeps source time is left alone. With `out` the file is written there instead, converted or
    as it is, and `path` is only read.
    """
    with av.open(os.fspath(path)) as source:
        chapters, packed = chapter_table(source)
        stream = source.streams.video[0]
        rate, clock = Fraction(stream.average_rate or 1), stream.time_base
        matroska = source.format.name.startswith("matroska")
    if not packed:
        if out is not None:
            # whoever asked for `out` opens it next, also when another process converted `path` a moment ago
            shutil.copyfile(path, out)
        return
    if not matroska:
        raise PackedChaptersError(f"{path} has {PACKED}, and only a Matroska file can be converted")
    table = [_mark(c.id, c.frames.start, c.frames.stop, c.payload, rate) for c in chapters]
    starts = [c.start for c in chapters]
    moves = [round((Fraction(mark["start"], 1000) - c.start) / clock) for mark, c in zip(table, chapters, strict=True)]

    def moved(packet: av.Packet) -> int:  # by the chapter it sat in, with the reader's half frame of slack
        return moves[max(bisect_right(starts, packet.pts * clock + Fraction(1, 2) / rate) - 1, 0)]

    _replace_chapters(Path(path), table, moved, None if out is None else Path(out))


def _neighbourhood(low: int, high: int, size: int) -> tuple[int, int]:
    """[low, high) widened by CONTEXT on each side and out to the 16-pixel grid, within [0, size)."""
    start = min(max(low - CONTEXT, 0) // 16 * 16, size)
    end = min(-(-(high + CONTEXT) // 16) * 16, size)
    return (start, end) if end > start else (0, size)


def _write_clip(video: Video, track: av.VideoStream, clip: ChapterClip, frames: Iterator[av.VideoFrame],
                rate: Fraction, crf: int, preset: str) -> int:
    # decoded before the graph is built: the first frame corrects header colour tags the scale filter would refuse
    first = next(frames, None)
    if first is None:
        return 0
    source, context, rotation = video.stream, video.stream.codec_context, video.rotation
    graph = av.filter.Graph()
    left, top, right, bottom = clip.box
    # a rotated source is turned upright first, as FFmpeg autorotates it, then cut like one stored upright
    frame_width, frame_height = source.width, source.height
    if rotation in (90, 270):
        frame_width, frame_height = frame_height, frame_width
    pixels = source.format
    shifted_chroma = (left % 2 and pixels.chroma_width(2) == 1) or (top % 2 and pixels.chroma_height(2) == 1)
    padding, x, y = [], 0, 0
    outside = left < 0 or top < 0 or right > frame_width or bottom > frame_height
    inside_left, inside_top = max(left, 0), max(top, 0)
    inside_right, inside_bottom = min(right, frame_width), min(bottom, frame_height)
    step_x, step_y = 4 // pixels.chroma_width(4), 4 // pixels.chroma_height(4)
    on_chroma_grid = not (left % step_x or right % step_x or inside_right % step_x
                          or top % step_y or bottom % step_y or inside_bottom % step_y)
    if outside or shifted_chroma:
        converts = context.color_range == ColorRange.JPEG or not (
            on_chroma_grid and inside_left < inside_right and inside_top < inside_bottom)
        if converts:
            # keep nearby pixels, so the conversion preserves chroma at the crop's edges
            (x0, x1), (y0, y1) = _neighbourhood(left, right, frame_width), _neighbourhood(top, bottom, frame_height)
        else:
            x0, y0, x1, y1 = inside_left, inside_top, inside_right, inside_bottom
        left, top, right, bottom = left - x0, top - y0, right - x0, bottom - y0
        x, y = max(0, -left), max(0, -top)
        padding = [graph.add("crop", f"{x1 - x0}:{y1 - y0}:{x0}:{y0}:exact=1")]
        if converts:
            padding.append(graph.add("format", "yuv444p16le"))
        padding.append(graph.add("pad", f"{max(x1 - x0, right) + x}:{max(y1 - y0, bottom) + y}:{x}:{y}:black"))
    width, height = clip.size
    graph.link_nodes(
        graph.add_buffer(template=source),
        # turned in the decoded format, which FFmpeg would otherwise convert whole for the padding first
        *rotation_filters(graph, rotation),
        *([graph.add("format", source.pix_fmt)] if rotation else []),
        *padding,
        graph.add("crop", f"{right - left}:{bottom - top}:{left + x}:{top + y}:exact=1"),
        # no implicit matrix or range conversion
        graph.add("scale", f"{width}:{height}:out_color_matrix={context.colorspace}:out_range={context.color_range}"),
        graph.add("format", source.pix_fmt),
        graph.add("buffersink"),
    )
    graph.configure()
    # each size needs its own encoder: an H.264 keyframe carries the dimensions
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = width, height, source.pix_fmt
    encoder.color_range, encoder.colorspace = context.color_range, context.colorspace
    encoder.color_primaries, encoder.color_trc = context.color_primaries, context.color_trc
    encoder.time_base, encoder.framerate = 1 / rate, rate
    encoder.max_b_frames = 0  # separate encoders must not reorder packets across a chapter boundary
    encoder.options = {"crf": str(crf), "preset": preset}
    for index, frame in enumerate(chain((first,), frames), start=clip.start_frame):
        graph.push(frame)
        cropped = graph.pull()
        cropped.pts, cropped.time_base, cropped.pict_type = index, 1 / rate, 0
        for packet in encoder.encode(cropped):
            packet.stream = track
            track.container.mux(packet)
    for packet in encoder.encode():
        packet.stream = track
        track.container.mux(packet)
    return index + 1 - clip.start_frame
