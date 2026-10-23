"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import chain, islice
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np
from av.video.reformatter import ColorRange

from simple_video_utils.frames import FrameSize, _frames_to_rgb, _rotation_filters, _select_frames_by_fps, _validate_fps
from simple_video_utils.metadata import _decoded_rate_and_count, _open_video

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


def list_chapters(src: str | Path | BinaryIO | av.container.InputContainer) -> Iterator[dict]:
    """Yield PyAV chapter dictionaries (start/end ticks, time_base and metadata)."""
    if isinstance(src, av.container.InputContainer):
        yield from src.chapters()
        return
    with _open_video(str(src) if isinstance(src, Path) else src) as container:
        yield from container.chapters()


def read_chapters(
    src: str | Path | BinaryIO, *, fps: float | None = None, size: FrameSize | None = None,
) -> Iterator[tuple[dict | None, Fraction, Iterator[np.ndarray]]]:
    """Yield (native chapter, source start seconds, RGB frame iterator) from one continuous video decode.

    Accepts files and incoming streams whose chapter table precedes the media. Consume each frame iterator
    before advancing to the next chapter. Frames use half-open chapter times, without estimating indices from
    average FPS, so non-keyframe boundaries and variable cadence work. Native resize and FPS selection are
    shared with the frame readers; FPS selection restarts per chapter. Use ``stack_frames(frames)`` for a batch.

    Chapters must not overlap and must have positive duration. A file without chapters yields one iterator
    with chapter None and source start 0. Source starts default to native times; an optional writer payload
    maps them to the original source timeline. No intermediate clips are produced.
    """
    _validate_fps(fps)
    if size is not None and not isinstance(size, FrameSize):
        message = "size must be a FrameSize"
        raise TypeError(message)

    def convert(frames):
        selected = _select_frames_by_fps(frames, fps) if fps is not None else frames
        return _frames_to_rgb(selected, size)

    with _open_video(str(src) if isinstance(src, Path) else src) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded = iter(container.decode(video=0))
        found = sorted(list_chapters(container), key=lambda chapter: chapter["start"] * chapter["time_base"])
        if not found:
            yield None, Fraction(0), convert(decoded)
            return
        frame = next(decoded, None)
        previous_end = None
        for chapter in found:
            low, high = (chapter[key] * chapter["time_base"] for key in ("start", "end"))
            if high <= low or (previous_end is not None and low < previous_end):
                message = "chapters must not overlap and must have positive duration"
                raise ValueError(message)
            previous_end = high

            def native(low=low, high=high):
                nonlocal frame
                while frame is not None:
                    if frame.pts is not None and frame.pts * frame.time_base >= high:
                        break
                    current, frame = frame, next(decoded, None)
                    if current.pts is not None and current.pts * current.time_base >= low:
                        yield current

            yield chapter, _source_start(chapter, stream.average_rate), convert(native())


def _source_start(chapter: dict, rate: Fraction | None) -> Fraction:
    start = chapter["start"] * chapter["time_base"]
    tags = {key.lower(): value for key, value in chapter["metadata"].items()}
    try:
        payload = json.loads(tags.get("payload", "null"))
        if isinstance(payload, dict) and "start_frame" in payload:
            return Fraction(payload["start_frame"]) / Fraction(payload.get("rate", rate))
    except (ValueError, TypeError, ZeroDivisionError):
        pass  # unrelated payload tags retain the native chapter timeline
    return start


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

    Output must not exist, and a failed write removes it. Coordinates refer to the displayed frame, rotation applied,
    as ``video_metadata`` and the frame readers report it, and a rotated source's chapters come out upright.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    ``preset`` is the x264 preset, x264's own by default. ``thread_type`` is PyAV's for decoding the source ("AUTO",
    "FRAME", "SLICE" or "NONE"); the frames do not change.
    """
    # clips are checked before the source is decoded or the output created, so a bad clip costs nothing
    if not clips:
        message = "write_chapters needs at least one clip"
        raise ValueError(message)
    for clip in clips:
        left, top, right, bottom = clip.box
        if not 0 <= clip.start_frame < clip.end_frame or left >= right or top >= bottom:
            message = f"clip needs 0 <= start_frame < end_frame and a non-empty box: {clip}"
            raise ValueError(message)
        width, height = clip.size
        if width % 16 or height % 16:
            message = "H.264 chapter width and height must be multiples of 16"
            raise ValueError(message)
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        # AUTO decodes single-slice H.264 frames in parallel, where FFmpeg's slice threading keeps one core busy
        stream.thread_type = thread_type
        # browser-recorded WebM has no average rate, and its guessed rate is the time base's reciprocal, so the
        # cadence comes from decoding the whole stream, which then rewinds
        rate = stream.average_rate or Fraction(_decoded_rate_and_count(source)[0] or 0).limit_denominator(1001)
        if not rate:
            message = "Video has no frame rate information"
            raise ValueError(message)
        chapters = []
        offset = 0
        for clip in clips:
            end = offset + clip.end_frame - clip.start_frame
            chapters.append({
                "id": len(chapters),
                "start": int(offset * 1000 / rate + Fraction(1, 2)),
                "end": int(end * 1000 / rate + Fraction(1, 2)),
                "time_base": Fraction(1, 1000), "metadata": {"payload": json.dumps(asdict(clip))},
            })
            offset = end
        with open(out, "xb") as file:
            try:
                with av.open(file, mode="w", format="matroska") as destination:
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
                        _write_clip(
                            stream, track, clip, frames, output_start, crf, preset, rate=rate, rotation=rotation,
                        )
                        source_end = clip.end_frame
                        output_start += clip.end_frame - clip.start_frame
            except BaseException:
                # a trim past the source's end, an unencodable pixel format or an interrupt leaves a truncated MKV,
                # which "xb" would then refuse to overwrite on retry
                file.close()
                Path(out).unlink()
                raise



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
    rate: Fraction,
    rotation: int = 0,
) -> None:
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
