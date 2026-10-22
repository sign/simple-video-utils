"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

import av
from av.video.reformatter import ColorRange

from simple_video_utils.metadata import _open_video, _probe_rotation

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
    preset: str = "medium",
    thread_type: str = "AUTO",
) -> None:
    """Write crops and trims consecutively, padding outside pixels black and optionally resizing.

    Output must not exist. Coordinates refer to the displayed frame, with rotation applied.
    Source frames are rotated upright before cropping and padding.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    ``preset`` is the x264 preset, x264's own by default. ``thread_type`` is PyAV's for decoding the source ("AUTO",
    "FRAME", "SLICE" or "NONE"); the frames do not change.
    """
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        # AUTO decodes single-slice H.264 frames in parallel, where FFmpeg's slice threading keeps one core busy
        stream.thread_type = thread_type
        rotation = _probe_rotation(source) % 360
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
            for clip in clips:
                if clip.start_frame < source_end:
                    source.seek(0, stream=stream)
                    decoded = source.decode(stream)
                    source_end = 0
                # shortcut: decodes gaps too; add keyframe seeking if long gaps matter.
                frames = islice(decoded, clip.start_frame - source_end, clip.end_frame - source_end)
                _write_clip(stream, track, clip, frames, output_start, crf, preset, rotation)
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
    rotation: int,
) -> None:
    rate = source.average_rate
    graph = av.filter.Graph()
    # PyAV exposes rotation metadata but does not autorotate decoded pixels.
    autorotate = []
    source_width, source_height = source.width, source.height
    if rotation in (90, 270):
        autorotate = [graph.add("transpose", "cclock" if rotation == 90 else "clock")]
        source_width, source_height = source_height, source_width
    elif rotation == 180:
        autorotate = [graph.add("hflip"), graph.add("vflip")]
    left, top, right, bottom = clip.box
    pixel_format = source.format
    shifted_chroma = (left % 2 and pixel_format.chroma_width(2) == 1) or (
        top % 2 and pixel_format.chroma_height(2) == 1
    )
    padding, x, y = [], 0, 0
    outside = left < 0 or top < 0 or right > source_width or bottom > source_height
    inside_left, inside_top = max(left, 0), max(top, 0)
    inside_right, inside_bottom = min(right, source_width), min(bottom, source_height)
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
            x0, x1 = _neighbourhood(left, right, source_width)
            y0, y1 = _neighbourhood(top, bottom, source_height)
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
        *autorotate,
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
