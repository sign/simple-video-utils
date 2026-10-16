"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

import av

from simple_video_utils.metadata import _open_video


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


def write_chapters(src: str | Path, out: str | Path, clips: Sequence[ChapterClip], *, crf: int = 23) -> None:
    """Write crops and trims consecutively, padding outside pixels black and optionally resizing.

    Output must not exist. Coordinates refer to encoded pixels, before rotation.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    """
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
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
            frames, position, written = stream.container.decode(stream), 0, 0
            for clip in clips:
                if clip.start_frame < position:
                    # Clips in source order share one decode; one that starts before the last ended decodes again.
                    stream.container.seek(0, stream=stream)
                    frames, position = stream.container.decode(stream), 0
                # ponytail: decodes the gaps between clips too; seek to each clip's keyframe if long gaps matter.
                trim = islice(frames, clip.start_frame - position, clip.end_frame - position)
                _write_clip(stream, track, clip, trim, written, crf)
                written += clip.end_frame - clip.start_frame
                position = clip.end_frame


def _write_clip(
    source: av.VideoStream,
    track: av.VideoStream,
    clip: ChapterClip,
    frames: Iterator[av.VideoFrame],
    start: int,
    crf: int,
) -> None:
    rate = source.average_rate
    graph = av.filter.Graph()
    left, top, right, bottom = clip.box
    x, y = max(0, -left), max(0, -top)
    pixel_format = source.format
    shifted_chroma = (left % 2 and pixel_format.chroma_width(2) == 1) or (
        top % 2 and pixel_format.chroma_height(2) == 1
    )
    padding = []
    if x or y or shifted_chroma or right > source.width or bottom > source.height:
        padding = [
            # Full-resolution, high-depth chroma preserves odd offsets and source precision.
            graph.add("format", "yuv444p16le"),
            graph.add("pad", f"{max(source.width, right) + x}:{max(source.height, bottom) + y}:{x}:{y}:black"),
        ]
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
    encoder.options = {"crf": str(crf)}
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
