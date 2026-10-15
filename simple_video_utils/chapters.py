"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import av

from simple_video_utils.metadata import _open_video


@dataclass(frozen=True)
class ChapterClip:
    """Source frames [start_frame, end_frame), cropped to (left, top, right, bottom)."""

    start_frame: int
    end_frame: int
    box: tuple[int, int, int, int]
    payload: dict | None = None


def list_chapters(src: str | Path) -> Iterator[dict]:
    """Yield PyAV chapter dictionaries (start/end ticks, time_base and metadata)."""
    with _open_video(str(src)) as container:
        yield from container.chapters()


def write_chapters(src: str | Path, out: str | Path, clips: Sequence[ChapterClip], *, crf: int = 23) -> None:
    """Write valid, in-bounds clips consecutively at the source's nominal frame rate.

    Output must not exist. Coordinates refer to encoded pixels, before rotation.
    Each chapter stores its source specification in a JSON ``payload`` tag.
    """
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        rate = stream.average_rate
        chapters = []
        offset = 0
        for clip in clips:
            left, top, right, bottom = clip.box
            if (right - left) % 16 or (bottom - top) % 16:
                message = "H.264 chapter width and height must be multiples of 16"
                raise ValueError(message)
            end = offset + clip.end_frame - clip.start_frame
            chapters.append({
                "id": len(chapters), "start": offset, "end": end,
                "time_base": 1 / rate, "metadata": {"payload": json.dumps(asdict(clip))},
            })
            offset = end
        with open(out, "xb") as file, av.open(file, mode="w", format="matroska") as destination:
            destination.set_chapters(chapters)
            track = destination.add_stream("libx264", rate=rate)
            track.width = clips[0].box[2] - clips[0].box[0]
            track.height = clips[0].box[3] - clips[0].box[1]
            track.pix_fmt = stream.pix_fmt
            for chapter, clip in zip(chapters, clips, strict=True):
                _write_clip(stream, track, clip, chapter["start"], crf)


def _write_clip(source: av.VideoStream, track: av.VideoStream, clip: ChapterClip, start: int, crf: int) -> None:
    rate = track.average_rate
    # shortcut: decode each trim's lead-in; seek when long sources need faster exports.
    source.container.seek(0, stream=source)
    graph = av.filter.Graph()
    left, top, right, bottom = clip.box
    graph.link_nodes(
        graph.add_buffer(template=source),
        graph.add("crop", f"{right - left}:{bottom - top}:{left}:{top}:exact=1"),
        graph.add("buffersink"),
    )
    graph.configure()
    # Each size needs its own encoder; H.264 keyframes carry the new dimensions.
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = right - left, bottom - top
    encoder.pix_fmt = track.pix_fmt
    encoder.time_base, encoder.framerate = 1 / rate, rate
    encoder.options = {"crf": str(crf)}
    frames = islice(source.container.decode(source), clip.start_frame, clip.end_frame)
    for index, frame in enumerate(frames, start):
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
