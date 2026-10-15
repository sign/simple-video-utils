"""Store rectangular crops and time trims as chapters of one Matroska video."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from simple_video_utils.metadata import _open_video


@dataclass(frozen=True)
class ChapterClip:
    """Source frames [start_frame, end_frame), with a (left, top, right, bottom) pixel crop.

    Coordinates refer to the encoded frame, before any display rotation.
    """

    start_frame: int
    end_frame: int
    box: tuple[int, int, int, int]
    payload: dict | None = None


def list_chapters(src: str | Path) -> Iterator[dict]:
    """Yield PyAV chapter dictionaries (start/end ticks, time_base and metadata)."""
    with _open_video(str(src)) as container:
        yield from container.chapters()


def write_chapters(src: str | Path, out: str | Path, clips: Sequence[ChapterClip], *, crf: int = 23) -> None:
    """Concatenate crops into one video-only Matroska file, in the supplied order.

    Each chapter keeps its source specification as JSON in its ``payload`` tag.
    H.264 crop widths and heights must be multiples of 16.
    PyAV decodes the source and its crop filter handles pixels without NumPy copies.
    The source must have a nominal frame rate; output uses that constant cadence.
    An unsuccessful write leaves any existing destination untouched.
    """
    if Path(src).resolve() == Path(out).resolve():
        message = "source and destination must differ"
        raise ValueError(message)
    if not clips:
        message = "at least one chapter is required"
        raise ValueError(message)
    with _open_video(str(src)) as source:
        stream = source.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate:
            message = "source must report a frame rate"
            raise ValueError(message)
        for clip in clips:
            left, top, right, bottom = clip.box
            if (
                any(type(value) is not int for value in (clip.start_frame, clip.end_frame, *clip.box))
                or not 0 <= clip.start_frame < clip.end_frame
                or not 0 <= left < right <= stream.width
                or not 0 <= top < bottom <= stream.height
            ):
                message = f"invalid chapter: {clip}"
                raise ValueError(message)
            if (right - left) % 16 or (bottom - top) % 16:
                message = "H.264 chapter width and height must be multiples of 16"
                raise ValueError(message)
        # Serialize before opening the destination, so invalid payloads fail without writing.
        payloads = [json.dumps(asdict(clip)) for clip in clips]
        chapters = []
        offset = 0
        for clip, payload in zip(clips, payloads, strict=True):
            end = offset + clip.end_frame - clip.start_frame
            chapters.append({
                "id": len(chapters), "start": offset, "end": end,
                "time_base": 1 / rate, "metadata": {"payload": payload},
            })
            offset = end
        with TemporaryDirectory(dir=Path(out).parent) as temporary:
            encoded = Path(temporary) / "chapters.mkv"
            with av.open(str(encoded), mode="w", format="matroska") as destination:
                destination.set_chapters(chapters)
                track = destination.add_stream("libx264", rate=rate)
                track.width = clips[0].box[2] - clips[0].box[0]
                track.height = clips[0].box[3] - clips[0].box[1]
                track.pix_fmt = stream.pix_fmt
                for chapter, clip in zip(chapters, clips, strict=True):
                    _write_clip(stream, track, clip, chapter["start"], crf)
            encoded.replace(out)


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
    count = 0
    for frame in islice(source.container.decode(source), clip.start_frame, clip.end_frame):
        graph.push(frame)
        cropped = graph.pull()
        cropped.pts = start + count
        cropped.time_base = 1 / rate
        cropped.pict_type = 0
        count += 1
        for packet in encoder.encode(cropped):
            packet.stream = track
            track.container.mux(packet)
    if count != clip.end_frame - clip.start_frame:
        message = f"source ends before frame {clip.end_frame}"
        raise ValueError(message)
    for packet in encoder.encode():
        packet.stream = track
        track.container.mux(packet)
