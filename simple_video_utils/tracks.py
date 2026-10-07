"""Gapped crop tracks: source file -> H.264 Matroska -> full-timeline MP4, using native frames."""

import json
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from simple_video_utils.metadata import _open_video, video_metadata_from_container


@dataclass
class VideoCrop:
    """Frames [start_frame, end_frame), cropped in stored coordinates (before display rotation)."""

    start_frame: int
    end_frame: int
    x: int
    y: int
    width: int
    height: int
    payload: dict | None = None


@contextmanager
def _output(path: str | Path, format: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        artifact = Path(directory) / path.name
        with av.open(str(artifact), "w", format=format) as container:
            yield container
        artifact.replace(path)


def _validate_ranges(ranges: Iterable[tuple[int, int]], total: int) -> None:
    previous_end = 0
    for start, end in ranges:
        if start < previous_end or end <= start or end > total:
            message = "track ranges must be disjoint and within source_frames"
            raise ValueError(message)
        previous_end = end


def _filter(frame: av.VideoFrame, filters: list[tuple[str, str]]) -> av.filter.Graph:
    graph = av.filter.Graph()
    graph.link_nodes(graph.add_buffer(template=frame),
                     *(graph.add(name, args) for name, args in filters), graph.add("buffersink"))
    graph.configure()
    return graph


def write_video_crops(source: str | Path, crops: Sequence[VideoCrop], output: str | Path) -> None:
    """Write source crops as separate H.264 Matroska tracks at their source frame positions.

    Ranges are half-open and disjoint; input order is arbitrary. Odd sizes are padded
    right and bottom to even. Tracks carry a JSON payload and a start-end title.
    Extraction decodes once in frame order, including gaps, and re-encodes at CRF 18.
    Frames stay in FFmpeg, with bounded memory. The format uses the nominal source
    rate, so variable-rate input is placed on a uniform cadence. Output is atomic.
    """
    crops = sorted(crops, key=lambda crop: crop.start_frame)
    if not crops:
        message = "at least one crop is required"
        raise ValueError(message)
    with _open_video(str(source)) as video:
        source_stream = video.streams.video[0]
        metadata = video_metadata_from_container(video)
        source_rate = source_stream.average_rate
        rate = Fraction(source_rate) if source_rate else Fraction(metadata.fps).limit_denominator(1_000_000)
        if rate <= 0 or metadata.nb_frames is None:
            message = "source needs a positive frame rate and a frame count"
            raise ValueError(message)
        _validate_ranges(((crop.start_frame, crop.end_frame) for crop in crops), metadata.nb_frames)
        if any(c.x < 0 or c.y < 0 or c.width <= 0 or c.height <= 0
               or c.x + c.width > source_stream.width or c.y + c.height > source_stream.height for c in crops):
            message = "crops must have positive dimensions and lie within the stored source frame"
            raise ValueError(message)
        with _output(output, "matroska") as destination:
            destination.metadata.update(gapped_tracks="1", source_fps=str(rate), source_frames=str(metadata.nb_frames))
            streams = []
            # Register all tracks before the first packet writes the Matroska header.
            for crop in crops:
                stream = destination.add_stream("libx264", rate=rate, options={"crf": "18"})
                stream.width, stream.height = crop.width + crop.width % 2, crop.height + crop.height % 2
                stream.pix_fmt = "yuv420p"
                stream.metadata.update(
                    title=f"{crop.start_frame}-{crop.end_frame}", payload=json.dumps(crop.payload or {})
                )
                streams.append(stream)
            decoded, previous_end = video.decode(source_stream), 0
            for crop, stream in zip(crops, streams, strict=True):
                graph, count = None, 0
                for count, frame in enumerate(islice(decoded, crop.start_frame - previous_end,
                                                    crop.end_frame - previous_end), start=1):
                    if graph is None:
                        # 4:4:4 preserves odd crop origins and the final odd row/column during padding.
                        filters = [("format", "pix_fmts=yuv444p")] if any(
                            v % 2 for v in (crop.x, crop.y, crop.width, crop.height)
                        ) else []
                        graph = _filter(frame, filters + [
                            ("crop", f"{crop.width}:{crop.height}:{crop.x}:{crop.y}:exact=1"),
                            ("pad", f"{stream.width}:{stream.height}:0:0:black"),
                        ])
                    graph.push(frame)
                    frame = graph.pull()
                    frame.pts, frame.time_base, frame.pict_type = crop.start_frame + count - 1, Fraction(1) / rate, 0
                    destination.mux(stream.encode(frame))
                if count != crop.end_frame - crop.start_frame:
                    message = f"track {crop.start_frame}-{crop.end_frame} decoded {count} frames"
                    raise ValueError(message)
                destination.mux(stream.encode())
                previous_end = crop.end_frame


def _read_tracks(video: av.container.InputContainer) -> tuple[Fraction, int, list[tuple[int, int, av.VideoStream]]]:
    metadata = {key.lower(): value for key, value in video.metadata.items()}
    if metadata.get("gapped_tracks") != "1" or not video.streams.video:
        message = "video needs gapped_tracks=1 and video tracks"
        raise ValueError(message)
    rate, total = Fraction(metadata.get("source_fps", "0")), int(metadata.get("source_frames", "-1"))
    if rate <= 0 or total < 0:
        message = "source_fps must be positive and source_frames nonnegative"
        raise ValueError(message)
    clips = []
    for stream in video.streams.video:
        start, end = map(int, stream.metadata.get("title", "").split("-"))
        clips.append((start, end, stream))
    clips.sort(key=lambda clip: clip[0])
    _validate_ranges(((start, end) for start, end, _ in clips), total)
    return rate, total, clips


def merge_video_tracks(source: str | Path, output: str | Path, *, gap_color: str = "black") -> None:
    """Merge crop tracks into H.264 MP4, centered on the largest track's canvas.

    Gaps use an FFmpeg color (black by default); clip borders stay black. Ordered
    ranges and payloads are stored as JSON in the MP4 comment. Frames stay native,
    output cadence is source_fps, and the complete source_frames timeline is restored.
    """
    with _open_video(str(source)) as video:
        rate, total, clips = _read_tracks(video)
        width, height = max(s.width for _, _, s in clips), max(s.height for _, _, s in clips)
        payloads = [{"start_frame": start, "end_frame": end, "payload": json.loads(s.metadata.get("PAYLOAD", "{}"))}
                    for start, end, s in clips]
        gaps = av.filter.Graph()
        gaps.link_nodes(gaps.add("color", f"c={gap_color}:s={width}x{height}:r={rate}"), gaps.add("buffersink"))
        gaps.configure()
        with _output(output, "mp4") as destination:
            destination.metadata["comment"] = json.dumps(payloads)
            stream = destination.add_stream("libx264", rate=rate, options={"crf": "18"})
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            index = 0

            def encode(frame: av.VideoFrame) -> None:
                nonlocal index
                frame.pts, frame.time_base, frame.pict_type = index, Fraction(1) / rate, 0
                destination.mux(stream.encode(frame))
                index += 1

            for start, end, clip in clips:
                while index < start:
                    encode(gaps.pull())
                video.seek(clip.start_time or 0, stream=clip)
                graph, count = None, 0
                left, top = (width - clip.width) // 2, (height - clip.height) // 2
                for frame in video.decode(clip):
                    if graph is None:
                        filters = [("format", "pix_fmts=yuv444p")] if left % 2 or top % 2 else []
                        graph = _filter(frame, filters + [("pad", f"{width}:{height}:{left}:{top}:black")])
                    graph.push(frame)
                    encode(graph.pull())
                    count += 1
                if count != end - start:
                    message = f"track {start}-{end} decoded {count} frames"
                    raise ValueError(message)
            while index < total:
                encode(gaps.pull())
            destination.mux(stream.encode())
