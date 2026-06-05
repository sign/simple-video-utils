import io
from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple

import av


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    nb_frames: int | None  # from the header or counted by demuxing; estimated for non-seekable input
    time_base: str | None
    duration: float | None  # seconds; None if the container header doesn't carry one
    rotation: int = 0  # display-matrix rotation in degrees; width/height already account for it


@contextmanager
def _open_container(source: str | io.BytesIO):
    """Context manager for safely opening and closing PyAV containers."""
    container = None
    try:
        # metadata_errors='replace': some files carry non-UTF-8 stream metadata
        # (e.g. handler_name in stray mp4s data tracks), which would otherwise
        # raise UnicodeDecodeError before the video stream is even reachable.
        container = av.open(source, metadata_errors="replace")
        yield container
    except Exception as e:
        msg = "Failed to open video"
        raise RuntimeError(msg) from e
    finally:
        if container:
            container.close()


def _count_video_packets(container: av.container.InputContainer) -> int | None:
    """
    Count video frames exactly by demuxing packets (no decoding), then rewind.

    Video codecs carry one frame per packet, so the packet count is the frame
    count. Requires a seekable container; returns None if demuxing fails.
    """
    try:
        return sum(1 for packet in container.demux(video=0) if packet.size)
    except (av.FFmpegError, OSError):
        return None
    finally:
        container.seek(0)


def _probe_rotation(container: av.container.InputContainer) -> int:
    """
    Read the display-matrix rotation by decoding the first frame, then rewind.

    PyAV only exposes the rotation per-frame (``VideoFrame.rotation``), not on
    the stream. Requires a seekable container; returns 0 if the video can't be
    decoded.
    """
    try:
        frame = next(container.decode(video=0), None)
        rotation = frame.rotation if frame is not None else 0
    except (av.FFmpegError, OSError):
        rotation = 0
    container.seek(0)
    return rotation


def video_metadata_from_container(
    container: av.container.InputContainer,
    rotation: int | None = None,
) -> VideoMetadata:
    """
    Extract metadata from an open PyAV container.

    Width/height are reported in display orientation (rotation applied),
    matching the frames yielded by the frames module.

    Args:
        container: Open PyAV container.
        rotation: Display rotation in degrees if already known (e.g. from a
            decoded frame). When None, it is probed by decoding the first
            frame and rewinding — pass it explicitly for non-seekable input.
    """
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    time_base = str(stream.time_base) if stream.time_base else None
    # Prefer the video stream's duration over container.duration. They usually
    # match, but when audio outlasts video the container header reports the
    # longer of the two — and downstream consumers (ffmpeg padding, model APIs
    # that measure the video stream) only see the video stream. PyAV's
    # `stream.duration` is in stream time_base units; some containers (notably
    # browser-recorded webm) don't stamp it, so we fall back to
    # container.duration (microseconds) in that case.
    if stream.duration and stream.time_base:
        duration = float(stream.duration * stream.time_base)
    elif container.duration:
        duration = container.duration / av.time_base
    else:
        duration = None

    if stream.frames > 0:
        nb_frames = stream.frames
    else:
        # Matroska/WebM headers often omit the frame count. When the container
        # is seekable (rotation is None — same contract as rotation probing),
        # demux the packets for an exact count. Otherwise estimate; the
        # container duration can overshoot the video stream, so the estimate
        # may run a frame high.
        nb_frames = _count_video_packets(container) if rotation is None else None
        if nb_frames is None and fps and duration:
            nb_frames = round(duration * fps)

    if rotation is None:
        rotation = _probe_rotation(container)
    rotation %= 360

    width, height = stream.width, stream.height
    if rotation % 180 == 90:
        width, height = height, width

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        nb_frames=nb_frames,
        time_base=time_base,
        duration=duration,
        rotation=rotation,
    )




def video_metadata_from_bytes(data: bytes) -> VideoMetadata:
    """Return key video stream metadata from video bytes."""
    with _open_container(io.BytesIO(data)) as container:
        return video_metadata_from_container(container)


@lru_cache(maxsize=8)
def video_metadata(url_or_path: str) -> VideoMetadata:
    """Return key video stream metadata."""
    with _open_container(url_or_path) as container:
        return video_metadata_from_container(container)
