import io
from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple, Optional, Union

import av


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    nb_frames: Optional[int]
    time_base: Optional[str]
    duration: Optional[float]  # seconds; None if the container header doesn't carry one


@contextmanager
def _open_container(source: Union[str, io.BytesIO]):
    """Context manager for safely opening and closing PyAV containers."""
    container = None
    try:
        container = av.open(source)
        yield container
    except Exception as e:
        msg = "Failed to open video"
        raise RuntimeError(msg) from e
    finally:
        if container:
            container.close()


def video_metadata_from_container(container: av.container.InputContainer) -> VideoMetadata:
    """Extract metadata from an open PyAV container."""
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    nb_frames = stream.frames if stream.frames > 0 else None
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

    return VideoMetadata(
        width=stream.width,
        height=stream.height,
        fps=fps,
        nb_frames=nb_frames,
        time_base=time_base,
        duration=duration,
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
