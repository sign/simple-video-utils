from collections.abc import Generator
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import VideoMetadata, _open_container, video_metadata_from_container


def _generate_frames(
    container: av.container.InputContainer,
    skip_frames: int = 0,
    max_frames: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate RGB frames from a container's current position.

    Decodes frames from the container's current position and yields frames
    after skipping the specified number of frames.

    Args:
        container: Open PyAV container (may be seeked to any position).
        skip_frames: Number of frames to skip from current position before yielding.
        max_frames: Maximum number of frames to yield, or None for all remaining.

    Yields:
        RGB numpy arrays (H, W, 3) for frames after skipping.
    """
    frames_decoded = 0
    frames_yielded = 0

    for frame in container.decode(video=0):
        if frames_decoded < skip_frames:
            frames_decoded += 1
            continue

        yield frame.to_ndarray(format='rgb24')
        frames_yielded += 1

        if max_frames is not None and frames_yielded >= max_frames:
            break
        frames_decoded += 1

def _validate_parameters(
    start_frame: int | None,
    end_frame: int | None,
    start_time: float | None,
    end_time: float | None,
) -> tuple[bool, bool]:
    """Validate that time and frame parameters aren't mixed."""
    has_frame_params = start_frame is not None or end_frame is not None
    has_time_params = start_time is not None or end_time is not None

    if has_frame_params and has_time_params:
        msg = "Cannot mix frame-based and time-based parameters"
        raise ValueError(msg)

    return has_frame_params, has_time_params


def _convert_time_to_frames(
    start_time: float | None,
    end_time: float | None,
    fps: float,
) -> tuple[int, int | None]:
    """Convert time-based parameters to frame indices."""
    start = int((start_time or 0.0) * fps)
    end = int(end_time * fps) if end_time is not None else None

    if end is not None and end < start:
        msg = "invalid frame range"
        raise ValueError(msg)

    return start, end


def _normalize_frame_range(
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int | None]:
    """Normalize frame parameters with defaults and validation."""
    start = start_frame if start_frame is not None else 0

    if end_frame is not None:
        assert end_frame >= start >= 0, "invalid frame range"
    else:
        assert start >= 0, "start_frame must be non-negative"

    return start, end_frame


def _calculate_seek_position(
    target_start_frame: int,
    fps: float,
    stream,
    container,
) -> int:
    """
    Calculate and perform seeking if beneficial.

    Seeks directly to the target position in the file - does NOT decode
    or process frames before the seek position. This allows efficient
    extraction from any point in the video without reading the entire file.

    Returns the frame number where container is positioned after seeking.
    """
    min_seek_seconds = 3.0  # Only seek if target is 3+ seconds from start
    seek_buffer_seconds = 1.0  # Seek 1 second before target

    target_time = target_start_frame / fps

    # Only seek if far enough from start
    if target_time < min_seek_seconds:
        return 0  # Start from beginning

    # Seek to 1 second before target (jumps directly in file, no decoding)
    seek_time = target_time - seek_buffer_seconds
    seek_timestamp = int(seek_time / float(stream.time_base))
    container.seek(seek_timestamp, stream=stream)

    return int(seek_time * fps)


def read_frames_exact(
    src: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Return frames as RGB np.ndarrays from specified range.

    Supports both frame-based and time-based range specification.
    Uses PyAV for efficient frame extraction.

    Args:
        src: Path to video file or URL.
        start_frame: Starting frame index (0-based). Mutually exclusive with start_time.
        end_frame: Ending frame index (inclusive), or None for end of video.
        start_time: Starting time in seconds. Mutually exclusive with start_frame.
        end_time: Ending time in seconds, or None for end of video.

    Returns:
        Generator yielding RGB numpy arrays (H, W, 3).

    Examples:
        # All frames
        frames = list(read_frames_exact("video.mp4"))

        # Frame-based
        frames = list(read_frames_exact("video.mp4", start_frame=0, end_frame=10))

        # Time-based
        frames = list(read_frames_exact("video.mp4", start_time=1.5, end_time=3.0))
    """
    # Validate parameters early (before opening file)
    has_frame_params, has_time_params = _validate_parameters(
        start_frame, end_frame, start_time, end_time
    )

    # Early validation for frame-based parameters (before opening file)
    if has_frame_params:
        _normalize_frame_range(start_frame, end_frame)

    with _open_container(src) as container:
        stream = container.streams.video[0]

        # Get FPS - required for all operations
        if not stream.average_rate:
            msg = "Video has no FPS information"
            raise ValueError(msg)
        fps = float(stream.average_rate)

        # Convert parameters to frame indices
        if has_time_params:
            target_start, target_end = _convert_time_to_frames(start_time, end_time, fps)
        else:
            target_start, target_end = _normalize_frame_range(start_frame, end_frame)

        # Seek to approximate position (if beneficial)
        seek_position = _calculate_seek_position(target_start, fps, stream, container)

        # Calculate how many frames to skip/yield from seek position
        skip_count = target_start - seek_position
        frame_count = (target_end - target_start + 1) if target_end is not None else None

        yield from _generate_frames(container, skip_count, frame_count)


def read_frames_from_stream(
    stream: BinaryIO,
    skip_frames: int = 0,
) -> tuple[VideoMetadata, Generator[np.ndarray, None, None]]:
    """
    Read frames from a video stream (file-like object).

    Args:
        stream: A file-like object containing video data (e.g., uploaded file).
        skip_frames: Number of initial frames to skip (for resume support).

    Returns:
        A tuple of (VideoMetadata, frame_generator).
        The generator yields np.ndarray frames in RGB format (H, W, 3).

    Note:
        For streaming-friendly formats (WebM), frames are yielded as they're
        decoded without waiting for the complete file. For formats requiring
        seeking (MP4 with moov at end), the stream must be fully available.
    """
    container = av.open(stream, mode='r')
    meta = video_metadata_from_container(container)

    def frame_generator() -> Generator[np.ndarray, None, None]:
        try:
            yield from _generate_frames(container, skip_frames=skip_frames, max_frames=None)
        finally:
            container.close()

    return meta, frame_generator()
