from collections.abc import Generator, Iterable
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import VideoMetadata, _open_container, video_metadata_from_container


def _frames_to_rgb(frames: Iterable[av.VideoFrame]) -> Generator[np.ndarray, None, None]:
    """
    Convert decoded frames to RGB arrays in display orientation.

    PyAV decodes frames in their stored orientation and does not apply the
    container's display-matrix rotation (unlike the ffmpeg CLI, which
    autorotates). Phone-recorded videos commonly store landscape frames with
    a 90° rotation tag, so we apply it here.

    A single reformatter is reused across all frames of a read so the
    swscale conversion context is built once instead of being reallocated on
    every ``frame.to_ndarray(format='rgb24')`` call — byte-identical output,
    dramatically faster full-clip decode.
    """
    reformatter = av.video.reformatter.VideoReformatter()
    for frame in frames:
        array = reformatter.reformat(frame, format='rgb24').to_ndarray()
        rotation = frame.rotation % 360
        if rotation and rotation % 90 == 0:
            # rotation=90 with k=1 (counterclockwise) matches ffmpeg autorotate pixel-exactly.
            # np.rot90 returns a non-contiguous view, which consumers like MediaPipe
            # and OpenCV reject — copy to a contiguous array.
            array = np.ascontiguousarray(np.rot90(array, k=rotation // 90))
        yield array


def _generate_frames(
    container: av.container.InputContainer,
    skip_frames: int = 0,
    max_frames: int | None = None,
) -> Generator[av.VideoFrame, None, None]:
    """
    Generate decoded frames from a container's current position.

    Decodes frames from the container's current position and yields frames
    after skipping the specified number of frames.

    Args:
        container: Open PyAV container (may be seeked to any position).
        skip_frames: Number of frames to skip from current position before yielding.
        max_frames: Maximum number of frames to yield, or None for all remaining.

    Yields:
        Decoded ``av.VideoFrame`` objects for frames after skipping.
    """
    frames_decoded = 0
    frames_yielded = 0

    for frame in container.decode(video=0):
        if frames_decoded < skip_frames:
            frames_decoded += 1
            continue

        yield frame
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


def _seek_near(
    target_start_frame: int,
    fps: float,
    stream,
    container,
) -> bool:
    """
    Seek toward the target position if it is far enough from the start.

    Seeks directly in the file — does NOT decode or process frames before
    the seek position. This allows efficient extraction from any point in
    the video without reading the entire file. Returns whether a seek
    happened; the resulting position is the keyframe at or before the seek
    point, which can be anywhere, so callers must locate frames by
    timestamp rather than by counting.
    """
    min_seek_seconds = 3.0  # Only seek if target is 3+ seconds from start
    seek_buffer_seconds = 1.0  # Seek 1 second before target

    target_time = target_start_frame / fps
    if target_time < min_seek_seconds:
        return False

    # stream timestamps don't have to start at 0 — seek on the stream's timeline
    origin = stream.start_time or 0
    seek_timestamp = int((target_time - seek_buffer_seconds) / float(stream.time_base)) + origin
    container.seek(seek_timestamp, stream=stream)
    return True


def _generate_frames_by_index(
    container: av.container.InputContainer,
    stream,
    fps: float,
    target_start: int,
    target_end: int | None,
) -> Generator[av.VideoFrame, None, None]:
    """Yield frames whose timestamp-derived index falls in [target_start, target_end]."""
    origin = float((stream.start_time or 0) * stream.time_base)
    for frame in container.decode(video=0):
        if frame.time is None:
            continue
        index = round((frame.time - origin) * fps)
        if index < target_start:
            continue
        if target_end is not None and index > target_end:
            break
        yield frame


def read_frames_exact(
    src: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    thread_type: str = "AUTO",
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
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE", or "NONE").

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
        stream.thread_type = thread_type

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

        if _seek_near(target_start, fps, stream, container):
            frames = _generate_frames_by_index(container, stream, fps, target_start, target_end)
        else:
            frame_count = (target_end - target_start + 1) if target_end is not None else None
            frames = _generate_frames(container, target_start, frame_count)

        yield from _frames_to_rgb(frames)


def read_frames_from_stream(
    stream: BinaryIO,
    skip_frames: int = 0,
    thread_type: str = "AUTO",
    buffer_size: int = 32768, # PyAV default buffer size, can be reduced for lower latency when realtime streaming
) -> tuple[VideoMetadata, Generator[np.ndarray, None, None]]:
    """
    Read frames from a video stream (file-like object).

    Args:
        stream: A file-like object containing video data (e.g., uploaded file).
        skip_frames: Number of initial frames to skip (for resume support).
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE", or "NONE").
        buffer_size: Size of PyAV's internal read buffer in bytes. Smaller values
            reduce latency when streaming (frames are decoded sooner), but increase
            the number of read syscalls. Default is 32768 (PyAV default).

    Returns:
        A tuple of (VideoMetadata, frame_generator).
        The generator yields np.ndarray frames in RGB format (H, W, 3).

    Note:
        For streaming-friendly formats (WebM), frames are yielded as they're
        decoded without waiting for the complete file. For formats requiring
        seeking (MP4 with moov at end), the stream must be fully available.
    """
    # metadata_errors='replace' tolerates non-UTF-8 stream metadata (see _open_container)
    container = av.open(stream, mode='r', buffer_size=buffer_size, metadata_errors="replace")
    try:
        for s in container.streams.video:
            s.thread_type = thread_type

        # The display-matrix rotation is only exposed per-frame, and the stream may
        # not be seekable (e.g. a pipe) — so decode the first frame eagerly for the
        # metadata and hand it back through the generator.
        first_frame = next(container.decode(video=0), None)
        rotation = first_frame.rotation if first_frame is not None else 0
        meta = video_metadata_from_container(container, rotation=rotation)
    except Exception:
        container.close()
        raise

    def raw_frames() -> Generator[av.VideoFrame, None, None]:
        remaining_skip = skip_frames
        if first_frame is not None:
            if remaining_skip == 0:
                yield first_frame
            else:
                remaining_skip -= 1
        yield from _generate_frames(container, skip_frames=remaining_skip, max_frames=None)

    def frame_generator() -> Generator[np.ndarray, None, None]:
        try:
            yield from _frames_to_rgb(raw_frames())
        finally:
            container.close()

    return meta, frame_generator()
