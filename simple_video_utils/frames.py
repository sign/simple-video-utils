import operator
import sys
from collections.abc import Generator, Iterable
from itertools import islice, pairwise
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import VideoMetadata, _open_container, video_metadata_from_container

# Below this frame size, swscale's threads=0 (auto, one slice job per CPU)
# loses to single-threaded: dispatch overhead exceeds the conversion itself —
# measured 2x slower at 320x240, a wash at 640x360, auto wins from ~960x540.
# Output is byte-identical at any thread count. The crossover is
# hardware-dependent (thread dispatch cost vs single-core throughput), so this
# is a calibration knob, not a derived value.
_SINGLE_THREAD_MAX_PIXELS = 300_000

# rotation=90 maps to a counterclockwise transpose (np.rot90 k=1 equivalent),
# matching ffmpeg autorotate pixel-exactly.
_ROTATION_FILTERS = {
    90: (("transpose", "cclock"),),
    180: (("hflip", None), ("vflip", None)),
    270: (("transpose", "clock"),),
}


def _rotation_graph(template: av.VideoFrame, rotation: int) -> av.filter.Graph:
    """Build a filter graph rotating rgb24 frames shaped like ``template``."""
    graph = av.filter.Graph()
    chain = [graph.add_buffer(template=template)]
    for name, arg in _ROTATION_FILTERS[rotation]:
        chain.append(graph.add(name, arg))
    chain.append(graph.add("buffersink"))
    for a, b in pairwise(chain):
        a.link_to(b)
    graph.configure()
    return graph


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

    Rotation runs as a libavfilter transpose/flip on the rgb24 frame — a pure
    pixel permutation, so it stays byte-identical to np.rot90 while ffmpeg's
    cache-blocked transpose is ~10x faster than numpy's strided copy. It is
    applied after the rgb24 reformat, not on the decoded yuv420p frame:
    transposing before chroma upsampling changes the interpolation and breaks
    pixel-exactness (measured maxdiff 2-3).
    """
    reformatter = av.video.reformatter.VideoReformatter()
    graph = None
    graph_key = None
    for frame in frames:
        threads = 1 if frame.width * frame.height < _SINGLE_THREAD_MAX_PIXELS else 0
        rgb = reformatter.reformat(frame, format='rgb24', threads=threads)
        rotation = frame.rotation % 360
        if rotation and rotation % 90 == 0:
            key = (rotation, rgb.width, rgb.height)
            if key != graph_key:
                graph = _rotation_graph(rgb, rotation)
                graph_key = key
            graph.push(rgb)
            rgb = graph.pull()
            # The filtered frame's linesize may be padded past width*3, making
            # to_ndarray a non-contiguous view — consumers like MediaPipe and
            # OpenCV reject those, so copy (no-op when already contiguous).
            yield np.ascontiguousarray(rgb.to_ndarray())
        else:
            yield rgb.to_ndarray()


def _prepend(
    first: av.VideoFrame,
    rest: Iterable[av.VideoFrame],
) -> Generator[av.VideoFrame, None, None]:
    """Yield ``first`` then everything in ``rest``."""
    yield first
    # Unlike itertools.chain (which pins its arguments until exhaustion), drop
    # the reference so the decoded frame is collectable once consumed.
    del first
    yield from rest


def _validate_parameters(
    start_frame: int | None,
    end_frame: int | None,
    start_time: float | None,
    end_time: float | None,
) -> bool:
    """Validate parameter combinations; returns whether the range is time-based."""
    has_frame_params = start_frame is not None or end_frame is not None
    has_time_params = start_time is not None or end_time is not None

    if has_frame_params and has_time_params:
        msg = "Cannot mix frame-based and time-based parameters"
        raise ValueError(msg)

    # fps-free time checks run here, before the container opens — errors raised
    # inside `with _open_container` get wrapped as "Failed to open video".
    if end_time is not None and (end_time < 0 or end_time < (start_time or 0.0)):
        msg = "invalid time range"
        raise ValueError(msg)

    return has_time_params


def _convert_time_to_frames(
    start_time: float | None,
    end_time: float | None,
    fps: float,
) -> tuple[int, int | None]:
    """Convert time-based parameters to frame indices."""
    # Clamp so a padded window (e.g. clip_start - 0.5s) degrades to "from the
    # beginning" instead of a negative index blowing up islice downstream.
    # Fully-negative ranges were already rejected by _validate_parameters.
    start = max(int((start_time or 0.0) * fps), 0)
    end = int(end_time * fps) if end_time is not None else None
    return start, end


def _nonnegative_index(value: int, name: str) -> int:
    """
    Validate an index eagerly, with a clear error.

    islice would reject bad values only once a read is underway — after
    av.open (leaking the container in read_frames_from_stream) or wrapped in
    a misleading "Failed to open video" (in read_frames_exact). operator.index
    keeps duck-typed integers (np.int64) working while rejecting floats.
    """
    index = operator.index(value)
    if index < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    # bound leaves room for islice's exclusive stop (end + 1)
    if index > sys.maxsize - 1:
        msg = f"{name} is too large"
        raise ValueError(msg)
    return index


def _normalize_frame_range(
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int | None]:
    """Normalize frame parameters with defaults and validation."""
    start = _nonnegative_index(start_frame, "start_frame") if start_frame is not None else 0
    end = _nonnegative_index(end_frame, "end_frame") if end_frame is not None else None

    if end is not None and end < start:
        msg = "invalid frame range"
        raise ValueError(msg)

    return start, end


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


def _select_frames_by_index(
    frames: Iterable[av.VideoFrame],
    origin: float,
    fps: float,
    target_start: int,
    target_end: int | None,
) -> Generator[av.VideoFrame, None, None]:
    """Yield frames whose timestamp-derived index falls in [target_start, target_end]."""
    for frame in frames:
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

    Raises:
        ValueError: If frame and time parameters are mixed, or a range is
            invalid (negative or inverted). Raised at call time, before the
            file is opened. A negative start_time is clamped to 0 instead
            (padded windows are common), as long as the range stays valid.
        TypeError: If a frame index is not an integer (np.int64 is fine).

    Examples:
        # All frames
        frames = list(read_frames_exact("video.mp4"))

        # Frame-based
        frames = list(read_frames_exact("video.mp4", start_frame=0, end_frame=10))

        # Time-based
        frames = list(read_frames_exact("video.mp4", start_time=1.5, end_time=3.0))
    """
    has_time_params = _validate_parameters(start_frame, end_frame, start_time, end_time)
    if not has_time_params:
        frame_range = _normalize_frame_range(start_frame, end_frame)

    # Inner generator so validation above raises at call time, not first next().
    def generate() -> Generator[np.ndarray, None, None]:
        with _open_container(src) as container:
            stream = container.streams.video[0]
            stream.thread_type = thread_type

            # Get FPS - required for all operations
            if not stream.average_rate:
                msg = "Video has no FPS information"
                raise ValueError(msg)
            fps = float(stream.average_rate)

            if has_time_params:
                target_start, target_end = _convert_time_to_frames(start_time, end_time, fps)
            else:
                target_start, target_end = frame_range

            did_seek = _seek_near(target_start, fps, stream, container)
            decoded = container.decode(video=0)
            if did_seek:
                origin = float((stream.start_time or 0) * stream.time_base)
                frames = _select_frames_by_index(decoded, origin, fps, target_start, target_end)
            else:
                stop = target_end + 1 if target_end is not None else None
                frames = islice(decoded, target_start, stop)

            yield from _frames_to_rgb(frames)

    return generate()


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

    Raises:
        ValueError: If skip_frames is negative or too large. Raised at call
            time, before the stream is opened.
        TypeError: If skip_frames is not an integer (np.int64 is fine).

    Note:
        For streaming-friendly formats (WebM), frames are yielded as they're
        decoded without waiting for the complete file. For formats requiring
        seeking (MP4 with moov at end), the stream must be fully available.
        The generator owns the container: if it is discarded without ever
        being iterated, the container is closed only at garbage collection.
    """
    skip_frames = _nonnegative_index(skip_frames, "skip_frames")

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

    # Built outside frame_generator so its closure doesn't capture (and pin)
    # the decoded first_frame (~3 MB at 1080p) for the whole read.
    decoded = container.decode(video=0)
    if first_frame is not None:
        decoded = _prepend(first_frame, decoded)
    frames = islice(decoded, skip_frames, None)

    def frame_generator() -> Generator[np.ndarray, None, None]:
        try:
            yield from _frames_to_rgb(frames)
        finally:
            container.close()

    return meta, frame_generator()
