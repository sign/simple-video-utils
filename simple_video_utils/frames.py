import math
import operator
import sys
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import islice, tee
from typing import BinaryIO

import av
import numpy as np
from av.video.reformatter import Interpolation, VideoReformatter

from simple_video_utils.metadata import (
    VideoMetadata,
    _open_container,
    _open_video,
    video_metadata,
    video_metadata_from_container,
)
from simple_video_utils.timeline import Timeline

# For RGB conversion below this frame size, threads=0 (auto, one slice job per CPU)
# loses to single-threaded: dispatch overhead exceeds the conversion itself —
# measured 2x slower at 320x240, a wash at 640x360, auto wins from ~960x540.
# Output is byte-identical at any thread count. The crossover is
# hardware-dependent (thread dispatch cost vs single-core throughput), so this
# is a calibration knob, not a derived value.
_SINGLE_THREAD_MAX_PIXELS = 300_000


@dataclass(frozen=True)
class FrameSize:
    """Resize to (width, height) using the specified interpolation.

    Dimensions are in display orientation. Interpolation defaults to PyAV's AREA.
    """

    resolution: tuple[int, int]
    method: Interpolation = Interpolation.AREA

    def __post_init__(self) -> None:
        if len(self.resolution) != 2 or any(operator.index(side) <= 0 for side in self.resolution):
            message = "resolution must be two positive integers, (width, height)"
            raise ValueError(message)
        if not isinstance(self.method, Interpolation):
            message = "method must be a PyAV Interpolation"
            raise TypeError(message)


@dataclass(frozen=True)
class FrameMetadata:
    """Native decoded-frame information, before resize, skipping or FPS selection.

    ``index`` counts decoded frames from the beginning of this read, starting at zero.
    """

    index: int
    pts: int | None
    time_base: Fraction | None

    @property
    def time(self) -> Fraction | None:
        """Exact presentation time in seconds, when the decoder supplies a timestamp."""
        return self.pts * self.time_base if self.pts is not None and self.time_base is not None else None


def _rotation_filters(graph: av.filter.Graph, rotation: int) -> list[av.filter.context.FilterContext]:
    """Filters that turn stored pixels upright; 90 degrees is counterclockwise, matching display-matrix rotation."""
    if rotation == 180:
        return [graph.add("hflip"), graph.add("vflip")]
    if rotation in (90, 270):
        return [graph.add("transpose", "cclock" if rotation == 90 else "clock")]
    return []


def _rotation_graph(rotation: int, width: int, height: int, time_base: Fraction | None) -> av.filter.Graph:
    """Rotate RGB pixels to display orientation."""
    graph = av.filter.Graph()
    # add_buffer without an explicit time_base is deprecated; the value only
    # affects pts interpretation, never pixels, and we only take arrays.
    source = graph.add_buffer(width=width, height=height, format="rgb24",
                              time_base=time_base or Fraction(1, 1000))
    graph.link_nodes(source, *_rotation_filters(graph=graph, rotation=rotation), graph.add("buffersink"))
    graph.configure()
    return graph


def _frames_to_rgb(
    frames: Iterable[av.VideoFrame], size: FrameSize | None = None,
) -> Generator[np.ndarray, None, None]:
    """Resize native planes, convert to RGB, then apply display rotation.

    PyAV does not apply display rotation while decoding. Rotate after RGB conversion
    so chroma interpolation stays unchanged; native transpose also avoids NumPy's strided copy.
    """
    # Separate contexts retain each stage's scaling and conversion setup.
    scaler = VideoReformatter()
    converter = VideoReformatter()
    # Graphs are stateful: reuse per geometry within this read, never across readers.
    rotation_graph = lru_cache(maxsize=1)(_rotation_graph)
    for frame in frames:
        rotation = frame.rotation % 360
        if size is not None:
            width, height = size.resolution[::-1] if rotation in (90, 270) else size.resolution
            frame = scaler.reformat(frame, width=width, height=height, interpolation=size.method, threads=1)
        threads = 1 if frame.width * frame.height < _SINGLE_THREAD_MAX_PIXELS else 0
        rgb = converter.reformat(frame, format='rgb24', threads=threads)
        if rotation in (90, 180, 270):
            graph = rotation_graph(rotation, rgb.width, rgb.height, frame.time_base)
            graph.push(rgb)
            rgb = graph.pull()
        # Padded linesizes can produce strided views, which downstream image tools reject.
        yield np.ascontiguousarray(rgb.to_ndarray())


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
    # frames timed inside [start_time, end_time]; epsilon: 0.7 s at 30 fps is 20.999999999999996 in floats
    start = max(math.ceil((start_time or 0.0) * fps - 1e-6), 0)
    end = int(end_time * fps + 1e-6) if end_time is not None else None
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
    target_time: float,
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
    """
    Yield the frames at indices [target_start, target_end] after a seek.

    Only the *first* frame is located via its timestamp-derived index; the
    rest are counted from it, like the non-seeked islice path. Re-deriving
    the index per frame from average_rate drifts against the true cadence
    when the header rate is slightly off, over- or under-delivering by a
    frame every 1/|fps_true - average_rate| seconds (issue #26).
    """
    iterator = iter(frames)
    for first in iterator:
        if first.time is not None and round((first.time - origin) * fps) >= target_start:
            break
    else:
        return
    yield first
    yield from islice(iterator, target_end - target_start if target_end is not None else None)


def _select_frames_by_fps(
    frames: Iterable,
    fps: float,
    get_frame: Callable = lambda frame: frame,
    clock: Callable = lambda time: time,
) -> Generator:
    """
    Downsample to ~fps by keeping the first frame of each 1/fps bucket.

    Drop-only: when fps meets or exceeds the source rate every frame passes
    through unchanged — frames are never duplicated. Buckets are anchored at
    the first frame's timestamp, so selection stays uniform across VFR gaps
    and after seeks.

    ``get_frame`` lets the same selection run over ``(index, frame)`` pairs
    (the return_indices path) — items are yielded as given. ``clock`` maps
    frame seconds first, e.g. ``Timeline.squeeze`` keeps one grid across gaps.
    """
    origin = None
    last_bucket = None
    for item in frames:
        frame = get_frame(item)
        if frame.time is None:
            continue
        time = clock(frame.time)
        if origin is None:
            origin = time
        # one tick of slack: containers round timestamps to their clock (a 15 fps mkv stores 0, 67, 133 ms)
        bucket = int((time - origin + float(frame.time_base)) * fps)
        if bucket != last_bucket:
            yield item
            last_bucket = bucket


def _frames_to_rgb_with_info[T](
    items: Iterable[tuple[T, av.VideoFrame]],
    size: FrameSize | None = None,
) -> Generator[tuple[np.ndarray, T], None, None]:
    """Convert pixels while preserving the paired index or metadata."""
    frames, pairs = tee(items, 2)
    for rgb, (info, _) in zip(_frames_to_rgb((f for _, f in frames), size), pairs, strict=True):
        yield rgb, info


def _stream_frames_to_rgb(frames, size, return_frame_metadata):
    if not return_frame_metadata:
        yield from _frames_to_rgb((frame for _, frame in frames), size)
        return
    items = ((FrameMetadata(i, frame.pts, frame.time_base), frame) for i, frame in frames)
    yield from _frames_to_rgb_with_info(items, size)


def _validate_fps(fps: float | None) -> None:
    if fps is not None and fps <= 0:
        msg = "fps must be positive"
        raise ValueError(msg)


def read_frames_exact(
    src: str | av.container.InputContainer,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    thread_type: str = "AUTO",
    fps: float | None = None,
    return_indices: bool = False,
    size: FrameSize | None = None,
) -> Generator[np.ndarray, None, None] | Generator[tuple[np.ndarray, int], None, None]:
    """
    Return frames as RGB np.ndarrays from specified range.

    Supports both frame-based and time-based range specification.
    Uses PyAV for efficient frame extraction.

    Args:
        src: Path to video file, URL, or an open container (see
            metadata.open_video) to reuse across metadata and frame reads.
        start_frame: Starting frame index (0-based). Mutually exclusive with start_time.
        end_frame: Ending frame index (inclusive), or None for end of video.
        start_time: Starting time in seconds. Mutually exclusive with start_frame.
        end_time: Ending time in seconds, or None for end of video. A time range selects
            the frames timed inside it: frames ceil(start_time * fps) through floor(end_time * fps).
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE", or "NONE").
        fps: Target frame rate. Drops frames (roughly uniformly, by timestamp)
            to approximate this rate; never duplicates, so a target at or
            above the source rate returns every frame. None keeps all frames.
        return_indices: When True, yield ``(array, index)`` pairs, where
            ``index`` is the frame's absolute 0-based position in the whole
            video (the start offset is included, and fps drop-selection
            reports exactly the frames it kept). Default False: plain arrays.
        size: Optional output resolution and interpolation in display orientation.
            Resizes native planes before RGB conversion. None preserves each frame's size.

    Returns:
        Generator yielding RGB numpy arrays (H, W, 3), or (array, index)
        pairs when return_indices is True.

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
    _validate_fps(fps)
    if size is not None and not isinstance(size, FrameSize):
        message = "size must be a FrameSize"
        raise TypeError(message)
    if not has_time_params:
        frame_range = _normalize_frame_range(start_frame, end_frame)

    # Inner generator so validation above raises at call time, not first next().
    def generate() -> Generator[np.ndarray, None, None]:
        with _open_container(src) as container:
            stream = container.streams.video[0]
            # A reused container keeps its owner's policy (open_video's thread_type,
            # e.g. "NONE" for forked workers), even before its codec has opened.
            if not isinstance(src, av.container.InputContainer):
                stream.thread_type = thread_type

            # Get FPS - required for all operations. Some containers
            # (browser-recorded WebM) omit the average rate; _count_and_rate
            # recovers the true cadence through the metadata layer — cached
            # per path, so repeated reads (e.g. slice_video's per-range
            # calls) pay the decode-derived rate once.
            # ponytail: a reused open_video container re-derives per call;
            # cache per container if no-rate files show up in that path.
            timeline = Timeline.read(container)
            source_fps = float(timeline.rate or stream.average_rate or 0) or _count_and_rate(src)[1]
            if not source_fps:
                msg = "Video has no FPS information"
                raise ValueError(msg)

            if has_time_params:
                target_start, target_end = _convert_time_to_frames(start_time, end_time, source_fps)
            else:
                target_start, target_end = frame_range

            indexed = _read_range(container, timeline, source_fps, target_start, target_end)
            if fps is not None:
                indexed = _select_frames_by_fps(indexed, fps, get_frame=operator.itemgetter(1), clock=timeline.squeeze)

            if return_indices:
                yield from _frames_to_rgb_with_info(indexed, size)
            else:
                yield from _frames_to_rgb((frame for _, frame in indexed), size)

    return generate()


def _read_range(container, timeline, source_fps, target_start, target_end):
    if target_end is not None and target_end < target_start:
        return iter(())  # a time range between two frames holds none
    if timeline.chapters.size:
        return _read_chapters(container, timeline, source_fps, target_start, target_end)
    return _read_frames(container, source_fps, target_start, target_end)


def _read_frames(container, source_fps, target_start, target_end):
    """(index, frame) for frames [target_start, target_end] of an unchaptered video."""
    stream = container.streams.video[0]
    did_seek = _seek_near(target_start / source_fps, stream, container)
    decoded = container.decode(video=0)
    if did_seek:
        origin = float((stream.start_time or 0) * stream.time_base)
        # guessed_rate (av_guess_frame_rate) recovers the true cadence
        # when the header's average_rate drifts from the actual frame
        # timestamps (issue #26) — average_rate would locate the start
        # frame off by one past each drift point. But when there is no
        # average_rate at all, guessed_rate is garbage (the time_base
        # reciprocal — 1000 on Matroska), so only the derived rate can
        # locate frames.
        locate_fps = float(stream.guessed_rate) if stream.average_rate and stream.guessed_rate else source_fps
        frames = _select_frames_by_index(decoded, origin, locate_fps, target_start, target_end)
    else:
        stop = target_end + 1 if target_end is not None else None
        frames = islice(decoded, target_start, stop)
    return enumerate(frames, start=target_start)


def _read_chapters(container, timeline, rate, target_start, target_end):
    """(source index, frame) for source frames [target_start, target_end] that a chaptered video holds.

    Each chapter's first frame is located by timestamp and later ones are counted, as after a seek.
    """
    stream = container.streams.video[0]
    origin = float((stream.start_time or 0) * stream.time_base)
    _seek_near(timeline.to_file(target_start / rate) - origin, stream, container)
    starts, _, sources = timeline.chapters.T
    chapter = index = None
    for frame in container.decode(video=0):
        # half a frame of slack: chapter starts sit on Matroska's millisecond clock
        current = max(int(np.searchsorted(starts, frame.time + 0.5 / rate, side="right")) - 1, 0)
        index = index + 1 if current == chapter else round((sources[current] + frame.time - starts[current]) * rate)
        chapter = current
        if target_end is not None and index > target_end:
            return
        if index >= target_start:
            yield index, frame


def stack_frames(frames: Iterable[np.ndarray], size_hint: int = 0) -> np.ndarray:
    """
    Stack an iterable of equal-shape arrays into one (N, ...) array.

    Equals ``np.stack(list(frames))``, but each array is copied into
    preallocated chunks as it arrives — while cache-hot, letting its source
    buffer free immediately — instead of everything staying pinned until one
    bulk copy over cold memory. With an accurate ``size_hint`` the batch is
    allocated once and returned directly; without one (or when it's wrong),
    chunks grow geometrically and one concatenation assembles them.

    Raises:
        ValueError: If the iterable is empty or arrays differ in shape, like np.stack.
    """
    chunk = size_hint or 64
    chunks = []
    buf = None
    count = 0
    shape = None
    for array in frames:
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            message = f"frames have different shapes: {shape} and {array.shape}"
            raise ValueError(message)
        if buf is None:
            buf = np.empty((chunk, *array.shape), array.dtype)
            chunk *= 2
            count = 0
        buf[count] = array
        count += 1
        if count == len(buf):
            chunks.append(buf)
            buf = None
    if buf is not None:
        chunks.append(buf[:count])
    if not chunks:
        msg = "no frames to stack"
        raise ValueError(msg)
    if len(chunks) == 1 and chunks[0].base is None:
        # exactly filled one chunk (accurate size_hint): zero extra copies
        return chunks[0]
    return np.concatenate(chunks)


@lru_cache(maxsize=128)
def _count_and_rate_cached(src: str) -> tuple[int, float]:
    """
    Frame-count hint and fps from the header or duration, with metadata as a fallback.

    The count feeds stack_frames' size_hint, which tolerates being wrong: the
    accurate-but-heavy video_metadata (its rotation probe decodes a frame)
    would cost ~25% of a small clip's read time for precision a hint doesn't
    need. The rate is load-bearing — it becomes read_frames_exact's
    source_fps — and is exact whichever branch supplies it: average_rate
    straight from the header, or video_metadata's decode-derived recovery
    when the header carries none.
    """
    with _open_container(src) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        total = _header_frames(container, stream, fps)
    return _fill_from_metadata(total, fps, lambda: video_metadata(src))


def _header_frames(container: av.container.InputContainer, stream: av.VideoStream, fps: float) -> int:
    """
    Estimate missing frame counts from duration; this only sizes the batch.

    Counting packets via video_metadata would scan the whole file.
    """
    if stream.frames or not fps:
        return stream.frames
    chapters = Timeline.read(container).chapters
    if chapters.size:
        seconds = float((chapters[:, 1] - chapters[:, 0]).sum())
    elif stream.duration and stream.time_base:
        seconds = float(stream.duration * stream.time_base)
    else:
        seconds = (container.duration or 0) / av.time_base
    return math.ceil(seconds * fps)


def _fill_from_metadata(total: int, fps: float, get_meta) -> tuple[int, float]:
    """Fill whatever the header didn't carry from the metadata layer."""
    if not total or not fps:
        meta = get_meta()
        total, fps = total or meta.nb_frames or 0, fps or meta.fps
    return total, fps


def _count_and_rate(src: str | av.container.InputContainer) -> tuple[int, float]:
    if isinstance(src, str):
        return _count_and_rate_cached(src)
    # Open containers read the header directly, uncached — caching keyed on a
    # container would pin it and serve stale answers after it's closed.
    stream = src.streams.video[0]
    fps = float(stream.average_rate or 0)
    return _fill_from_metadata(_header_frames(src, stream, fps), fps, lambda: video_metadata_from_container(src))


def read_frames_batched(
    src: str | av.container.InputContainer,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    thread_type: str = "AUTO",
    fps: float | None = None,
    return_indices: bool = False,
    size: FrameSize | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Read a frame range as one batched (N, H, W, 3) RGB uint8 array.

    ``stack_frames`` over ``read_frames_exact`` (same parameters), with the
    batch preallocated from a size hint: the requested range is converted to a frame
    count with the same helpers read_frames_exact uses, clamped to the
    clip's frame count from its header or duration — 26-38% faster than
    stacking afterward.
    ``torch.from_numpy(...)`` wraps the result zero-copy; building the batch
    in torch directly measured 3x slower (per-op dispatch), so numpy is the
    fast path either way.

    With ``return_indices=True`` returns ``(frames, indices)``: a 1-D int64
    array of the same length as the batch, giving each frame's absolute
    0-based index in the whole video (the start offset is included, and fps
    drop-selection reports exactly the frames it kept).

    ``size`` resizes before RGB conversion, as in ``read_frames_exact``.
    Supply it when reading across chapters with different frame dimensions.

    Raises:
        ValueError: Same call-time validation as read_frames_exact, and,
            like np.stack, if the range contains no frames or frames differ in shape.
    """
    # Created first so parameter validation raises before the metadata peek.
    frames = read_frames_exact(src, start_frame, end_frame, start_time, end_time, thread_type, fps,
                               return_indices=return_indices, size=size)

    total, source_fps = _count_and_rate(src)
    if start_time is not None or end_time is not None:
        start, end = _convert_time_to_frames(start_time, end_time, source_fps)
    else:
        start, end = _normalize_frame_range(start_frame, end_frame)
    if total:
        end = total - 1 if end is None else end
    # chaptered reads count source frames, so the window can span frames the file doesn't hold
    hint = max(min(end - start + 1, total or end - start + 1), 0) if end is not None else 0
    if hint and fps is not None and source_fps and fps < source_fps:
        # bucket selection keeps ~one frame per 1/fps; hint tolerates the estimate
        hint = int(hint * fps / source_fps) + 1
    if not return_indices:
        return stack_frames(frames, size_hint=hint)

    indices: list[int] = []
    return (
        stack_frames((indices.append(i) or a for a, i in frames), size_hint=hint),
        np.asarray(indices, dtype=np.int64),
    )

def read_frames_from_stream(
    stream: BinaryIO | av.container.InputContainer,
    skip_frames: int = 0,
    thread_type: str = "AUTO",
    buffer_size: int = 32768, # PyAV default buffer size, can be reduced for lower latency when realtime streaming
    fps: float | None = None,
    *,
    size: FrameSize | None = None,
    return_frame_metadata: bool = False,
) -> tuple[VideoMetadata, Generator[np.ndarray | tuple[np.ndarray, FrameMetadata], None, None]]:
    """
    Read frames from a video stream (file-like object).

    Args:
        stream: A file-like object containing video data, or an open input container.
            An open container is read from its current position, without rewinding or closing it.
        skip_frames: Number of initial frames to skip (for resume support).
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE", or "NONE").
        buffer_size: Size of PyAV's internal read buffer in bytes. Smaller values
            reduce latency when streaming (frames are decoded sooner), but increase
            the number of read syscalls. Default is 32768 (PyAV default).
        fps: Target frame rate. Drops frames (roughly uniformly, by timestamp)
            to approximate this rate; never duplicates, so a target at or
            above the source rate returns every frame. None keeps all frames.

        size: Resize native planes before RGB conversion, as in read_frames_exact.
        return_frame_metadata: Yield (RGB array, FrameMetadata) instead of just the array.
            Metadata retains native pts/time_base and the decoded index before skip/FPS selection.

    Returns:
        A tuple of (VideoMetadata, frame_generator).
        The generator yields RGB arrays (H, W, 3), or (array, FrameMetadata) pairs when requested.

    Raises:
        ValueError: If skip_frames is negative or too large. Raised at call
            time, before the stream is opened.
        TypeError: If skip_frames is not an integer (np.int64 is fine).
        RuntimeError: If the stream cannot be opened as a video, same as the
            file-path readers.

    Note:
        For streaming-friendly formats (WebM), frames are yielded as they're
        decoded without waiting for the complete file. For formats requiring
        seeking (MP4 with moov at end), the stream must be fully available.
        The generator owns the container: if it is discarded without ever
        being iterated, the container is closed only at garbage collection.

        When the container omits the average rate (browser-recorded WebM),
        the metadata's ``fps`` is recovered only for seekable input (an
        uploaded blob, BytesIO) — recovering it means decoding ahead and
        rewinding, which a pipe can't do, so a non-seekable stream reports
        ``fps=0.0``. Guard against it before dividing.
    """
    skip_frames = _nonnegative_index(skip_frames, "skip_frames")
    _validate_fps(fps)

    if size is not None and not isinstance(size, FrameSize):
        message = "size must be a FrameSize"
        raise TypeError(message)
    reused = isinstance(stream, av.container.InputContainer)
    seekable = not reused and getattr(stream, "seekable", lambda: False)()

    # _open_container rewinds reused containers; streaming must preserve their position.
    # A container opened here stays open until the returned generator closes.
    container = stream if reused else _open_video(stream, buffer_size=buffer_size)
    try:
        video = container.streams.video[0]
        if not reused:
            video.thread_type = thread_type

        if seekable and not container.streams.video[0].average_rate:
            # A no-rate file on a seekable stream (uploaded blob, BytesIO)
            # gets the same metadata as a path — the missing-rate recovery
            # needs rewinds a pipe can't do, and rewinds back to frame 0
            # when done. Gated on average_rate so normal streams don't pay
            # the recovery's extra passes.
            meta = video_metadata_from_container(container)
            first_frame = None
            decoded = container.decode(video=0)
        else:
            # The display-matrix rotation is only exposed per-frame — decode the
            # first frame eagerly for the metadata and hand it back through the
            # generator. The same generator must be reused for the remaining
            # frames: a fresh decode() after the eager read hits a flushed
            # frame-threaded decoder and raises EOFError on clips shorter than
            # the decoder delay (issue #18).
            decoded = container.decode(video=0)
            first_frame = next(decoded, None)
            rotation = first_frame.rotation if first_frame is not None else 0
            meta = video_metadata_from_container(container, rotation=rotation)
    except Exception:
        if not reused:
            container.close()
        raise

    # Built outside frame_generator so its closure doesn't capture (and pin)
    # the decoded first_frame (~3 MB at 1080p) for the whole read.
    if first_frame is not None:
        decoded = _prepend(first_frame, decoded)
    frames = islice(enumerate(decoded), skip_frames, None)
    if fps is not None:
        frames = _select_frames_by_fps(frames, fps, get_frame=operator.itemgetter(1))

    def frame_generator():
        try:
            yield from _stream_frames_to_rgb(frames, size, return_frame_metadata)
        finally:
            if not reused:
                container.close()

    return meta, frame_generator()
