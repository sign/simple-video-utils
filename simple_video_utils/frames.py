import operator
import sys
from collections.abc import Callable, Generator, Iterable
from fractions import Fraction
from functools import lru_cache
from itertools import islice, tee
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import (
    VideoMetadata,
    _open_container,
    _open_video,
    video_metadata,
    video_metadata_from_container,
)

# Below this frame size, swscale's threads=0 (auto, one slice job per CPU)
# loses to single-threaded: dispatch overhead exceeds the conversion itself —
# measured 2x slower at 320x240, a wash at 640x360, auto wins from ~960x540.
# Output is byte-identical at any thread count. The crossover is
# hardware-dependent (thread dispatch cost vs single-core throughput), so this
# is a calibration knob, not a derived value.
_SINGLE_THREAD_MAX_PIXELS = 300_000

def _rotation_graph(rotation: int, width: int, height: int, time_base: Fraction | None) -> av.filter.Graph:
    """
    Build a filter graph rotating rgb24 frames.

    rotation=90 maps to a counterclockwise transpose, matching ffmpeg
    autorotate (and np.rot90 k=1) pixel-exactly. Graphs are stateful
    (push/pull is a FIFO), so cache only per-read (the lru_cache in
    _frames_to_rgb) — a shared/module-level cache would interleave
    frames across concurrent reads.
    """
    graph = av.filter.Graph()
    if rotation == 180:
        filters = [graph.add("hflip"), graph.add("vflip")]
    else:
        filters = [graph.add("transpose", "cclock" if rotation == 90 else "clock")]
    # add_buffer without an explicit time_base is deprecated; the value only
    # affects pts interpretation, never pixels, and we only take arrays.
    source = graph.add_buffer(width=width, height=height, format="rgb24",
                              time_base=time_base or Fraction(1, 1000))
    graph.link_nodes(source, *filters, graph.add("buffersink"))
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
    # One-slot cache: built on the first rotated frame, reused for the whole
    # read, rebuilt only if stream geometry changes mid-read. Created here so
    # each read gets its own graph (see _rotation_graph on why).
    rotation_graph = lru_cache(maxsize=1)(_rotation_graph)
    for frame in frames:
        threads = 1 if frame.width * frame.height < _SINGLE_THREAD_MAX_PIXELS else 0
        rgb = reformatter.reformat(frame, format='rgb24', threads=threads)
        rotation = frame.rotation % 360
        if rotation and rotation % 90 == 0:
            graph = rotation_graph(rotation, rgb.width, rgb.height, frame.time_base)
            graph.push(rgb)
            rgb = graph.pull()
        # The frame's linesize may be padded past width*3, making to_ndarray
        # a non-contiguous view — consumers like MediaPipe and OpenCV reject
        # those, so copy (no-op when already contiguous).
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
    get_time: Callable = operator.attrgetter("time"),
) -> Generator:
    """
    Downsample to ~fps by keeping the first frame of each 1/fps bucket.

    Drop-only: when fps meets or exceeds the source rate every frame passes
    through unchanged — frames are never duplicated. Buckets are anchored at
    the first frame's timestamp, so selection stays uniform across VFR gaps
    and after seeks.

    ``get_time`` lets the same selection run over ``(index, frame)`` pairs
    (the return_indices path) — items are yielded as given.
    """
    origin = None
    last_bucket = None
    for frame in frames:
        time = get_time(frame)
        if time is None:
            continue
        if origin is None:
            origin = time
        # epsilon absorbs float error when a timestamp lands on a bucket edge
        bucket = int((time - origin) * fps + 1e-6)
        if bucket != last_bucket:
            yield frame
            last_bucket = bucket


def _indexed_frames_to_rgb(
    items: Iterable[tuple[int, av.VideoFrame]],
) -> Generator[tuple[np.ndarray, int], None, None]:
    """Convert (index, frame) pairs to (rgb_array, index) pairs."""
    frames, pairs = tee(items, 2)
    for rgb, (index, _) in zip(_frames_to_rgb(f for _, f in frames), pairs, strict=True):
        yield rgb, index


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
        end_time: Ending time in seconds, or None for end of video.
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE", or "NONE").
        fps: Target frame rate. Drops frames (roughly uniformly, by timestamp)
            to approximate this rate; never duplicates, so a target at or
            above the source rate returns every frame. None keeps all frames.
        return_indices: When True, yield ``(array, index)`` pairs, where
            ``index`` is the frame's absolute 0-based position in the whole
            video (the start offset is included, and fps drop-selection
            reports exactly the frames it kept). Default False: plain arrays.

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
    if not has_time_params:
        frame_range = _normalize_frame_range(start_frame, end_frame)

    # Inner generator so validation above raises at call time, not first next().
    def generate() -> Generator[np.ndarray, None, None]:
        with _open_container(src) as container:
            stream = container.streams.video[0]
            # A reused container's codec may already be open (e.g. metadata's
            # rotation probe decoded a frame) and PyAV forbids changing
            # thread_type then — the value set at first decode stays in effect.
            if not stream.codec_context.is_open:
                stream.thread_type = thread_type

            # Get FPS - required for all operations. Some containers
            # (browser-recorded WebM) omit the average rate; _count_and_rate
            # recovers the true cadence through the metadata layer — cached
            # per path, so repeated reads (e.g. slice_video's per-range
            # calls) pay the decode-derived rate once.
            # ponytail: a reused open_video container re-derives per call;
            # cache per container if no-rate files show up in that path.
            source_fps = float(stream.average_rate or 0) or _count_and_rate(src)[1]
            if not source_fps:
                msg = "Video has no FPS information"
                raise ValueError(msg)

            if has_time_params:
                target_start, target_end = _convert_time_to_frames(start_time, end_time, source_fps)
            else:
                target_start, target_end = frame_range

            did_seek = _seek_near(target_start, source_fps, stream, container)
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

            indexed = enumerate(frames, start=target_start)
            if fps is not None:
                indexed = _select_frames_by_fps(indexed, fps, get_time=lambda item: item[1].time)

            if return_indices:
                yield from _indexed_frames_to_rgb(indexed)
            else:
                yield from _frames_to_rgb(frame for _, frame in indexed)

    return generate()


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
        ValueError: If the iterable is empty, like np.stack.
    """
    chunk = size_hint or 64
    chunks = []
    buf = None
    count = 0
    for array in frames:
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
    Frame count and fps, header-first — metadata only when the header lacks one.

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
        total, fps = stream.frames, float(stream.average_rate or 0)
    return _fill_from_metadata(total, fps, lambda: video_metadata(src))


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
    total, fps = stream.frames, float(stream.average_rate or 0)
    return _fill_from_metadata(total, fps, lambda: video_metadata_from_container(src))


def read_frames_batched(
    src: str | av.container.InputContainer,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    thread_type: str = "AUTO",
    fps: float | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Read a frame range as one batched (N, H, W, 3) RGB uint8 array.

    ``stack_frames`` over ``read_frames_exact`` (same parameters), with the
    batch preallocated exactly: the requested range is converted to a frame
    count with the same helpers read_frames_exact uses, clamped to the
    clip's header frame count — 26-38% faster than stacking afterward.
    ``torch.from_numpy(...)`` wraps the result zero-copy; building the batch
    in torch directly measured 3x slower (per-op dispatch), so numpy is the
    fast path either way.

    With ``return_indices=True`` returns ``(frames, indices)``: a 1-D int64
    array of the same length as the batch, giving each frame's absolute
    0-based index in the whole video (the start offset is included, and fps
    drop-selection reports exactly the frames it kept).

    Raises:
        ValueError: Same call-time validation as read_frames_exact, and,
            like np.stack, if the range contains no frames.
    """
    # Created first so parameter validation raises before the metadata peek.
    frames = read_frames_exact(src, start_frame, end_frame, start_time, end_time, thread_type, fps,
                               return_indices=return_indices)

    total, source_fps = _count_and_rate(src)
    if start_time is not None or end_time is not None:
        start, end = _convert_time_to_frames(start_time, end_time, source_fps)
    else:
        start, end = _normalize_frame_range(start_frame, end_frame)
    if total:
        end = total - 1 if end is None else min(end, total - 1)
    hint = max(end - start + 1, 0) if end is not None else 0
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
    stream: BinaryIO,
    skip_frames: int = 0,
    thread_type: str = "AUTO",
    buffer_size: int = 32768, # PyAV default buffer size, can be reduced for lower latency when realtime streaming
    fps: float | None = None,
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
        fps: Target frame rate. Drops frames (roughly uniformly, by timestamp)
            to approximate this rate; never duplicates, so a target at or
            above the source rate returns every frame. None keeps all frames.

    Returns:
        A tuple of (VideoMetadata, frame_generator).
        The generator yields np.ndarray frames in RGB format (H, W, 3).

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

    seekable = getattr(stream, "seekable", lambda: False)()

    # Not _open_container: the returned generator owns the container, so it
    # must stay open after this function returns.
    container = _open_video(stream, buffer_size=buffer_size)
    try:
        for s in container.streams.video:
            s.thread_type = thread_type

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
        container.close()
        raise

    # Built outside frame_generator so its closure doesn't capture (and pin)
    # the decoded first_frame (~3 MB at 1080p) for the whole read.
    if first_frame is not None:
        decoded = _prepend(first_frame, decoded)
    frames = islice(decoded, skip_frames, None)
    if fps is not None:
        frames = _select_frames_by_fps(frames, fps)

    def frame_generator() -> Generator[np.ndarray, None, None]:
        try:
            yield from _frames_to_rgb(frames)
        finally:
            container.close()

    return meta, frame_generator()
