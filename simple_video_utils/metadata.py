import io
from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple

import av


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    nb_frames: int | None  # best effort: header, cross-checked against duration×fps, decoded on disagreement
    time_base: str | None
    duration: float | None  # seconds; None if the container header doesn't carry one
    rotation: int = 0  # display-matrix rotation in degrees; width/height already account for it


def _open_video(source: str | io.BytesIO, **av_kwargs) -> av.container.InputContainer:
    """Open a container, wrapping failures as RuntimeError('Failed to open video')."""
    try:
        # metadata_errors='replace': some files carry non-UTF-8 stream metadata
        # (e.g. handler_name in stray mp4s data tracks), which would otherwise
        # raise UnicodeDecodeError before the video stream is even reachable.
        return av.open(source, metadata_errors="replace", **av_kwargs)
    except Exception as e:
        msg = "Failed to open video"
        raise RuntimeError(msg) from e


@contextmanager
def _open_container(source: str | io.BytesIO | av.container.InputContainer):
    """
    Context manager yielding an open PyAV container.

    Paths and streams are opened here and closed on exit. An already-open
    container (see open_video) is passed through instead: it is rewound so
    every call reads from frame 0 regardless of what ran before, and left
    open for its owner to close — so reuse requires seekable input.
    """
    reused = isinstance(source, av.container.InputContainer)
    container = source if reused else _open_video(source)
    try:
        if reused:
            container.seek(0)
        yield container
    except Exception as e:
        msg = "Failed to open video"
        raise RuntimeError(msg) from e
    finally:
        if not reused:
            container.close()


@contextmanager
def open_video(source: str | io.BytesIO, thread_type: str = "AUTO"):
    """
    Open a video once and reuse the container across helper calls.

    The source-accepting helpers — video_metadata_from_container,
    count_frames, keyframe_indices, and the frames module's
    read_frames_exact / read_frames_batched — all accept the yielded
    container, so metadata and frame reads share a single av.open instead
    of paying one container open per call (issue #8).

    Each helper rewinds the container before reading, so call order doesn't
    matter — but the source must be seekable, and frame generators must be
    consumed one at a time (they share the container's decode position).

    Args:
        source: Path, URL, or file-like object to open.
        thread_type: PyAV thread type for decoding ("AUTO", "FRAME", "SLICE",
            or "NONE"), applied up front to every video stream. Needed here,
            not just on read_frames_exact/read_frames_batched, because PyAV
            forbids changing thread_type once a stream's codec is open — and
            the metadata rotation probe opens it on first use. Whatever this
            call sets is what the container decodes with for its whole
            lifetime; a thread_type passed to a later read call on the same
            container has no effect once the codec is already open. Callers
            that fork worker processes around decoding (e.g. a DataLoader)
            should pass "NONE" — an inherited AUTO-threaded decoder can
            deadlock post-fork.

    Example:
        with open_video("video.mp4") as video:
            meta = video_metadata_from_container(video)
            frames = list(read_frames_exact(video, start_frame=0, end_frame=10))
    """
    container = _open_video(source)
    try:
        # PyAV forbids changing thread_type once the codec opens, and the
        # metadata rotation probe opens it — set the decode threading policy
        # NOW or the container is locked into unthreaded decode for its whole
        # lifetime (measured ~1.2x slower window reads on 256px h264 at the
        # AUTO default).
        for stream in container.streams.video:
            stream.thread_type = thread_type
        yield container
    finally:
        container.close()


def _count_video_packets(container: av.container.InputContainer) -> int | None:
    """
    Count video packets by demuxing (no decoding), then rewind.

    Video codecs carry one frame per packet, so this approximates the frame
    count — but buggy files can carry trailing packets that never decode.
    Requires a seekable container; returns None if demuxing fails.
    """
    try:
        return sum(1 for packet in container.demux(video=0) if packet.size)
    except (av.FFmpegError, OSError):
        return None
    finally:
        container.seek(0)


def _count_decoded_frames(container: av.container.InputContainer) -> int | None:
    """Ground-truth frame count — the count half of _decoded_rate_and_count."""
    return _decoded_rate_and_count(container)[1]


def _best_effort_nb_frames(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    fps: float,
    duration: float | None,
    seekable: bool,
) -> int | None:
    """
    Do our best to report the true frame count (see issue #4).

    Container headers can lie: some MOV/MP4 files declare more frames than
    actually decode, and Matroska/WebM headers often omit the count entirely.
    Cross-check the cheap candidate (header, else packet count) against
    duration × fps; on agreement trust it, on disagreement decode for the
    ground truth. Non-seekable input can't rewind, so it gets the cheap
    signals only.
    """
    header = stream.frames if stream.frames > 0 else None
    derived = round(duration * fps) if duration and fps else None

    if not seekable:
        return header if header is not None else derived

    candidate = header if header is not None else _count_video_packets(container)
    if candidate is None:
        return derived
    if derived is None or abs(candidate - derived) <= 1:
        return candidate

    decoded = _count_decoded_frames(container)
    return decoded if decoded is not None else candidate


def _decoded_rate_and_count(
    container: av.container.InputContainer,
) -> tuple[float | None, int | None]:
    """
    Ground-truth frame rate and count by decoding the whole video stream.

    Browser-recorded (MediaRecorder) WebM often carries no rate hint at all —
    no DefaultDuration, irregular cluster timestamps — so PyAV's
    ``stream.average_rate`` comes back None (ffprobe: ``avg_frame_rate 0/0``).
    With no header rate, every cheap signal is suspect: packet counts include
    trailing packets that never decode (issue #4), and container.duration
    spans the longest stream, which audio can pad past the video. Decoding
    sidesteps both: N frames spanning their first and last timestamps give
    the true cadence as N-1 intervals over that span. Slow — O(stream
    duration) — but these files are the rare exception, and video_metadata
    caches the result per path. Requires a seekable container; rewinds it.
    Returns (None, None) if decoding fails; the rate alone is None when
    fewer than two frames carry timestamps.
    """
    first = last = None
    count = 0
    try:
        for frame in container.decode(video=0):
            count += 1
            if frame.time is None:
                continue
            if first is None:
                first = frame.time
            last = frame.time
    except (av.FFmpegError, OSError):
        return None, None
    finally:
        container.seek(0)
    if last is not None and last > first:
        return (count - 1) / (last - first), count
    return None, count


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
    seekable: bool | None = None,
) -> VideoMetadata:
    """
    Extract metadata from an open PyAV container.

    Width/height are reported in display orientation (rotation applied),
    matching the frames yielded by the frames module.

    Args:
        container: Open PyAV container.
        rotation: Display rotation in degrees if already known (e.g. from a
            decoded frame). When None, it is probed by decoding the first
            frame — seekable input only; a non-seekable container with no
            known rotation reports 0.
        seekable: Whether the container can rewind. Rewinding is what the
            rotation probe, the frame-count cross-check, and the
            missing-rate recovery need; when False the header values are
            trusted as-is. Defaults to ``rotation is None`` — the historical
            contract where passing a known rotation implied non-seekable
            input.
    """
    if seekable is None:
        seekable = rotation is None
    if seekable:
        # rewind so a container reused via open_video reads from frame 0 even
        # if a previous frame read left it mid-stream (packet counting would
        # otherwise undercount).
        container.seek(0)
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

    nb_frames = None
    if not fps and seekable:
        # Recover the missing rate before _best_effort_nb_frames: deriving it
        # from the packet count afterward would make the issue-#4 cross-check
        # (round(duration × fps) vs count) circular — true by construction.
        derived_fps, nb_frames = _decoded_rate_and_count(container)
        fps = derived_fps or 0.0
    if nb_frames is None:
        nb_frames = _best_effort_nb_frames(container, stream, fps, duration, seekable=seekable)

    if rotation is None:
        rotation = _probe_rotation(container) if seekable else 0
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




def count_frames(source: str | io.BytesIO | av.container.InputContainer) -> int:
    """
    Ground-truth frame count by decoding the entire video stream.

    Slow — O(stream duration). ``video_metadata(...).nb_frames`` is the
    best-effort answer and usually matches; use this when you need certainty
    regardless of what the container header claims.
    """
    with _open_container(source) as container:
        count = _count_decoded_frames(container)
        if count is None:
            msg = "Failed to decode video"
            raise RuntimeError(msg)
        return count


def keyframe_indices(source: str | io.BytesIO | av.container.InputContainer) -> list[int]:
    """
    Presentation-order frame indices of the keyframes (the GOP anchors).

    Demux-only — packets are inspected, nothing is decoded, so this is cheap
    even for long videos. Packets arrive in decode order; when timestamps are
    present they are re-sorted into presentation order (B-frames reorder the
    two). Like packet-based frame counting, this trusts the container: buggy
    files with trailing packets that never decode can shift indices near the
    tail.
    """
    with _open_container(source) as container:
        packets = [(p.pts if p.pts is not None else p.dts, p.is_keyframe)
                   for p in container.demux(video=0) if p.size]
    order = range(len(packets))
    if all(ts is not None for ts, _ in packets):
        order = sorted(order, key=lambda j: packets[j][0])
    return [i for i, j in enumerate(order) if packets[j][1]]


def video_metadata_from_bytes(data: bytes) -> VideoMetadata:
    """Return key video stream metadata from video bytes."""
    with _open_container(io.BytesIO(data)) as container:
        return video_metadata_from_container(container)


@lru_cache(maxsize=8)
def video_metadata(url_or_path: str) -> VideoMetadata:
    """Return key video stream metadata."""
    with _open_container(url_or_path) as container:
        return video_metadata_from_container(container)
