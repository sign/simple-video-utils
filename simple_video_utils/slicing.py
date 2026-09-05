"""Cut a video into clips by time range.

When no pixel change is needed — the target ``size`` is None, or the source is
already ``size`` x ``size`` and unrotated — packets are stream-copied: fast and
lossless. The file still carries the lead-in from the keyframe at/before
``start`` (required for decoding), but an edit list hides it, so playback
starts at ``start``; a few frames past ``end`` may remain visible (B-frame
reordering). Otherwise
frames are decoded, center-cropped to a square, resized, and re-encoded.

Every clip is a streamable MP4 (``moov`` before ``mdat``): a progressive reader
— a pipe, a socket — can demux it without seeking.
"""

import math
import tempfile
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from itertools import takewhile
from typing import BinaryIO, NamedTuple

import av
import numpy as np

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import _open_video, video_metadata

# MP4 output hides each clip's keyframe lead-in behind an edit list, keeping
# packet copy frame-accurate for players — but only for codecs MP4 can carry.
# Everything else is re-encoded so clips stay exact: WebM has no edit list (a
# copied VP8/VP9 clip would visibly start at the previous keyframe), and PyAV
# can't mux VP9 into MP4 anyway. The ideal source is H.264 in a streamable
# MP4 (faststart/fragmented) or MPEG-TS: opens without seeking and always
# takes the lossless copy path.
_MP4_COPY_CODECS = {"h264", "hevc", "av1", "mpeg4"}


class Clip(NamedTuple):
    """One window of a sliced stream, and where it sits on the source timeline.

    ``start`` is what a consumer cannot recover on its own: windows that yield
    no packets are skipped, so counting the clips it receives does not tell it
    how far into the video each one begins.
    """

    start: float  # seconds from the start of the stream
    data: bytes


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _mux_mp4(write: Callable[[av.container.OutputContainer], None]) -> bytes:
    """Run ``write`` against an MP4 muxer and return the file as streamable bytes.

    libav writes ``moov`` last, so a reader that cannot seek (a pipe) has to
    buffer the whole clip before it finds the stream headers, and fails once
    the clip outgrows its probe buffer. ``faststart`` moves ``moov`` to the
    front on close by rewriting the file through its path — it cannot target
    a BytesIO, hence the temp file. Fragmented MP4 would need no rewrite but
    drops the edit list that hides a copied clip's keyframe lead-in.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
        with av.open(file.name, mode="w", format="mp4", options={"movflags": "+faststart"}) as container:
            write(container)
        file.seek(0)
        return file.read()


def _encode_clip(src: str, start: float, end: float, fps: float, size: int | None) -> bytes:
    frames = list(read_frames_exact(src, start_time=start, end_time=end))
    if not frames:
        return b""
    height, width = (size, size) if size else frames[0].shape[:2]

    def write(output: av.container.OutputContainer) -> None:
        stream = output.add_stream("h264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        # One reformatter per clip so the swscale context is built once, not
        # per frame (same fix as _frames_to_rgb) — byte-identical output.
        reformatter = av.video.reformatter.VideoReformatter()
        for frame in frames:
            array = _center_crop_square(frame) if size else frame
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            output.mux(stream.encode(reformatter.reformat(video_frame, width=width, height=height)))
        output.mux(stream.encode())

    return _mux_mp4(write)


def _mux_packets(stream: av.VideoStream, packets: list[av.Packet], start: int) -> bytes:
    def write(container: av.container.OutputContainer) -> None:
        destination = container.add_stream_from_template(stream)
        # mux a copy: the source packets are shared between clips (the keyframe
        # lead-in), so rebasing them in place would corrupt the next clip
        for source in packets:
            packet = av.Packet(bytes(source))
            packet.pts = source.pts - start
            packet.dts = source.dts - start
            packet.duration = source.duration
            packet.time_base = source.time_base
            packet.is_keyframe = source.is_keyframe
            packet.stream = destination
            container.mux(packet)

    return _mux_mp4(write)


def _stream_pts(seconds: float, stream: av.VideoStream) -> int:
    """``seconds`` on the stream's absolute pts timeline.

    round, not int: a frame landing exactly on a boundary must not be
    excluded by float noise in the division.
    """
    return round(seconds / stream.time_base) + (stream.start_time or 0)


def _window(packets: list[av.Packet], start: int) -> list[av.Packet]:
    """Trim ``packets`` (in place) to the keyframe lead-in needed to decode from
    ``start`` and return the clip, or [] when nothing reaches ``start``."""
    keyframe = next(
        (index for index in reversed(range(len(packets)))
         if packets[index].is_keyframe and packets[index].pts <= start),
        0,
    )
    del packets[:keyframe]
    return packets if any(packet.pts >= start for packet in packets) else []


def _iter_packets(container: av.container.InputContainer, stream: av.VideoStream) -> Iterator[av.Packet]:
    for packet in container.demux(stream):
        if packet.pts is not None and packet.dts is not None and packet.size:
            yield packet


def _copy_clip(src: str, start: float, end: float) -> bytes:
    """
    Remux [start, end] seconds without re-encoding.

    The copied packets start on the keyframe at/before ``start`` (the lead-in
    is needed to decode) and may run a few frames past ``end``: packets arrive
    in decode order, so a B-frame with pts <= end can follow a packet with
    pts > end — cutting on pts would drop it. Cutting on dts (monotonic,
    dts <= pts) keeps every frame in range at the cost of a few trailing ones.

    Timestamps are rebased so ``start`` is t=0, putting the lead-in at
    negative pts — the mp4 muxer records that as an edit list, so players
    begin playback at ``start`` and the reported duration excludes the
    lead-in. Consumers that enumerate raw decoded frames still see the
    lead-in (with pts < 0); only presentation skips it.
    """
    with _open_video(src) as source:
        in_stream = source.streams.video[0]
        start_pts = _stream_pts(start, in_stream)
        end_pts = _stream_pts(end, in_stream)
        source.seek(start_pts, stream=in_stream, backward=True)
        packets = list(takewhile(lambda p: p.dts <= end_pts, _iter_packets(source, in_stream)))
        clip = _window(packets, start_pts)
        return _mux_packets(in_stream, clip, start_pts) if clip else b""


def slice_video(
    src: str,
    slices: Sequence[tuple[float, float]],
    size: int | None = None,
) -> Iterator[bytes]:
    """Yield one MP4 clip (bytes) per (start, end) second range, in order.

    Yields lazily so a long slice list never holds every clip in memory. ``size``
    center-crops each frame to a square and resizes to ``size`` x ``size``; a
    source that is already that size (and unrotated) is stream-copied instead.
    Every slice must be within the video (``0 <= start <= end <= duration``);
    an out-of-range or empty slice raises ``ValueError``.

    Stream-copied clips keep the lead-in from the keyframe before ``start``
    and a few trailing frames past ``end``; an edit list hides them, so
    players are unaffected, but readers that enumerate raw decoded frames —
    including ``read_frames_exact`` and ``read_frames_from_stream`` — see
    them. Pass ``size`` to force re-encoding when exact frames matter.
    Sources whose codec MP4 can't carry (e.g. VP8/VP9 WebM) are always
    re-encoded, at the source resolution when ``size`` is None.
    """
    meta = video_metadata(src)
    with _open_video(src) as container:
        codec = container.streams.video[0].codec_context.name
    should_copy = codec in _MP4_COPY_CODECS and (
        size is None or (meta.width == meta.height == size and meta.rotation == 0)
    )

    for start, end in slices:
        if start < 0 or end <= start or (meta.duration is not None and end > meta.duration):
            msg = f"slice ({start}, {end}) out of range [0, {meta.duration}]"
            raise ValueError(msg)
        clip = _copy_clip(src, start, end) if should_copy else _encode_clip(src, start, end, meta.fps, size)
        if not clip:
            msg = f"slice ({start}, {end}) has no frames"
            raise ValueError(msg)
        yield clip


def _copy_windows(container: av.container.InputContainer, source: av.VideoStream,
                  duration: float) -> Iterator[Clip]:
    """Packet-copied windows: lossless, keyframe lead-in hidden by an edit list."""
    packets: list[av.Packet] = []
    index = 0
    end = _stream_pts(duration, source)
    for packet in _iter_packets(container, source):
        while packet.dts > end:
            start = _stream_pts(index * duration, source)
            if clip := _window(packets, start):
                yield Clip(index * duration, _mux_packets(source, clip, start))
            index += 1
            end = _stream_pts((index + 1) * duration, source)
        packets.append(packet)
    start = _stream_pts(index * duration, source)
    if clip := _window(packets, start):
        yield Clip(index * duration, _mux_packets(source, clip, start))


def _encode_window(frames: list[av.VideoFrame], source: av.VideoStream, start: int) -> bytes:
    def write(container: av.container.OutputContainer) -> None:
        # rate is nominal metadata; the rebased frame pts carry the real timing,
        # so a source without an average_rate only gets a mislabeled fps
        stream = container.add_stream("h264", rate=source.average_rate or 30)
        stream.width, stream.height, stream.pix_fmt = source.width, source.height, "yuv420p"
        reformatter = av.video.reformatter.VideoReformatter()
        for frame in frames:
            video_frame = reformatter.reformat(frame, format="yuv420p")
            video_frame.pts = frame.pts - start
            video_frame.time_base = frame.time_base
            video_frame.pict_type = 0  # let the encoder choose frame types
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())

    return _mux_mp4(write)


def _encode_windows(container: av.container.InputContainer, source: av.VideoStream,
                    duration: float) -> Iterator[Clip]:
    """Frame-exact windows for codecs MP4 can't carry: decode and re-encode."""
    frames: list[av.VideoFrame] = []
    index = 0
    end = _stream_pts(duration, source)
    for frame in container.decode(source):
        while frame.pts >= end:
            if frames:
                yield Clip(index * duration,
                           _encode_window(frames, source, _stream_pts(index * duration, source)))
                frames = []
            index += 1
            end = _stream_pts((index + 1) * duration, source)
        frames.append(frame)
    if frames:
        yield Clip(index * duration,
                   _encode_window(frames, source, _stream_pts(index * duration, source)))


def slice_video_stream(
    stream: BinaryIO,
    duration: float = 0.5,
    buffer_size: int = 32768,  # PyAV default; reduce for lower latency on live sources
) -> Iterator[Clip]:
    """Yield one ``Clip`` per ``duration`` seconds, as the stream arrives.

    Each clip is yielded as soon as its window has been read, so a live source
    produces a clip roughly every ``duration`` seconds (plus up to
    ``buffer_size`` bytes of read-ahead — lower it for lower latency). Sources
    whose codec MP4 can carry are packet-copied: lossless, keeping the keyframe
    lead-in each clip needs to decode, hidden from playback by an edit list.
    Anything else (e.g. VP8/VP9 WebM) is decoded and re-encoded into exact
    windows instead — see ``_MP4_COPY_CODECS``. Memory holds one window plus,
    when copying, the lead-in back to the last keyframe.

    Each clip carries its ``start`` on the source timeline, which a consumer
    cannot derive from the order it receives them: a window with no packets
    yields nothing while the timeline still advances past it.
    """
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)
    with _open_video(stream, buffer_size=buffer_size) as container:
        source = container.streams.video[0]
        windows = _copy_windows if source.codec_context.name in _MP4_COPY_CODECS else _encode_windows
        yield from windows(container, source, duration)
