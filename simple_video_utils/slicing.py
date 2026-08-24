"""Cut a video into clips by time range.

When no pixel change is needed — the target ``size`` is None, or the source is
already ``size`` x ``size`` and unrotated — packets are stream-copied: fast and
lossless. The file still carries the lead-in from the keyframe at/before
``start`` (required for decoding), but an edit list hides it, so playback
starts at ``start``; a few frames past ``end`` may remain visible (B-frame
reordering). Otherwise
frames are decoded, center-cropped to a square, resized, and re-encoded.
"""

import io
import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from itertools import takewhile
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import _open_video, video_metadata


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _encode_clip(src: str, start: float, end: float, fps: float, size: int) -> bytes:
    frames = list(read_frames_exact(src, start_time=start, end_time=end))
    if not frames:
        return b""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as output:
        stream = output.add_stream("h264", rate=Fraction(fps).limit_denominator(1000))
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        # One reformatter per clip so the swscale context is built once, not
        # per frame (same fix as _frames_to_rgb) — byte-identical output.
        reformatter = av.video.reformatter.VideoReformatter()
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_center_crop_square(frame), format="rgb24")
            output.mux(stream.encode(reformatter.reformat(video_frame, width=size, height=size)))
        output.mux(stream.encode())
    return buffer.getvalue()


def _copy_packet(packet: av.Packet) -> av.Packet:
    copy = av.Packet(bytes(packet))
    copy.pts = packet.pts
    copy.dts = packet.dts
    copy.duration = packet.duration
    copy.time_base = packet.time_base
    copy.is_keyframe = packet.is_keyframe
    return copy


def _mux_packets(stream: av.VideoStream, packets: list[av.Packet], start: int) -> bytes:
    output = io.BytesIO()
    format_name = "webm" if stream.codec_context.name in {"vp8", "vp9"} else "mp4"
    with av.open(output, mode="w", format=format_name) as container:
        destination = container.add_stream_from_template(stream)
        for source in packets:
            packet = _copy_packet(source)
            packet.pts -= start
            packet.dts -= start
            packet.stream = destination
            container.mux(packet)
    return output.getvalue()


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
    """Yield one clip (bytes; MP4, or WebM for VP8/VP9 sources) per (start, end) range, in order.

    Yields lazily so a long slice list never holds every clip in memory. ``size``
    center-crops each frame to a square and resizes to ``size`` x ``size``; a
    source that is already that size (and unrotated) is stream-copied instead.
    Every slice must be within the video (``0 <= start <= end <= duration``);
    an out-of-range or empty slice raises ``ValueError``.

    Stream-copied clips keep the lead-in from the keyframe before ``start``
    and a few trailing frames past ``end``. In MP4 an edit list hides them, so
    players are unaffected, but readers that enumerate raw decoded frames —
    including ``read_frames_exact`` and ``read_frames_from_stream`` — see
    them; WebM has no edit list, so there even players see the lead-in.
    Pass ``size`` to force re-encoding when exact frames matter.
    """
    meta = video_metadata(src)
    should_copy = size is None or (meta.width == meta.height == size and meta.rotation == 0)

    for start, end in slices:
        if start < 0 or end <= start or (meta.duration is not None and end > meta.duration):
            msg = f"slice ({start}, {end}) out of range [0, {meta.duration}]"
            raise ValueError(msg)
        clip = _copy_clip(src, start, end) if should_copy else _encode_clip(src, start, end, meta.fps, size)
        if not clip:
            msg = f"slice ({start}, {end}) has no frames"
            raise ValueError(msg)
        yield clip


def slice_video_stream(
    stream: BinaryIO,
    duration: float = 0.5,
    buffer_size: int = 32768,  # PyAV default; reduce for lower latency on live sources
) -> Iterator[bytes]:
    """Yield one packet-copied clip per ``duration`` seconds, as the stream arrives.

    Each clip is yielded as soon as its window's packets have been read, so a
    live source produces a clip roughly every ``duration`` seconds (plus up to
    ``buffer_size`` bytes of read-ahead — lower it for lower latency). Nothing
    is re-encoded: every clip keeps the keyframe lead-in it needs to decode.
    MP4 clips hide the lead-in behind an edit list; WebM clips (VP8/VP9
    sources) have no edit list, so there it is visible content reaching back
    to the last keyframe. Memory holds the current window plus that lead-in —
    bounded by the source's keyframe interval, not the stream length.
    """
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)
    with _open_video(stream, buffer_size=buffer_size) as container:
        source = container.streams.video[0]
        packets: list[av.Packet] = []
        index = 0
        end = _stream_pts(duration, source)
        for packet in _iter_packets(container, source):
            while packet.dts > end:
                start = _stream_pts(index * duration, source)
                if clip := _window(packets, start):
                    yield _mux_packets(source, clip, start)
                index += 1
                end = _stream_pts((index + 1) * duration, source)
            packets.append(packet)
        start = _stream_pts(index * duration, source)
        if clip := _window(packets, start):
            yield _mux_packets(source, clip, start)
