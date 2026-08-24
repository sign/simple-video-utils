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


def _keyframe_index(packets: list[av.Packet], start: int) -> int:
    """Index of the last keyframe at/before ``start`` — the decode entry point."""
    return max(
        (index for index, packet in enumerate(packets)
         if packet.is_keyframe and packet.pts <= start),
        default=0,
    )


def _iter_packets(container: av.InputContainer, stream: av.VideoStream) -> Iterator[av.Packet]:
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
    with av.open(src) as source:
        in_stream = source.streams.video[0]
        origin = in_stream.start_time or 0  # pts is on the stream's absolute timeline
        # round, not int: a frame landing exactly on a boundary must not be
        # excluded by float noise in the division
        start_pts = round(start / in_stream.time_base) + origin
        end_pts = round(end / in_stream.time_base) + origin
        source.seek(start_pts, stream=in_stream, backward=True)
        packets = []
        for packet in _iter_packets(source, in_stream):
            packets.append(packet)
            if packet.dts > end_pts:
                break
        selected = [p for p in packets[_keyframe_index(packets, start_pts):] if p.dts <= end_pts]
        return _mux_packets(in_stream, selected, start_pts) if selected else b""


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


def slice_video_stream(stream: BinaryIO, duration: float = 0.5) -> Iterator[bytes]:
    """Yield one packet-copied clip per ``duration`` seconds, as the stream arrives.

    Each clip is yielded as soon as its window's packets have been read, so a
    live source produces a clip roughly every ``duration`` seconds. Nothing is
    re-encoded: like ``slice_video``'s copy path, every clip keeps the keyframe
    lead-in it needs to decode, hidden from playback by an edit list. Memory
    holds only the current window plus that lead-in, never the whole stream.
    """
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)
    with _open_video(stream) as container:
        source = container.streams.video[0]
        origin = source.start_time or 0
        packets: list[av.Packet] = []
        index = 0

        def boundary(number: int) -> int:
            return round(number * duration / source.time_base) + origin

        start, end = boundary(0), boundary(1)
        for packet in _iter_packets(container, source):
            packets.append(packet)
            while packet.dts > end:
                del packets[: _keyframe_index(packets, start)]
                clip = [item for item in packets if item.dts <= end]
                if any(item.pts >= start for item in clip):
                    yield _mux_packets(source, clip, start)
                index += 1
                start, end = boundary(index), boundary(index + 1)

        del packets[: _keyframe_index(packets, start)]
        if any(item.pts >= start for item in packets):
            yield _mux_packets(source, packets, start)
