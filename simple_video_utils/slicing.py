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
from collections.abc import Iterator, Sequence
from fractions import Fraction

import av
import numpy as np

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import video_metadata


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
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_center_crop_square(frame), format="rgb24")
            output.mux(stream.encode(video_frame.reformat(width=size, height=size)))
        output.mux(stream.encode())
    return buffer.getvalue()


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
        time_base = in_stream.time_base
        origin = in_stream.start_time or 0  # pts is on the stream's absolute timeline
        start_pts = int(start / time_base) + origin
        end_pts = end / time_base + origin
        buffer = io.BytesIO()
        muxed = False
        with av.open(buffer, mode="w", format="mp4") as output:
            out_stream = output.add_stream_from_template(in_stream)
            source.seek(start_pts, stream=in_stream, backward=True)
            for packet in source.demux(in_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                if packet.dts > end_pts:
                    break
                packet.pts -= start_pts
                packet.dts -= start_pts
                packet.stream = out_stream
                output.mux(packet)
                muxed = True
        return buffer.getvalue() if muxed else b""


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
