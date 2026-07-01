"""Cut a video into clips by time range.

When no pixel change is needed — the target ``size`` is None, or the source is
already ``size`` x ``size`` and unrotated — packets are stream-copied: fast and
lossless, but the cut snaps to the keyframe at or before ``start``. Otherwise
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


def _encode_clip(src: str, start: float, end: float, fps: float, size: int, codec: str) -> bytes:
    frames = list(read_frames_exact(src, start_time=start, end_time=end))
    if not frames:
        msg = f"slice {start}-{end}s has no frames"
        raise ValueError(msg)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as output:
        stream = output.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_center_crop_square(frame), format="rgb24")
            output.mux(stream.encode(video_frame.reformat(width=size, height=size)))
        output.mux(stream.encode())
    return buffer.getvalue()


def _copy_clip(src: str, start: float, end: float) -> bytes:
    """Remux [start, end] seconds without re-encoding, cutting on the keyframe at/before start."""
    with av.open(src) as source:
        in_stream = source.streams.video[0]
        time_base = in_stream.time_base
        origin = in_stream.start_time or 0  # pts is on the stream's absolute timeline
        buffer = io.BytesIO()
        first_dts = None
        with av.open(buffer, mode="w", format="mp4") as output:
            out_stream = output.add_stream_from_template(in_stream)
            source.seek(int(start / time_base) + origin, stream=in_stream, backward=True)
            end_pts = end / time_base + origin
            for packet in source.demux(in_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                if packet.pts > end_pts:
                    break
                if first_dts is None:
                    first_dts = packet.dts
                packet.pts -= first_dts
                packet.dts -= first_dts
                packet.stream = out_stream
                output.mux(packet)
        if first_dts is None:
            msg = f"slice {start}-{end}s has no frames"
            raise ValueError(msg)
        return buffer.getvalue()


def slice_video(
    src: str,
    slices: Sequence[tuple[float, float]],
    size: int | None = None,
    codec: str = "h264",
) -> Iterator[bytes]:
    """Yield one MP4 clip (bytes) per (start, end) second range, in order.

    Yields lazily so a long slice list never holds every clip in memory. ``size``
    center-crops each frame to a square and resizes to ``size`` x ``size``; a
    source that is already that size (and unrotated) is stream-copied instead.
    Every slice must be within the video (``0 <= start <= end <= duration``);
    an out-of-range slice raises ``ValueError``.
    """
    meta = video_metadata(src)
    should_copy = size is None or (meta.width == meta.height == size and meta.rotation == 0)

    for start, end in slices:
        if start < 0 or end < start or (meta.duration is not None and end > meta.duration):
            msg = f"slice ({start}, {end}) out of range [0, {meta.duration}]"
            raise ValueError(msg)
        yield _copy_clip(src, start, end) if should_copy else _encode_clip(src, start, end, meta.fps, size, codec)
