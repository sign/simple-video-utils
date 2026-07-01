"""Cut a video into clips by time range.

Without a target ``size`` this stream-copies packets (no re-encode): fast and
lossless, but cuts land on keyframe boundaries, so a clip may include a little
footage before the requested start. With ``size`` set, frames are decoded,
center-cropped to a square, resized, and re-encoded.
"""

import io
from collections.abc import Iterator, Sequence
from fractions import Fraction

import av
import numpy as np

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import _open_container, video_metadata


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _encode_clip(frames: Sequence[np.ndarray], fps: float, size: int, codec: str) -> bytes:
    if not frames:
        return b""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_center_crop_square(frame), format="rgb24")
            video_frame = video_frame.reformat(width=size, height=size)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _copy_clip(src: str, start: float, end: float) -> bytes:
    """Remux [start, end] seconds without re-encoding. Cuts on the keyframe at/before start."""
    with av.open(src) as source:
        in_stream = source.streams.video[0]
        time_base = in_stream.time_base
        if in_stream.duration and start >= in_stream.duration * time_base:
            return b""

        buffer = io.BytesIO()
        first_dts = None
        # packet.pts is on the stream's absolute timeline, which may not start at 0.
        origin = in_stream.start_time or 0
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
        return buffer.getvalue() if first_dts is not None else b""


def _already_square(src: str, size: int) -> bool:
    with _open_container(src) as container:
        stream = container.streams.video[0]
        return stream.codec_context.width == size and stream.codec_context.height == size


def _copy_slices(src: str, slices: Sequence[tuple[float, float]]) -> Iterator[bytes]:
    for start, end in slices:
        yield _copy_clip(src, start, end)


def _encode_slices(src: str, slices: Sequence[tuple[float, float]], size: int, codec: str) -> Iterator[bytes]:
    fps = video_metadata(src).fps
    for start, end in slices:
        yield _encode_clip(list(read_frames_exact(src, start_time=start, end_time=end)), fps, size, codec)


def slice_video(
    src: str,
    slices: Sequence[tuple[float, float]],
    size: int | None = None,
    codec: str = "h264",
) -> Iterator[bytes]:
    """Cut ``src`` into clips, one per (start, end) second range, as encoded MP4 bytes.

    Yields one clip at a time so a long list of slices doesn't hold every clip in
    memory at once.

    Args:
        src: Path or URL to the source video.
        slices: (start, end) second ranges.
        size: If set, frames are center-cropped to a square and resized to
            ``size`` x ``size``. When the source is already ``size`` x ``size``
            (or ``size`` is None) packets are stream-copied — faster and lossless,
            but the cut snaps to the keyframe at or before ``start``.
        codec: Output codec, used only when re-encoding.

    Yields:
        One ``bytes`` MP4 per slice, in order (empty ``bytes`` for a slice with
        no frames).
    """
    for start, end in slices:
        if end < start:
            msg = f"slice end {end} is before start {start}"
            raise ValueError(msg)

    if size is None or _already_square(src, size):
        return _copy_slices(src, slices)
    return _encode_slices(src, slices, size, codec)
