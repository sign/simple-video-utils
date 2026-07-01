"""Cut a video into clips by time range, optionally transforming to a square size."""

import io
from collections.abc import Sequence
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


def _encode_clip(frames: Sequence[np.ndarray], fps: float, size: int | None, codec: str) -> bytes:
    if size is not None:
        out_width = out_height = size
    elif len(frames):
        out_height, out_width = frames[0].shape[:2]
    else:
        return b""

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        stream.width = out_width
        stream.height = out_height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            if size is not None:
                frame = _center_crop_square(frame)
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            if size is not None:
                video_frame = video_frame.reformat(width=size, height=size)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def slice_video(
    src: str,
    slices: Sequence[tuple[float, float]],
    size: int | None = None,
    codec: str = "h264",
) -> list[bytes]:
    """Cut ``src`` into clips, one per (start, end) second range, as encoded MP4 bytes.

    Args:
        src: Path or URL to the source video.
        slices: (start, end) second ranges; end is exclusive of the next frame.
        size: If set, each frame is center-cropped to a square and resized to
            ``size`` x ``size`` (e.g. 256 for models that expect square input).
            If None, clips keep the source resolution.
        codec: Output video codec.

    Returns:
        One ``bytes`` MP4 per slice, in the same order (empty ``bytes`` for a
        slice that contains no frames).
    """
    fps = video_metadata(src).fps
    return [
        _encode_clip(list(read_frames_exact(src, start_time=start, end_time=end)), fps, size, codec)
        for start, end in slices
    ]
