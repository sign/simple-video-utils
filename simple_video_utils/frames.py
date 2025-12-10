import io
from collections.abc import Generator
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import VideoMetadata, _open_container, video_metadata_from_bytes


def _generate_frames(
    container: av.container.InputContainer,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generate RGB frames from a container.

    Args:
        container: Open PyAV container.
        start_frame: First frame index to yield (0-based).
        end_frame: Last frame index to yield (inclusive), or None for all.

    Yields:
        RGB numpy arrays (H, W, 3).
    """
    frame_index = 0
    for frame in container.decode(video=0):
        if frame_index < start_frame:
            frame_index += 1
            continue

        if end_frame is not None and frame_index > end_frame:
            break

        yield frame.to_ndarray(format='rgb24')
        frame_index += 1

def read_frames_exact(
    src: str,
    start_frame: int,
    end_frame: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Return frames [start_frame, end_frame] inclusive as RGB np.ndarrays.
    If end_frame is None, reads from start_frame to the end of the video.
    Uses PyAV for efficient frame extraction.
    """
    if end_frame is not None:
        assert end_frame >= start_frame >= 0, "invalid frame range"
    else:
        assert start_frame >= 0, "start_frame must be non-negative"

    with _open_container(src) as container:
        stream = container.streams.video[0]

        # Seek to approximate start position if not starting from beginning
        if start_frame > 0:
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            seek_time_sec = max(0, (start_frame - 30) / fps)
            # Convert seconds to stream time_base units
            seek_timestamp = int(seek_time_sec / float(stream.time_base))
            container.seek(seek_timestamp, stream=stream)

        yield from _generate_frames(container, start_frame, end_frame)


def read_frames_from_stream(
    stream: BinaryIO,
    skip_frames: int = 0,
) -> tuple[VideoMetadata, Generator[np.ndarray, None, None]]:
    """
    Read frames from a video stream (file-like object).

    Args:
        stream: A file-like object containing video data (e.g., uploaded file).
        skip_frames: Number of initial frames to skip (for resume support).

    Returns:
        A tuple of (VideoMetadata, frame_generator).
        The generator yields np.ndarray frames in RGB format (H, W, 3).

    Note:
        PyAV handles format detection and seeking automatically.
        Works with MP4, WebM, and other formats.
    """
    video_data = stream.read()
    meta = video_metadata_from_bytes(video_data)

    def frame_generator() -> Generator[np.ndarray, None, None]:
        """Generator that yields frames from the video data."""
        with _open_container(io.BytesIO(video_data)) as container:
            yield from _generate_frames(container, start_frame=skip_frames)

    return meta, frame_generator()
