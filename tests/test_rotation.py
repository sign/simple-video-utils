"""Regression tests for videos with a display-matrix rotation (e.g. phone recordings).

PyAV decodes frames in their stored orientation and does not apply the
rotation side data. These tests ensure frames are rotated to display
orientation and metadata reports display dimensions, matching the ffmpeg
CLI's autorotate behavior.
"""

import os
import subprocess
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_exact, read_frames_from_stream
from simple_video_utils.metadata import video_metadata, video_metadata_from_bytes


@pytest.fixture
def video_path():
    """Vertical phone-style video: stored as 640x360 landscape with rotation=90."""
    return str(Path(__file__).parent / "assets" / "rotated90.mp4")


@pytest.fixture
def video_bytes(video_path):
    """Load the rotated video as bytes."""
    return Path(video_path).read_bytes()


def ffmpeg_autorotated_frames(src: str, num_frames: int, width: int, height: int) -> np.ndarray:
    """Decode frames with the ffmpeg CLI, which applies the display rotation."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", src,
        "-frames:v", str(num_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(num_frames, height, width, 3)


def assert_frames_close(frame: np.ndarray, ref: np.ndarray, err_msg: str):
    """Assert frames match up to YUV->RGB rounding differences between ffmpeg builds.

    A wrong rotation direction produces a mean difference of ~70, so the
    tolerance still catches orientation errors.
    """
    assert frame.shape == ref.shape, err_msg
    diff = np.abs(frame.astype(int) - ref.astype(int))
    assert diff.max() <= 3, f"{err_msg}: max diff {diff.max()}"
    assert diff.mean() < 1, f"{err_msg}: mean diff {diff.mean():.2f}"


class TestRotatedVideo:
    """Tests using a 90°-rotated vertical video."""

    def test_metadata_reports_display_orientation(self, video_path):
        """Metadata width/height must be the display (portrait) dimensions."""
        meta = video_metadata(video_path)

        assert (meta.width, meta.height) == (360, 640)
        assert meta.rotation == 90
        assert meta.nb_frames == 30

    def test_metadata_from_bytes_reports_display_orientation(self, video_bytes):
        """Bytes-based metadata must match the path-based behavior."""
        meta = video_metadata_from_bytes(video_bytes)

        assert (meta.width, meta.height) == (360, 640)
        assert meta.rotation == 90

    def test_unrotated_video_has_zero_rotation(self):
        """A video without a display matrix reports rotation=0 and unchanged dims."""
        meta = video_metadata(str(Path(__file__).parent / "assets" / "example.mp4"))

        assert meta.rotation == 0

    def test_frames_are_rotated_to_display_orientation(self, video_path):
        """Decoded frames must come out portrait, not the stored landscape."""
        frames = list(read_frames_exact(video_path, 0, 4))

        assert len(frames) == 5
        assert all(frame.shape == (640, 360, 3) for frame in frames)
        # np.rot90 alone yields a non-contiguous view, which MediaPipe/OpenCV reject
        assert all(frame.flags["C_CONTIGUOUS"] for frame in frames)

    def test_frames_match_ffmpeg_autorotate(self, video_path):
        """Rotating the wrong way yields the same shape — compare pixels against ffmpeg."""
        frames = list(read_frames_exact(video_path, 0, 4))
        ref = ffmpeg_autorotated_frames(video_path, 5, width=360, height=640)

        for i, frame in enumerate(frames):
            assert_frames_close(frame, ref[i], f"Frame {i} differs from ffmpeg")

    def test_stream_reading_rotates(self, video_bytes, video_path):
        """The stream path must rotate frames and report display-oriented metadata."""
        meta, frames = read_frames_from_stream(BytesIO(video_bytes))
        ref = ffmpeg_autorotated_frames(video_path, 3, width=360, height=640)

        assert (meta.width, meta.height, meta.rotation) == (360, 640, 90)
        for i in range(3):
            assert_frames_close(next(frames), ref[i], f"Frame {i} differs from ffmpeg")

    def test_stream_reading_with_skip_frames(self, video_bytes, video_path):
        """skip_frames must still work with the eagerly-decoded first frame."""
        ref = ffmpeg_autorotated_frames(video_path, 3, width=360, height=640)
        _, frames = read_frames_from_stream(BytesIO(video_bytes), skip_frames=2)

        assert_frames_close(next(frames), ref[2], "Skipped-to frame differs from ffmpeg")

    def test_stream_reading_from_non_seekable_pipe(self, video_bytes):
        """Rotation must work on a truly non-seekable stream (BytesIO can seek; a pipe cannot)."""
        read_fd, write_fd = os.pipe()
        read_file = os.fdopen(read_fd, "rb")
        write_file = os.fdopen(write_fd, "wb")

        def writer():
            write_file.write(video_bytes)
            write_file.close()

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            meta, frames = read_frames_from_stream(read_file)
            frame_list = list(frames)
        finally:
            thread.join()
            read_file.close()

        assert (meta.width, meta.height, meta.rotation) == (360, 640, 90)
        assert len(frame_list) == 30
        assert all(frame.shape == (640, 360, 3) for frame in frame_list)

