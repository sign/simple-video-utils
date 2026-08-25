"""Regression tests comparing PyAV implementation against ffprobe/ffmpeg ground truth."""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_exact as pyav_read_frames_exact
from simple_video_utils.metadata import video_metadata as pyav_video_metadata


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    nb_frames: int | None
    time_base: str | None


@lru_cache(maxsize=8)
def ffprobe(url_or_path: str) -> VideoMetadata:
    """Return key video stream metadata using ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", url_or_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        msg = f"ffprobe failed: {e.stderr.strip()}"
        raise RuntimeError(msg) from e

    info = json.loads(result.stdout)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    num, den = map(int, v["avg_frame_rate"].split("/")) if "avg_frame_rate" in v else (0, 1)
    fps = num / den if den else 0.0
    nb = v.get("nb_frames", "")

    return VideoMetadata(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        nb_frames=int(nb) if nb.isdigit() else None,
        time_base=v.get("time_base"),
    )


@lru_cache(maxsize=8)
def ffmpeg_decode(src: str) -> np.ndarray:
    """Decode every video frame as RGB using ffmpeg."""
    meta = ffprobe(src)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, meta.height, meta.width, 3)


def ffmpeg_read_frames_exact(
    src: str,
    start_frame: int,
    end_frame: int | None = None,
) -> np.ndarray:
    """
    Return frames [start_frame, end_frame] inclusive as RGB np.ndarrays using ffmpeg.
    If end_frame is None, reads from start_frame to the end of the video.
    """
    if end_frame is not None:
        assert end_frame >= start_frame >= 0, "invalid frame range"
    else:
        assert start_frame >= 0, "start_frame must be non-negative"

    stop = None if end_frame is None else end_frame + 1
    return ffmpeg_decode(src)[start_frame:stop]


class TestRegressionAgainstFFmpeg:
    """Regression tests comparing PyAV implementation against ffmpeg/ffprobe."""

    @pytest.fixture
    def video_path(self):
        """Path to the example video file."""
        return str(Path(__file__).parent / "assets" / "example.mp4")

    def test_metadata_matches_ffprobe(self, video_path):
        """Test that our metadata extraction matches ffprobe output."""
        # Get metadata using ffprobe (ground truth)
        ffprobe_meta = ffprobe(video_path)

        # Get metadata using our implementation
        pyav_meta = pyav_video_metadata(video_path)

        # Compare all fields
        assert pyav_meta.width == ffprobe_meta.width, "Width mismatch"
        assert pyav_meta.height == ffprobe_meta.height, "Height mismatch"
        assert abs(pyav_meta.fps - ffprobe_meta.fps) < 0.01, "FPS mismatch"

        # nb_frames might be slightly different or missing, but should be close if both exist
        if pyav_meta.nb_frames is not None and ffprobe_meta.nb_frames is not None:
            assert abs(pyav_meta.nb_frames - ffprobe_meta.nb_frames) <= 1, "Frame count mismatch"

    def test_frames_match_ffmpeg_from_start(self, video_path):
        """Test that frames extracted from start match ffmpeg output."""
        start_frame = 0
        end_frame = 10

        # Extract using ffmpeg (ground truth)
        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, start_frame, end_frame))

        # Extract using our implementation
        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=start_frame, end_frame=end_frame))

        # Should have same number of frames
        assert len(pyav_frames) == len(ffmpeg_frames), (
            f"Frame count mismatch: PyAV={len(pyav_frames)}, FFmpeg={len(ffmpeg_frames)}"
        )

        # Every frame should be identical (pixel-perfect)
        for i, (pyav_frame, ffmpeg_frame) in enumerate(zip(pyav_frames, ffmpeg_frames, strict=False)):
            np.testing.assert_array_equal(
                pyav_frame,
                ffmpeg_frame,
                err_msg=f"Frame {i} differs between PyAV and FFmpeg",
            )

    def test_frames_match_ffmpeg_with_seeking(self, video_path):
        """Test that frames extracted with seeking match ffmpeg output."""
        start_frame = 50
        end_frame = 60

        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, start_frame, end_frame))
        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=start_frame, end_frame=end_frame))

        assert len(pyav_frames) == len(ffmpeg_frames), (
            f"Frame count mismatch: PyAV={len(pyav_frames)}, FFmpeg={len(ffmpeg_frames)}"
        )

        for i, (pyav_frame, ffmpeg_frame) in enumerate(zip(pyav_frames, ffmpeg_frames, strict=True)):
            np.testing.assert_array_equal(
                pyav_frame,
                ffmpeg_frame,
                err_msg=f"Frame {start_frame + i} differs between PyAV and FFmpeg",
            )

    def test_frames_match_ffmpeg_time_based(self, video_path):
        """Test that time-based extraction matches ffmpeg frame-based output."""
        # Get FPS to convert time to frames
        meta = pyav_video_metadata(video_path)
        fps = meta.fps

        # Test 1-2 seconds
        start_time = 1.0
        end_time = 2.0
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Extract using ffmpeg with frame indices (ground truth)
        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, start_frame, end_frame))

        # Extract using our time-based implementation
        pyav_frames = list(pyav_read_frames_exact(video_path, start_time=start_time, end_time=end_time))

        # Should have same number of frames
        assert len(pyav_frames) == len(ffmpeg_frames), (
            f"Frame count mismatch: PyAV={len(pyav_frames)}, FFmpeg={len(ffmpeg_frames)}"
        )

        # Every frame should be identical
        for i, (pyav_frame, ffmpeg_frame) in enumerate(zip(pyav_frames, ffmpeg_frames, strict=False)):
            actual_frame_num = start_frame + i
            np.testing.assert_array_equal(
                pyav_frame,
                ffmpeg_frame,
                err_msg=f"Frame {actual_frame_num} ({start_time + i/fps:.3f}s) differs between PyAV and FFmpeg",
            )

    def test_single_frame_matches_ffmpeg(self, video_path):
        """Test that single frame extraction matches ffmpeg."""
        frame_idx = 42

        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, frame_idx, frame_idx))
        assert len(ffmpeg_frames) == 1

        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=frame_idx, end_frame=frame_idx))
        assert len(pyav_frames) == 1

        np.testing.assert_array_equal(
            pyav_frames[0],
            ffmpeg_frames[0],
            err_msg=f"Frame {frame_idx} differs between PyAV and FFmpeg",
        )
