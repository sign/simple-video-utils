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

    @pytest.mark.parametrize(("start_frame", "end_frame"), [(0, 10), (50, 60), (42, 42)])
    def test_frames_match_ffmpeg(self, video_path, start_frame, end_frame):
        """Test that frame-index extraction matches ffmpeg output."""
        ffmpeg_frames = ffmpeg_decode(video_path)[start_frame : end_frame + 1]
        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=start_frame, end_frame=end_frame))

        np.testing.assert_array_equal(pyav_frames, ffmpeg_frames)

    def test_frames_match_ffmpeg_time_based(self, video_path):
        """Test that time-based extraction matches ffmpeg frame-based output."""
        start_time = 1.0
        end_time = 2.0
        fps = pyav_video_metadata(video_path).fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        ffmpeg_frames = ffmpeg_decode(video_path)[start_frame : end_frame + 1]
        pyav_frames = list(pyav_read_frames_exact(video_path, start_time=start_time, end_time=end_time))

        np.testing.assert_array_equal(pyav_frames, ffmpeg_frames)
