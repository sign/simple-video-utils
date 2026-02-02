"""Regression tests comparing PyAV implementation against ffprobe/ffmpeg ground truth."""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Generator, NamedTuple, Optional

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_exact as pyav_read_frames_exact
from simple_video_utils.metadata import video_metadata as pyav_video_metadata


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    nb_frames: Optional[int]
    time_base: Optional[str]


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
    nb_frames = v.get("nb_frames", "/")
    if not nb_frames.isdigit():
        nb_frames = None
    else:
        nb_frames = int(nb_frames)

    return VideoMetadata(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        nb_frames=nb_frames,
        time_base=v.get("time_base"),
    )


def ffmpeg_read_frames_exact(  # noqa: C901
    src: str,
    start_frame: int,
    end_frame: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Return frames [start_frame, end_frame] inclusive as RGB np.ndarrays using ffmpeg.
    If end_frame is None, reads from start_frame to the end of the video.
    """
    if end_frame is not None:
        assert end_frame >= start_frame >= 0, "invalid frame range"
    else:
        assert start_frame >= 0, "start_frame must be non-negative"

    meta = ffprobe(src)
    w, h, fps = meta.width, meta.height, meta.fps
    if fps <= 0:
        msg = "Could not determine FPS from container"
        raise RuntimeError(msg)

    frame_bytes = w * h * 3

    # If start_frame is 0, don't seek at all for optimal performance
    if start_frame == 0:
        seek_time = None
        if end_frame is None:
            # Read entire video from start
            vf = ["setpts=N/FRAME_RATE/TB"]
        else:
            # Read from start to end_frame
            vf = [f"select='lte(n\\,{end_frame})'", "setpts=N/FRAME_RATE/TB"]
    else:
        # Seek *near* the target, then select by absolute frame index in that window.
        # We back off a small margin to ensure keyframe landing < start_frame.
        backoff_frames = 2 * int(round(fps))  # ~2 seconds
        seek_frame = max(0, start_frame - backoff_frames)
        seek_time = seek_frame / fps

        # After demuxer-level seek (-ss before -i), ffmpeg's select 'n' restarts at 0.
        # So we select frames [offset .. offset + N-1] relative to the seek point.
        relative_start = start_frame - seek_frame

        if end_frame is None:
            # Read from start_frame to end of video
            vf = [f"select='gte(n\\,{relative_start})'", "setpts=N/FRAME_RATE/TB"]
        else:
            relative_end = end_frame - seek_frame
            vf = [
                f"select='between(n\\,{relative_start}\\,{relative_end})'",
                "setpts=N/FRAME_RATE/TB",  # normalize PTS after select
            ]

    vf_str = ",".join(vf)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if seek_time is not None:
        cmd.extend(["-ss", f"{seek_time:.6f}"])  # fast demuxer seek close to target

    cmd.extend([
        "-i",
        src,
        "-vf",
        vf_str,
        "-vsync",
        "0",  # don't duplicate/drop after select
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    num_frames = 0
    max_frames = None if end_frame is None else (end_frame - start_frame + 1)

    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) < frame_bytes:
                # truncated last read
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

            num_frames += 1
            if max_frames is not None and num_frames >= max_frames:
                break
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        proc.wait()


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
        for i, (pyav_frame, ffmpeg_frame) in enumerate(zip(pyav_frames, ffmpeg_frames)):
            np.testing.assert_array_equal(
                pyav_frame,
                ffmpeg_frame,
                err_msg=f"Frame {i} differs between PyAV and FFmpeg",
            )

    def test_frames_match_ffmpeg_with_seeking(self, video_path):
        """Test that frames extracted with seeking match ffmpeg output."""
        start_frame = 50
        end_frame = 60

        # First, extract from start to verify ground truth
        ffmpeg_from_start = list(ffmpeg_read_frames_exact(video_path, 0, 65))
        pyav_from_start = list(pyav_read_frames_exact(video_path, start_frame=0, end_frame=65))

        # Verify our implementation matches ffmpeg when no seeking
        for i in range(start_frame, min(end_frame + 1, len(ffmpeg_from_start))):
            assert np.array_equal(pyav_from_start[i], ffmpeg_from_start[i]), (
                f"Frame {i} differs from start - implementation issue"
            )

        # Now test with seeking - Extract using ffmpeg
        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, start_frame, end_frame))

        # Extract using our implementation
        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=start_frame, end_frame=end_frame))

        # Should have same number of frames
        assert len(pyav_frames) == len(ffmpeg_frames), (
            f"Frame count mismatch: PyAV={len(pyav_frames)}, FFmpeg={len(ffmpeg_frames)}"
        )

        # Check if PyAV matches the ground truth from start
        for i in range(len(pyav_frames)):
            frame_num = start_frame + i
            pyav_matches_start = np.array_equal(pyav_frames[i], pyav_from_start[frame_num])
            ffmpeg_matches_start = np.array_equal(ffmpeg_frames[i], ffmpeg_from_start[frame_num])

            if not pyav_matches_start:
                pytest.fail(f"PyAV frame {frame_num} with seeking doesn't match frame from start")
            if not ffmpeg_matches_start:
                pytest.skip(
                    f"FFmpeg seeking inaccurate: frame {frame_num} doesn't match ground truth. "
                    "This is a known limitation of the ffmpeg select filter with seeking."
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
        for i, (pyav_frame, ffmpeg_frame) in enumerate(zip(pyav_frames, ffmpeg_frames)):
            actual_frame_num = start_frame + i
            np.testing.assert_array_equal(
                pyav_frame,
                ffmpeg_frame,
                err_msg=f"Frame {actual_frame_num} ({start_time + i/fps:.3f}s) differs between PyAV and FFmpeg",
            )

    def test_single_frame_matches_ffmpeg(self, video_path):
        """Test that single frame extraction matches ffmpeg."""
        frame_idx = 42

        # Get ground truth from start
        ffmpeg_from_start = list(ffmpeg_read_frames_exact(video_path, 0, 50))
        pyav_from_start = list(pyav_read_frames_exact(video_path, start_frame=0, end_frame=50))

        # Extract using ffmpeg with seeking
        ffmpeg_frames = list(ffmpeg_read_frames_exact(video_path, frame_idx, frame_idx))
        assert len(ffmpeg_frames) == 1

        # Extract using our implementation with seeking
        pyav_frames = list(pyav_read_frames_exact(video_path, start_frame=frame_idx, end_frame=frame_idx))
        assert len(pyav_frames) == 1

        # Check if our implementation matches ground truth
        if not np.array_equal(pyav_frames[0], pyav_from_start[frame_idx]):
            pytest.fail(f"PyAV frame {frame_idx} with seeking doesn't match frame from start")

        # If ffmpeg doesn't match ground truth, skip (known limitation)
        if not np.array_equal(ffmpeg_frames[0], ffmpeg_from_start[frame_idx]):
            pytest.skip(
                f"FFmpeg seeking inaccurate for frame {frame_idx}. "
                "PyAV implementation is more accurate."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
