from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_exact, read_frames_from_stream
from simple_video_utils.metadata import video_metadata


class TestReadFramesExact:
    """Tests for the read_frames_exact function using example.mp4."""

    @pytest.fixture
    def video_path(self):
        """Path to the example video file."""
        return str(Path(__file__).parent / "assets" / "example.mp4")

    def test_invalid_frame_range_negative_start(self):
        """Test that negative start frame raises AssertionError."""
        with pytest.raises(AssertionError, match="invalid frame range"):
            list(read_frames_exact("example.mp4", -1, 5))

    def test_invalid_frame_range_end_before_start(self):
        """Test that end_frame < start_frame raises AssertionError."""
        with pytest.raises(AssertionError, match="invalid frame range"):
            list(read_frames_exact("example.mp4", 10, 5))

    def test_read_single_frame(self, video_path):
        """Test reading a single frame from example.mp4."""
        frames = list(read_frames_exact(video_path, 0, 0))

        assert len(frames) == 1
        frame = frames[0]

        # Check frame properties
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # RGB channels

        # Check that frame contains actual image data (not all zeros)
        assert np.sum(frame) > 0

    def test_read_multiple_frames(self, video_path):
        """Test reading multiple consecutive frames."""
        frames = list(read_frames_exact(video_path, 0, 2))

        assert len(frames) == 3  # frames 0, 1, 2 (inclusive)

        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3

    def test_frame_range_consistency(self, video_path):
        """Test that reading the same frame multiple times gives consistent results."""
        frame1 = list(read_frames_exact(video_path, 5, 5))[0]
        frame2 = list(read_frames_exact(video_path, 5, 5))[0]

        np.testing.assert_array_equal(frame1, frame2)

    def test_sequential_vs_range_reading(self, video_path):
        """Test that reading frames individually vs as range gives same results."""
        # Read frames 1, 2, 3 as a range
        range_frames = list(read_frames_exact(video_path, 1, 3))

        # Read each frame individually
        individual_frames = [
            list(read_frames_exact(video_path, 1, 1))[0],
            list(read_frames_exact(video_path, 2, 2))[0],
            list(read_frames_exact(video_path, 3, 3))[0],
        ]

        assert len(range_frames) == len(individual_frames) == 3

        for range_frame, individual_frame in zip(range_frames, individual_frames):
            np.testing.assert_array_equal(range_frame, individual_frame)

    def test_frames_are_different(self, video_path):
        """Test that consecutive frames are actually different (video has motion)."""
        frames = list(read_frames_exact(video_path, 0, 10))

        if len(frames) >= 2:
            # Check that not all frames are identical
            differences = []
            for i in range(len(frames) - 1):
                diff = np.sum(np.abs(frames[i].astype(np.int16) - frames[i + 1].astype(np.int16)))
                differences.append(diff)

            # At least some frames should be different
            assert max(differences) > 0, "All consecutive frames are identical"

    def test_large_frame_range(self, video_path):
        """Test reading a larger range of frames."""
        # Get video metadata first to know how many frames we have
        meta = video_metadata(video_path)
        max_frames = meta.nb_frames or 30  # Default to 30 if unknown

        if max_frames and max_frames > 10:
            end_frame = min(max_frames - 1, 20)  # Read up to frame 20 or video end
            frames = list(read_frames_exact(video_path, 0, end_frame))

            expected_count = end_frame + 1
            assert len(frames) == expected_count

            # All frames should have same dimensions
            shapes = [frame.shape for frame in frames]
            assert all(shape == shapes[0] for shape in shapes)

    def test_end_frame_none_from_start(self, video_path):
        """Test reading from start to end of video with end_frame=None."""
        # Read entire video from start
        frames_all = list(read_frames_exact(video_path, 0, None))

        # Read first few frames with explicit end_frame
        frames_partial = list(read_frames_exact(video_path, 0, 5))

        # All frames should be valid
        assert len(frames_all) > 0
        assert len(frames_all) >= len(frames_partial)

        # First frames should match
        for i in range(min(len(frames_all), len(frames_partial))):
            np.testing.assert_array_equal(frames_all[i], frames_partial[i])

    def test_end_frame_none_from_middle(self, video_path):
        """Test reading from middle to end of video with end_frame=None."""
        start_frame = 5

        # Read from middle to end with end_frame=None
        frames_to_end = list(read_frames_exact(video_path, start_frame, None))

        # Should get some frames
        assert len(frames_to_end) > 0

        # Each frame should be valid
        for frame in frames_to_end:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3

    def test_start_frame_zero_no_seeking(self, video_path):
        """Test that start_frame=0 optimization works correctly."""
        # These should produce identical results
        frames_with_end = list(read_frames_exact(video_path, 0, 5))
        frames_without_end = list(read_frames_exact(video_path, 0, None))[:6]  # Take first 6 frames

        # Compare first 6 frames
        assert len(frames_with_end) == 6  # frames 0-5 inclusive
        assert len(frames_without_end) >= 6

        for i in range(6):
            np.testing.assert_array_equal(frames_with_end[i], frames_without_end[i])

    def test_end_frame_none_consistency(self, video_path):
        """Test that end_frame=None gives consistent results."""
        # Read twice with end_frame=None
        frames1 = list(read_frames_exact(video_path, 0, None))
        frames2 = list(read_frames_exact(video_path, 0, None))

        # Should get same number of frames
        assert len(frames1) == len(frames2)

        # Frames should be identical
        for f1, f2 in zip(frames1, frames2):
            np.testing.assert_array_equal(f1, f2)

    def test_end_frame_none_vs_explicit_end(self, video_path):
        """Test end_frame=None vs explicit end_frame for entire video."""
        # Get video metadata to find total frames
        meta = video_metadata(video_path)
        total_frames = meta.nb_frames

        if total_frames and total_frames > 10:
            # Read with end_frame=None
            frames_none = list(read_frames_exact(video_path, 0, None))

            # Read with explicit end_frame (assuming we know total frames)
            frames_explicit = list(read_frames_exact(video_path, 0, total_frames - 1))

            # Should get same number of frames (or close due to container metadata)
            # Allow small difference due to potential metadata inaccuracy
            assert abs(len(frames_none) - len(frames_explicit)) <= 1

            # First several frames should match
            min_len = min(len(frames_none), len(frames_explicit))
            for i in range(min(min_len, 10)):  # Compare first 10 frames
                np.testing.assert_array_equal(frames_none[i], frames_explicit[i])

    def test_bad_color_space_video(self):
        """Test reading frames from a video with unusual color space metadata."""
        strange_video = str(Path(__file__).parent / "assets" / "bad_colorspace.mp4")

        # Test reading frames (ffmpeg 8.0+ handles this video correctly)
        frames = list(read_frames_exact(strange_video, 0))
        assert len(frames) == 182

    def test_webm_file(self):
        """Test reading frames from a WebM file."""
        webm_video = str(Path(__file__).parent / "assets" / "example.webm")

        # Test reading frames
        frames = list(read_frames_exact(webm_video, 0))
        assert len(frames) == 67

    def test_remote_video_url(self):
        """Test reading frames from a remote video URL."""
        remote_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"

        # Test reading first frame
        frames = list(read_frames_exact(remote_url, 0, 0))
        assert len(frames) == 1

        frame = frames[0]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        assert np.sum(frame) > 0

        # Test reading multiple frames
        frames_multi = list(read_frames_exact(remote_url, 0, 2))
        assert len(frames_multi) == 3

    def test_time_based_extraction(self, video_path):
        """Test reading frames using time-based parameters."""
        # Read using time parameters
        frames = list(read_frames_exact(video_path, start_time=0.0, end_time=1.0))

        # Should get approximately 1 second worth of frames
        meta = video_metadata(video_path)
        expected_frames = int(meta.fps) + 1  # +1 because end frame is inclusive
        # Allow some tolerance for frame extraction
        assert abs(len(frames) - expected_frames) <= 2

    def test_time_vs_frame_equivalence(self, video_path):
        """Test that time-based and frame-based extraction produce equivalent results."""
        meta = video_metadata(video_path)
        fps = meta.fps

        # Extract frames 10-20 using frame indices
        frames_by_index = list(read_frames_exact(video_path, start_frame=10, end_frame=20))

        # Extract same frames using time
        start_time = 10 / fps
        end_time = 20 / fps
        frames_by_time = list(read_frames_exact(video_path, start_time=start_time, end_time=end_time))

        # Should get same number of frames
        assert len(frames_by_index) == len(frames_by_time)

        # Frames should be identical
        for i, (frame_idx, frame_time) in enumerate(zip(frames_by_index, frames_by_time)):
            np.testing.assert_array_equal(
                frame_idx,
                frame_time,
                err_msg=f"Frame {i} differs between time and index extraction",
            )

    def test_time_based_start_only(self, video_path):
        """Test time-based extraction with only start_time specified."""
        frames = list(read_frames_exact(video_path, start_time=0.5))

        # Should get frames from 0.5 seconds to end
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8

    def test_time_based_end_only(self, video_path):
        """Test time-based extraction with only end_time specified."""
        frames = list(read_frames_exact(video_path, end_time=1.0))

        # Should get frames from start to 1.0 seconds
        meta = video_metadata(video_path)
        expected_frames = int(meta.fps) + 1
        assert abs(len(frames) - expected_frames) <= 2

    def test_cannot_mix_frame_and_time_params(self, video_path):
        """Test that mixing frame and time parameters raises ValueError."""
        with pytest.raises(ValueError, match="Cannot mix frame-based and time-based"):
            list(read_frames_exact(video_path, start_frame=0, end_time=1.0))

        with pytest.raises(ValueError, match="Cannot mix frame-based and time-based"):
            list(read_frames_exact(video_path, start_time=0.0, end_frame=10))

        with pytest.raises(ValueError, match="Cannot mix frame-based and time-based"):
            list(read_frames_exact(video_path, start_frame=0, start_time=0.0))

    def test_no_parameters_reads_all(self, video_path):
        """Test that calling with no parameters reads all frames from start."""
        frames_no_params = list(read_frames_exact(video_path))
        frames_explicit = list(read_frames_exact(video_path, start_frame=0))

        # Should produce same result
        assert len(frames_no_params) == len(frames_explicit)
        for f1, f2 in zip(frames_no_params, frames_explicit):
            np.testing.assert_array_equal(f1, f2)

    def test_time_vs_frame_seeking_precision_remote(self):
        """Test that time and frame seeking produce identical frames on a longer remote video."""
        remote_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"

        # Get video metadata to calculate frame indices
        meta = video_metadata(remote_url)
        fps = meta.fps

        # Test 5-7 seconds
        start_time_sec = 5.0
        end_time_sec = 7.0

        # Calculate corresponding frame indices
        start_frame_idx = int(start_time_sec * fps)
        end_frame_idx = int(end_time_sec * fps)

        # Extract using time parameters
        frames_by_time = list(read_frames_exact(remote_url, start_time=start_time_sec, end_time=end_time_sec))

        # Extract using frame indices
        frames_by_frame = list(read_frames_exact(remote_url, start_frame=start_frame_idx, end_frame=end_frame_idx))

        # Should get same number of frames
        assert len(frames_by_time) == len(frames_by_frame), (
            f"Frame count mismatch: time-based={len(frames_by_time)}, "
            f"frame-based={len(frames_by_frame)}"
        )

        # Verify we got the expected number of frames
        expected_frame_count = end_frame_idx - start_frame_idx + 1  # +1 because end is inclusive
        assert len(frames_by_time) == expected_frame_count, (
            f"Expected {expected_frame_count} frames (from frame {start_frame_idx} to {end_frame_idx}), "
            f"got {len(frames_by_time)}"
        )

        # Every frame should be identical
        for i, (frame_time, frame_idx) in enumerate(zip(frames_by_time, frames_by_frame)):
            actual_frame_num = start_frame_idx + i
            np.testing.assert_array_equal(
                frame_time,
                frame_idx,
                err_msg=f"Frame {actual_frame_num} differs between time-based and frame-based extraction",
            )

        # Verify frames are not all identical (video has content)
        if len(frames_by_time) >= 2:
            diff = np.sum(np.abs(frames_by_time[0].astype(np.int16) - frames_by_time[-1].astype(np.int16)))
            assert diff > 0, "First and last frames are identical - video may not have motion"

    def test_corrupted_video_metadata_readable(self):
        """Test that metadata can be read from corrupted video (ffprobe passes)."""
        corrupted_path = str(Path(__file__).parent / "assets" / "corrupted.mp4")

        # Metadata should be readable even though video is corrupted
        meta = video_metadata(corrupted_path)
        assert meta.width == 256
        assert meta.height == 256
        assert meta.fps == 25.0

    def test_corrupted_video_full_read_fails(self):
        """Test that reading all frames from corrupted video raises RuntimeError."""
        corrupted_path = str(Path(__file__).parent / "assets" / "corrupted.mp4")

        # Reading all frames should fail when hitting corrupted data
        with pytest.raises(RuntimeError, match="Failed to open video"):
            list(read_frames_exact(corrupted_path, 0, None))


class TestReadFramesFromStream:
    """Tests for streaming video input via read_frames_from_stream."""

    @pytest.fixture
    def video_path(self):
        """Path to the example video file."""
        return str(Path(__file__).parent / "assets" / "example.mp4")

    @pytest.fixture
    def video_bytes(self, video_path):
        """Load example video as bytes."""
        return Path(video_path).read_bytes()

    def test_read_frames_from_stream_basic(self, video_bytes):
        """Test reading frames from a BytesIO stream."""
        stream = BytesIO(video_bytes)
        meta, frames_gen = read_frames_from_stream(stream)

        # Check metadata
        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0

        # Read first frame
        frame = next(frames_gen)
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (meta.height, meta.width, 3)
        assert np.sum(frame) > 0

    def test_read_frames_from_stream_all_frames(self, video_bytes, video_path):
        """Test that stream reading produces same frames as file reading."""
        stream = BytesIO(video_bytes)
        meta, frames_gen = read_frames_from_stream(stream)

        stream_frames = list(frames_gen)
        file_frames = list(read_frames_exact(video_path, 0, None))

        # Same number of frames
        assert len(stream_frames) == len(file_frames)

        # Frames should be identical
        for i, (stream_frame, file_frame) in enumerate(zip(stream_frames, file_frames)):
            np.testing.assert_array_equal(
                stream_frame,
                file_frame,
                err_msg=f"Frame {i} differs between stream and file reading",
            )

    def test_read_frames_from_stream_skip_frames(self, video_bytes, video_path):
        """Test skipping initial frames from stream."""
        skip = 5

        stream = BytesIO(video_bytes)
        _, frames_gen = read_frames_from_stream(stream, skip_frames=skip)
        stream_frames = list(frames_gen)

        # Compare with file-based reading starting at frame 5
        file_frames = list(read_frames_exact(video_path, skip, None))

        assert len(stream_frames) == len(file_frames)

        for i, (stream_frame, file_frame) in enumerate(zip(stream_frames, file_frames)):
            np.testing.assert_array_equal(
                stream_frame,
                file_frame,
                err_msg=f"Frame {i} (skipped {skip}) differs",
            )

    def test_read_frames_from_stream_metadata_matches(self, video_bytes, video_path):
        """Test that returned metadata matches expected values."""
        stream = BytesIO(video_bytes)
        meta_stream, _ = read_frames_from_stream(stream)
        meta_file = video_metadata(video_path)

        assert meta_stream.width == meta_file.width
        assert meta_stream.height == meta_file.height
        assert meta_stream.fps == meta_file.fps

    def test_read_frames_from_stream_webm(self):
        """Test reading frames from a WebM stream."""
        video_path = Path(__file__).parent / "assets" / "example.webm"
        video_bytes = video_path.read_bytes()

        stream = BytesIO(video_bytes)
        meta, frames_gen = read_frames_from_stream(stream)

        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0

        frames = list(frames_gen)
        assert len(frames) == 67  # Same as test_webm_file


if __name__ == "__main__":
    pytest.main([__file__])
