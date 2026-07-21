from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from simple_video_utils.frames import (
    _select_frames_by_index,
    read_frames_batched,
    read_frames_exact,
    read_frames_from_stream,
    stack_frames,
)
from simple_video_utils.metadata import video_metadata


class TestReadFramesExact:
    """Tests for the read_frames_exact function using example.mp4."""

    @pytest.fixture
    def video_path(self):
        """Path to the example video file."""
        return str(Path(__file__).parent / "assets" / "example.mp4")

    def test_invalid_frame_range_negative_start(self):
        """Test that negative start frame raises ValueError at call time."""
        with pytest.raises(ValueError, match="start_frame must be non-negative"):
            read_frames_exact("example.mp4", -1, 5)

    def test_invalid_frame_range_end_before_start(self):
        """Test that end_frame < start_frame raises ValueError at call time."""
        with pytest.raises(ValueError, match="invalid frame range"):
            read_frames_exact("example.mp4", 10, 5)

    def test_invalid_time_range_raises_at_call_time(self):
        """Inverted or fully-negative time windows fail fast, before opening."""
        with pytest.raises(ValueError, match="invalid time range"):
            read_frames_exact("example.mp4", start_time=3.0, end_time=1.0)
        with pytest.raises(ValueError, match="invalid time range"):
            read_frames_exact("example.mp4", start_time=-0.5, end_time=-0.01)

    def test_float_frame_index_raises_type_error(self):
        """Frame indices must be integers; floats fail fast, not as a container error."""
        with pytest.raises(TypeError):
            list(read_frames_exact("example.mp4", 2.0, 4.0))

    def test_negative_start_time_clamps_to_video_start(self, video_path):
        """A padded window reaching before 0 reads from the beginning."""
        padded = list(read_frames_exact(video_path, start_time=-0.5, end_time=0.2))
        from_zero = list(read_frames_exact(video_path, start_time=0.0, end_time=0.2))
        assert len(padded) == len(from_zero) > 0

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

        for range_frame, individual_frame in zip(range_frames, individual_frames, strict=False):
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
        for f1, f2 in zip(frames1, frames2, strict=False):
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

    def test_seek_path_matches_full_decode(self, video_path):
        """Starts past the 3s seek threshold must yield the exact target frames.

        Seeking lands on the keyframe at/before the seek point (this file's only
        keyframe is frame 0), so frames must be located by timestamp — counting
        from an assumed landing position used to yield frames ~2.5s too early.
        """
        all_frames = list(read_frames_exact(video_path))
        frames = list(read_frames_exact(video_path, start_frame=105, end_frame=110))
        assert len(frames) == 6
        for got, expected in zip(frames, all_frames[105:111], strict=True):
            np.testing.assert_array_equal(got, expected)

        by_time = list(read_frames_exact(video_path, start_time=3.5, end_time=110 / 30))
        assert len(by_time) == 6
        for got, expected in zip(by_time, frames, strict=True):
            np.testing.assert_array_equal(got, expected)

    def test_seek_path_with_nonzero_stream_start_time(self, video_path, tmp_path):
        """Frame indices must be measured from the stream origin, not t=0.

        Stream timestamps don't have to start at 0; the seek path used to
        compute every frame's index ~origin*fps too high and return nothing.
        """
        shifted = str(tmp_path / "shifted.mp4")
        with av.open(video_path) as source, av.open(shifted, mode="w") as output:
            in_stream = source.streams.video[0]
            out_stream = output.add_stream_from_template(in_stream)
            offset = round(100 / in_stream.time_base)
            for packet in source.demux(in_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                packet.pts += offset
                packet.dts += offset
                packet.stream = out_stream
                output.mux(packet)
        with av.open(shifted) as check:
            assert check.streams.video[0].start_time > 0, "fixture must not start at 0"

        all_frames = list(read_frames_exact(video_path))
        frames = list(read_frames_exact(shifted, start_frame=105, end_frame=110))
        assert len(frames) == 6
        for got, expected in zip(frames, all_frames[105:111], strict=True):
            np.testing.assert_array_equal(got, expected)

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
        remote_url = "https://www.papytane.com/mp4/accrobra.mp4"

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
        for i, (frame_idx, frame_time) in enumerate(zip(frames_by_index, frames_by_time, strict=False)):
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
        for f1, f2 in zip(frames_no_params, frames_explicit, strict=False):
            np.testing.assert_array_equal(f1, f2)

    def test_time_vs_frame_seeking_precision_remote(self):
        """Test that time and frame seeking produce identical frames on a longer remote video."""
        remote_url = "https://www.papytane.com/mp4/accrobra.mp4"

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
        for i, (frame_time, frame_idx) in enumerate(zip(frames_by_time, frames_by_frame, strict=False)):
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

    def test_invalid_utf8_metadata_video_decodes(self):
        """Test that all frames decode from a video with non-UTF-8 stream metadata."""
        video = str(Path(__file__).parent / "assets" / "invalid_utf8_metadata.mp4")

        frames = list(read_frames_exact(video))
        assert len(frames) == 59
        assert frames[0].shape == (360, 640, 3)

    def test_empty_video_raises(self):
        """A zero-frame video opens with no video stream, so frame reading must raise."""
        empty = str(Path(__file__).parent / "assets" / "empty.mp4")

        with pytest.raises(RuntimeError, match="Failed to open video"):
            list(read_frames_exact(empty))

    def test_corrupted_video_full_read_never_leaks_av_errors(self):
        """Corrupted data either raises our RuntimeError or decodes gracefully.

        What matters is that no raw av exception escapes: av <= 17 errors
        partway through the corrupted stream, av >= 18 recovers and decodes.
        """
        corrupted_path = str(Path(__file__).parent / "assets" / "corrupted.mp4")

        try:
            frames = list(read_frames_exact(corrupted_path, 0, None))
        except RuntimeError:
            pass
        else:
            assert len(frames) > 0


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
        for i, (stream_frame, file_frame) in enumerate(zip(stream_frames, file_frames, strict=False)):
            np.testing.assert_array_equal(
                stream_frame,
                file_frame,
                err_msg=f"Frame {i} differs between stream and file reading",
            )

    def test_read_frames_from_stream_skip_past_end(self, video_bytes):
        """Skipping more frames than the video has yields nothing, without error."""
        stream = BytesIO(video_bytes)
        _, frames_gen = read_frames_from_stream(stream, skip_frames=10_000)
        assert list(frames_gen) == []

    def test_read_frames_from_stream_negative_skip_raises(self, video_bytes):
        """Negative skip_frames fails fast with a clear error, before opening."""
        with pytest.raises(ValueError, match="skip_frames must be non-negative"):
            read_frames_from_stream(BytesIO(video_bytes), skip_frames=-1)

    def test_read_frames_from_stream_huge_skip_raises(self, video_bytes):
        """skip_frames beyond islice's bound fails fast instead of leaking the container."""
        with pytest.raises(ValueError, match="skip_frames is too large"):
            read_frames_from_stream(BytesIO(video_bytes), skip_frames=2**63)

    # skip=1 is the boundary where the eagerly-decoded first frame is consumed
    # by the skip instead of being yielded.
    @pytest.mark.parametrize("skip", [1, 5])
    def test_read_frames_from_stream_skip_frames(self, video_bytes, video_path, skip):
        """Test skipping initial frames from stream."""
        stream = BytesIO(video_bytes)
        _, frames_gen = read_frames_from_stream(stream, skip_frames=skip)
        stream_frames = list(frames_gen)

        file_frames = list(read_frames_exact(video_path, skip, None))

        assert len(stream_frames) == len(file_frames)

        for i, (stream_frame, file_frame) in enumerate(zip(stream_frames, file_frames, strict=False)):
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

    @staticmethod
    def _short_h264_clip(num_frames: int) -> BytesIO:
        buf = BytesIO()
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=60)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for i in range(num_frames):
                img = np.full((256, 256, 3), i * 30, dtype=np.uint8)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
            container.mux(stream.encode())
        buf.seek(0)
        return buf

    # Frame-threaded H.264 decoding ("AUTO") delays output by several frames;
    # starting a second decode() generator after the eager first-frame read
    # raised EOFError on clips shorter than that delay (issue #18).
    @pytest.mark.parametrize("fps", [None, 15])
    def test_read_frames_from_stream_short_h264_clip(self, fps):
        """Short H.264 clips must decode fully with the default thread_type."""
        _, frames_gen = read_frames_from_stream(self._short_h264_clip(8), fps=fps)
        expected = 2 if fps else 8
        assert len(list(frames_gen)) == expected

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


class TestStackFrames:
    """stack_frames must equal np.stack for any iterable of arrays."""

    @staticmethod
    def _arrays(n):
        rng = np.random.default_rng(seed=n)
        return [rng.integers(0, 255, (4, 6, 3), dtype=np.uint8) for _ in range(n)]

    @pytest.mark.parametrize("n", [1, 5, 64, 200])
    def test_matches_np_stack_without_hint(self, n):
        """Covers partial, exactly-one-chunk, and multi-chunk growth paths."""
        arrays = self._arrays(n)
        assert np.array_equal(stack_frames(iter(arrays)), np.stack(arrays))

    @pytest.mark.parametrize("hint", [1, 90, 100, 150])
    def test_wrong_hints_still_correct(self, hint):
        """Under- and over-counting hints compact or extend transparently."""
        arrays = self._arrays(100)
        assert np.array_equal(stack_frames(iter(arrays), size_hint=hint), np.stack(arrays))

    def test_exact_hint_returns_single_allocation(self):
        arrays = self._arrays(10)
        result = stack_frames(iter(arrays), size_hint=10)
        assert np.array_equal(result, np.stack(arrays))
        assert result.base is None  # owns its memory, no compaction copy

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no frames"):
            stack_frames(iter([]))


class TestReadFramesBatched:
    """read_frames_batched must equal np.stack over read_frames_exact."""

    ASSETS = Path(__file__).parent / "assets"

    @pytest.mark.parametrize("name", ["example.mp4", "rotated90.mp4", "no_nb_frames.webm"])
    def test_matches_stacked_generator(self, name):
        """Byte-equal to the generator path: known count, rotation, and unknown count (webm)."""
        src = str(self.ASSETS / name)
        expected = np.stack(list(read_frames_exact(src)))
        assert np.array_equal(read_frames_batched(src), expected)

    def test_frame_range(self):
        src = str(self.ASSETS / "example.mp4")
        expected = np.stack(list(read_frames_exact(src, start_frame=5, end_frame=20)))
        result = read_frames_batched(src, start_frame=5, end_frame=20)
        assert np.array_equal(result, expected)
        assert result.base is None  # range hint was exact: single allocation, no compaction

    def test_time_range(self):
        src = str(self.ASSETS / "example.mp4")
        expected = np.stack(list(read_frames_exact(src, start_time=1.0, end_time=2.0)))
        assert np.array_equal(read_frames_batched(src, start_time=1.0, end_time=2.0), expected)

    def test_range_past_end_is_compacted(self):
        """end_frame beyond the video must not leave uninitialized rows."""
        src = str(self.ASSETS / "example.mp4")
        expected = np.stack(list(read_frames_exact(src, start_frame=120, end_frame=500)))
        result = read_frames_batched(src, start_frame=120, end_frame=500)
        assert np.array_equal(result, expected)

    def test_empty_range_raises(self):
        src = str(self.ASSETS / "example.mp4")
        with pytest.raises(ValueError, match="no frames"):
            read_frames_batched(src, start_frame=100_000)

    def test_invalid_params_raise_before_open(self):
        with pytest.raises(ValueError, match="Cannot mix"):
            read_frames_batched("does-not-exist.mp4", start_frame=0, start_time=1.0)


class TestSelectFramesByIndex:
    """Unit tests for the timestamp-based frame filter, using stub frames."""

    @staticmethod
    def _frames(times):
        return [SimpleNamespace(time=t) for t in times]

    def test_selects_inclusive_index_range(self):
        frames = self._frames(i / 10 for i in range(10))
        selected = list(_select_frames_by_index(frames, origin=0.0, fps=10.0, target_start=3, target_end=5))
        assert selected == frames[3:6]

    def test_end_none_selects_to_exhaustion(self):
        frames = self._frames(i / 10 for i in range(5))
        selected = list(_select_frames_by_index(frames, origin=0.0, fps=10.0, target_start=2, target_end=None))
        assert selected == frames[2:]

    def test_skips_frames_without_timestamp(self):
        frames = self._frames([None, 0.0, 0.1])
        selected = list(_select_frames_by_index(frames, origin=0.0, fps=10.0, target_start=0, target_end=0))
        assert selected == [frames[1]]

    def test_nonzero_origin_offsets_indices(self):
        frames = self._frames(5.0 + i / 10 for i in range(5))
        selected = list(_select_frames_by_index(frames, origin=5.0, fps=10.0, target_start=1, target_end=2))
        assert selected == frames[1:3]


class TestFpsSelection:
    """The fps parameter drops frames onto a uniform grid — never duplicates."""

    @pytest.fixture
    def video_path(self):
        return str(Path(__file__).parent / "assets" / "example.mp4")

    def test_invalid_fps_raises_at_call_time(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            read_frames_exact("does-not-exist.mp4", fps=0)
        with pytest.raises(ValueError, match="fps must be positive"):
            read_frames_batched("does-not-exist.mp4", fps=-1)
        with pytest.raises(ValueError, match="fps must be positive"):
            read_frames_from_stream(BytesIO(b""), fps=0)

    def test_half_fps_roughly_halves_frame_count(self, video_path):
        meta = video_metadata(video_path)
        total = len(list(read_frames_exact(video_path)))
        half = list(read_frames_exact(video_path, fps=meta.fps / 2))
        assert abs(len(half) - total / 2) <= 1
        assert half[0].shape == (meta.height, meta.width, 3)

    def test_fps_above_source_keeps_every_frame(self, video_path):
        meta = video_metadata(video_path)
        total = len(list(read_frames_exact(video_path)))
        assert len(list(read_frames_exact(video_path, fps=meta.fps * 2))) == total

    def test_selected_frames_are_uniform_subset(self, video_path):
        meta = video_metadata(video_path)
        every_third = list(read_frames_exact(video_path, fps=meta.fps / 3))
        all_frames = list(read_frames_exact(video_path))
        for i, frame in enumerate(every_third):
            np.testing.assert_array_equal(frame, all_frames[3 * i])

    def test_batched_matches_exact(self, video_path):
        meta = video_metadata(video_path)
        exact = list(read_frames_exact(video_path, fps=meta.fps / 3))
        batched = read_frames_batched(video_path, fps=meta.fps / 3)
        np.testing.assert_array_equal(batched, np.stack(exact))

    def test_stream_downsamples(self, video_path):
        meta = video_metadata(video_path)
        total = len(list(read_frames_exact(video_path)))
        with open(video_path, "rb") as f:
            _, gen = read_frames_from_stream(BytesIO(f.read()), fps=meta.fps / 2)
        assert abs(len(list(gen)) - total / 2) <= 1
