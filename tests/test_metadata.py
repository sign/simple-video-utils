from pathlib import Path

import pytest

from simple_video_utils.metadata import count_frames, video_metadata, video_metadata_from_bytes


class TestVideoMetadata:
    """Tests for video metadata extraction functions."""

    @pytest.fixture
    def video_path(self):
        """Path to the example video file."""
        return str(Path(__file__).parent / "assets" / "example.mp4")

    @pytest.fixture
    def video_bytes(self, video_path):
        """Load example video as bytes."""
        return Path(video_path).read_bytes()

    def test_video_metadata(self, video_path):
        """Test that we can read video metadata."""
        meta = video_metadata(video_path)

        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0
        assert meta.duration is not None
        assert meta.duration > 0
        assert isinstance(meta.width, int)
        assert isinstance(meta.height, int)
        assert isinstance(meta.fps, float)
        assert isinstance(meta.duration, float)

    def test_video_metadata_from_bytes(self, video_bytes):
        """Test metadata extraction from video bytes."""
        meta = video_metadata_from_bytes(video_bytes)

        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0
        assert meta.duration is not None
        assert meta.duration > 0
        assert isinstance(meta.width, int)
        assert isinstance(meta.height, int)
        assert isinstance(meta.fps, float)

    def test_video_metadata_from_bytes_matches_file(self, video_bytes, video_path):
        """Test that bytes-based metadata matches file-based metadata."""
        meta_bytes = video_metadata_from_bytes(video_bytes)
        meta_file = video_metadata(video_path)

        assert meta_bytes.width == meta_file.width
        assert meta_bytes.height == meta_file.height
        assert meta_bytes.fps == meta_file.fps
        assert meta_bytes.duration == meta_file.duration

    def test_bad_color_space_video(self):
        """Test metadata extraction from a video with unusual color space."""
        strange_video = str(Path(__file__).parent / "assets" / "bad_colorspace.mp4")

        meta = video_metadata(strange_video)
        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0

    def test_webm_file(self):
        """Test metadata extraction from WebM file."""
        webm_video = str(Path(__file__).parent / "assets" / "example.webm")

        meta = video_metadata(webm_video)
        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0
        # webm without nb_frames in the header — common for browser
        # MediaRecorder output. duration must still be recoverable.
        assert meta.duration is not None
        assert meta.duration > 0

    @pytest.mark.parametrize(
        "filename",
        [
            "no_nb_frames.mkv",  # Matroska without nb_frames in the header
            "no_nb_frames.webm",  # WebM without nb_frames in the header
            "buggy_mov_header.mov",  # header declares more frames than actually decode
        ],
    )
    def test_nb_frames_matches_decoded_frames(self, filename):
        """nb_frames must match the true decoded count even when the header is absent or lies."""
        video = str(Path(__file__).parent / "assets" / filename)

        meta = video_metadata(video)
        assert meta.nb_frames == count_frames(video)

    def test_invalid_utf8_metadata_video(self):
        """Test a video whose stray data streams carry non-UTF-8 handler_name metadata."""
        video = str(Path(__file__).parent / "assets" / "invalid_utf8_metadata.mp4")

        meta = video_metadata(video)
        assert meta.width == 640
        assert meta.height == 360
        assert abs(meta.fps - 29.97) < 0.01
        assert meta.duration is not None
        assert meta.duration > 0

    def test_empty_video_raises(self):
        """
        A zero-frame video (created with `ffmpeg -f lavfi -i color -frames:v 0`):
        the mov demuxer drops the sample-less track, so the container opens with
        no video stream and metadata extraction must raise.
        """
        empty = str(Path(__file__).parent / "assets" / "empty.mp4")

        with pytest.raises(RuntimeError, match="Failed to open video"):
            video_metadata(empty)

    def test_remote_video_url(self):
        """Test metadata extraction from a remote video URL."""
        remote_url = "https://www.papytane.com/mp4/accrobra.mp4"

        meta = video_metadata(remote_url)
        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0
        assert meta.duration is not None
        assert meta.duration > 0

