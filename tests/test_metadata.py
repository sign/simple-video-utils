from pathlib import Path

import pytest

from simple_video_utils.metadata import video_metadata, video_metadata_from_bytes


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
        assert isinstance(meta.width, int)
        assert isinstance(meta.height, int)
        assert isinstance(meta.fps, float)

    def test_video_metadata_from_bytes(self, video_bytes):
        """Test metadata extraction from video bytes."""
        meta = video_metadata_from_bytes(video_bytes)

        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0
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

    def test_remote_video_url(self):
        """Test metadata extraction from a remote video URL."""
        remote_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"

        meta = video_metadata(remote_url)
        assert meta.width > 0
        assert meta.height > 0
        assert meta.fps > 0


if __name__ == "__main__":
    pytest.main([__file__])
