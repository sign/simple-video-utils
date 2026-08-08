"""Container reuse via open_video: one av.open serving metadata, keyframes, and frame reads (issue #8)."""
from pathlib import Path

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_batched, read_frames_exact
from simple_video_utils.metadata import (
    count_frames,
    keyframe_indices,
    open_video,
    video_metadata,
    video_metadata_from_container,
)

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture
def video_path():
    return str(ASSETS / "example.mp4")


class TestOpenVideo:
    def test_matches_path_based_calls(self, video_path):
        with open_video(video_path) as video:
            meta = video_metadata_from_container(video)
            keys = keyframe_indices(video)
            frames = list(read_frames_exact(video, start_frame=2, end_frame=5))

        assert meta == video_metadata(video_path)
        assert keys == keyframe_indices(video_path)
        expected = list(read_frames_exact(video_path, start_frame=2, end_frame=5))
        assert len(frames) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(frames, expected, strict=True))

    def test_call_order_does_not_matter(self, video_path):
        """Every helper rewinds, so metadata after a frame read isn't skewed."""
        with open_video(video_path) as video:
            first = list(read_frames_exact(video, start_frame=0, end_frame=3))
            meta = video_metadata_from_container(video)
            keys = keyframe_indices(video)
            again = list(read_frames_exact(video, start_frame=0, end_frame=3))

        assert meta == video_metadata(video_path)
        assert keys == keyframe_indices(video_path)
        assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))

    def test_count_frames_accepts_container(self, video_path):
        with open_video(video_path) as video:
            assert count_frames(video) == count_frames(video_path)

    def test_batched_matches_path(self, video_path):
        with open_video(video_path) as video:
            batch = read_frames_batched(video, start_frame=0, end_frame=4)
        assert np.array_equal(batch, read_frames_batched(video_path, start_frame=0, end_frame=4))

    def test_batched_headerless_container(self):
        """The size-hint fallback (no header frame count) works on a reused container."""
        path = str(ASSETS / "no_nb_frames.webm")
        with open_video(path) as video:
            batch = read_frames_batched(video, start_frame=0, end_frame=4)
        assert np.array_equal(batch, read_frames_batched(path, start_frame=0, end_frame=4))

    def test_rotated_video(self):
        path = str(ASSETS / "rotated90.mp4")
        with open_video(path) as video:
            meta = video_metadata_from_container(video)
            frames = list(read_frames_exact(video, start_frame=0, end_frame=1))
        assert meta == video_metadata(path)
        assert frames[0].shape[:2] == (meta.height, meta.width)

    def test_short_clip_repeated_reads(self):
        """Re-decoding after seek(0) must not hit a flushed decoder (issue #18's hazard)."""
        path = str(ASSETS / "example-short.mp4")
        with open_video(path) as video:
            first = list(read_frames_exact(video))
            again = list(read_frames_exact(video))
        expected = list(read_frames_exact(path))
        assert len(first) == len(again) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(first, expected, strict=True))

    def test_open_failure(self):
        with pytest.raises(RuntimeError, match="Failed to open video"), open_video(str(ASSETS / "missing.mp4")):
            pass
