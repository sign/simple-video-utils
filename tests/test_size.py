from pathlib import Path

import numpy as np
import pytest

from simple_video_utils.frames import read_frames_batched, read_frames_exact, stack_frames

ASSETS = Path(__file__).parent / "assets"


def test_size_scales_down_by_area_and_up_by_bicubic():
    path = str(ASSETS / "example.mp4")
    full = read_frames_batched(path, end_frame=1)
    height, width = full.shape[1:3]
    half = read_frames_batched(path, end_frame=1, size=(width // 2, height // 2))
    assert half.shape == (2, height // 2, width // 2, 3)
    even = full[:, : height // 2 * 2, : width // 2 * 2]
    pooled = even.reshape(2, height // 2, 2, width // 2, 2, 3).mean(axis=(2, 4))
    assert np.abs(half - pooled).mean() < 3
    larger = read_frames_batched(path, end_frame=1, size=(width + 16, height + 16))
    assert larger.shape == (2, height + 16, width + 16, 3)


def test_size_applies_in_display_orientation():
    """rotated90.mp4 is stored 640x360 and shown 360x640: a size is the shown frame's."""
    path = str(ASSETS / "rotated90.mp4")
    full = next(read_frames_exact(path, end_frame=0))
    half = next(read_frames_exact(path, end_frame=0, size=(180, 320)))
    assert full.shape == (640, 360, 3)
    assert half.shape == (320, 180, 3)
    pooled = full.reshape(320, 2, 180, 2, 3).mean(axis=(1, 3))
    assert np.abs(half - pooled).mean() < 3


def test_size_is_validated_at_call_time():
    with pytest.raises(ValueError, match="size"):
        read_frames_exact("missing.mp4", size=(0, 10))


def test_a_frame_of_another_size_is_named_by_its_index():
    # the third frame lands just after the first chunk of two fills
    frames = [np.zeros((2, 2, 3), np.uint8)] * 2 + [np.zeros((3, 3, 3), np.uint8)]
    with pytest.raises(ValueError, match="frame 2 has shape"):
        stack_frames(frames, size_hint=2)
