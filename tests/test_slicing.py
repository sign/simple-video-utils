import io

import av
import numpy as np
import pytest

from simple_video_utils.slicing import slice_video


def _write_video(path, width=320, height=240, frames=30, fps=30):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            arr = np.full((height, width, 3), i * 8 % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


@pytest.fixture
def video(tmp_path):
    return _write_video(tmp_path / "src.mp4")


def _dims(clip: bytes) -> tuple[int, int]:
    with av.open(io.BytesIO(clip), mode="r") as container:
        frame = next(container.decode(video=0))
        return frame.width, frame.height


def test_slice_returns_one_clip_per_range(video):
    clips = list(slice_video(video, [(0.0, 0.3), (0.5, 0.8)]))
    assert len(clips) == 2
    assert all(clip for clip in clips)


def test_slice_keeps_source_size_by_default(video):
    # Default path stream-copies, so the source resolution is preserved.
    [clip] = slice_video(video, [(0.0, 0.3)])
    assert _dims(clip) == (320, 240)


def test_slice_center_crops_and_resizes(video):
    [clip] = slice_video(video, [(0.0, 0.3)], size=256)
    assert _dims(clip) == (256, 256)


def test_matching_size_is_copied(tmp_path):
    # Source is already 128x128, so size=128 needs no re-encode.
    square = _write_video(tmp_path / "square.mp4", width=128, height=128, frames=15)
    [clip] = slice_video(square, [(0.0, 0.3)], size=128)
    assert _dims(clip) == (128, 128)


def test_out_of_range_slice_raises(video):
    # Source is 1s; slices past the end, before 0, or reversed are errors.
    for bad in [(5.0, 5.5), (-0.1, 0.3), (0.5, 0.2)]:
        with pytest.raises(ValueError, match="out of range"):
            list(slice_video(video, [bad]))
        with pytest.raises(ValueError, match="out of range"):
            list(slice_video(video, [bad], size=256))
