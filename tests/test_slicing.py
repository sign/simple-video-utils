import io

import av
import numpy as np
import pytest

from simple_video_utils.slicing import slice_video


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "src.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for i in range(30):
            arr = np.full((240, 320, 3), i * 8 % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def _dims(clip: bytes) -> tuple[int, int]:
    with av.open(io.BytesIO(clip), mode="r") as container:
        frame = next(container.decode(video=0))
        return frame.width, frame.height


def test_slice_returns_one_clip_per_range(video):
    clips = slice_video(video, [(0.0, 0.3), (0.5, 0.8)])
    assert len(clips) == 2
    assert all(clip for clip in clips)


def test_slice_keeps_source_size_by_default(video):
    [clip] = slice_video(video, [(0.0, 0.3)])
    assert _dims(clip) == (320, 240)


def test_slice_center_crops_and_resizes(video):
    [clip] = slice_video(video, [(0.0, 0.3)], size=256)
    assert _dims(clip) == (256, 256)
