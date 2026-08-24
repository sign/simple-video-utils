import io

import av
import numpy as np
import pytest

from simple_video_utils.slicing import slice_video_stream


def _streaming_video(frames=36, fps=24, codec="h264", format="mpegts") -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format=format) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            array = np.full((48, 64, 3), i * 7 % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return output.getvalue()


def _frames(video: bytes) -> list[np.ndarray]:
    with av.open(io.BytesIO(video)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_clips_are_split_into_duration_windows():
    data = _streaming_video()
    source_frames = _frames(data)

    clips = list(slice_video_stream(io.BytesIO(data), duration=0.5))
    decoded = [_frames(clip) for clip in clips]
    assert len(decoded) == 3
    assert [frames[0].shape[:2] for frames in decoded] == [(48, 64)] * 3
    for index, frames in enumerate(decoded):
        assert len(frames) >= 12
        np.testing.assert_array_equal(frames[0], source_frames[index * 12])


@pytest.mark.parametrize("duration", [0, -1, float("inf")])
def test_duration_must_be_positive_and_finite(duration):
    with pytest.raises(ValueError, match="duration"):
        list(slice_video_stream(io.BytesIO(b"video"), duration=duration))
