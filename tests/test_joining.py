import io

import av
import numpy as np
import pytest

from simple_video_utils.joining import join_videos
from simple_video_utils.slicing import slice_video_stream


def _video(values: list[int], fps: int = 24, codec: str = "h264", format: str = "mp4") -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format=format) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for value in values:
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), value, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return output.getvalue()


def _frames(video: bytes) -> list[np.ndarray]:
    with av.open(io.BytesIO(video)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_overlapping_slices_are_joined_without_reencoding():
    source = _video(list(range(36)), format="mpegts")

    joined = join_videos(list(slice_video_stream(io.BytesIO(source), duration=0.5)))
    expected, actual = _frames(source), _frames(joined)
    assert len(actual) == len(expected)
    for source_frame, joined_frame in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(source_frame, joined_frame)


def test_webm_slices_join_via_reencode():
    # slice_video_stream re-encodes WebM sources into exact windows, so those
    # clips share no packets: joining decodes and re-encodes them, once.
    source = _video(list(range(36)), codec="libvpx", format="webm")

    joined = join_videos(list(slice_video_stream(io.BytesIO(source), duration=0.5)))
    expected, actual = _frames(source), _frames(joined)
    assert len(actual) == len(expected)
    for source_frame, joined_frame in zip(expected, actual, strict=True):
        np.testing.assert_allclose(  # atol: two lossy H.264 generations
            joined_frame.astype(int), source_frame.astype(int), atol=6)


def test_unrelated_videos_are_reencoded_in_order():
    first = _video([20] * 4)
    second = _video([220] * 4)

    frames = _frames(join_videos([first, second]))

    assert len(frames) == 8
    assert np.mean(frames[0]) < 50
    assert np.mean(frames[-1]) > 190


def test_one_video_is_unchanged():
    video = _video([20])
    assert join_videos([video]) is video


def test_join_needs_a_video():
    with pytest.raises(ValueError, match="at least one"):
        join_videos([])
