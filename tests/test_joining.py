import io
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.joining import join_videos
from simple_video_utils.slicing import slice_video, slice_video_stream


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


def _times(video: bytes) -> list[float]:
    with av.open(io.BytesIO(video)) as container:
        return [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]


def test_overlapping_slices_are_joined_without_reencoding():
    source = _video(list(range(36)), format="mpegts")

    # a generator, as in the README — join_videos must accept one
    joined = join_videos(clip.data for clip in slice_video_stream(io.BytesIO(source), duration=0.5))
    expected, actual = _frames(source), _frames(joined)
    assert len(actual) == len(expected)
    for source_frame, joined_frame in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(source_frame, joined_frame)


def test_webm_slices_join_via_reencode():
    # slice_video_stream re-encodes WebM sources into exact windows, so those
    # clips share no packets: joining decodes and re-encodes them, once.
    source = _video(list(range(36)), codec="libvpx", format="webm")

    joined = join_videos([clip.data for clip in slice_video_stream(io.BytesIO(source), duration=0.5)])
    expected, actual = _frames(source), _frames(joined)
    assert len(actual) == len(expected)
    for source_frame, joined_frame in zip(expected, actual, strict=True):
        np.testing.assert_allclose(  # atol: two lossy H.264 generations
            joined_frame.astype(int), source_frame.astype(int), atol=6)


def test_unrelated_videos_are_reencoded_in_order():
    first = _video([20] * 4)
    second = _video([220] * 4)

    joined = join_videos([first, second])
    frames = _frames(joined)

    assert len(frames) == 8
    assert np.mean(frames[0]) < 50
    assert np.mean(frames[-1]) > 190
    # re-encoded frames must carry real presentation times, not collapse to t=0
    times = _times(joined)
    assert times == sorted(times)
    assert len(set(times)) == 8
    assert times[-1] == pytest.approx(7 / 24)


def test_gapped_slices_reencode_without_the_gap():
    # Packet copy can't express a gap between clips: the second clip's hidden
    # keyframe lead-in would surface as visible content bridging it. Gapped
    # joins fall back to re-encoding just the clips' visible frames (jump cut).
    src = str(Path(__file__).parent / "assets" / "example.mp4")
    clips = list(slice_video(src, [(0.0, 0.5), (3.0, 3.5)]))
    frames = _frames(join_videos(clips))
    assert 30 <= len(frames) <= 40  # two 15-frame windows plus B-frame trailing extras


def test_identical_clips_merge_into_one():
    # Overlaps merge on the source timeline rather than repeating: a clip
    # fully contained in the previous ones adds nothing.
    source = _video(list(range(36)), format="mpegts")
    clip = next(slice_video_stream(io.BytesIO(source), duration=0.5)).data
    assert len(_frames(join_videos([clip, clip]))) == len(_frames(clip))


def test_one_video_is_unchanged():
    video = _video([20])
    assert join_videos([video]) is video


def test_join_needs_a_video():
    with pytest.raises(ValueError, match="at least one"):
        join_videos([])
