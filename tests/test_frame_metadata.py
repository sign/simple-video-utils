import io
from fractions import Fraction

import av
import numpy as np
import pytest

from simple_video_utils.chapters import read_chapters
from simple_video_utils.frames import FrameMetadata, FrameSize, read_frames_from_stream


@pytest.fixture
def vfr_video():
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        container.set_chapters([
            {"id": i, "start": start, "end": end, "time_base": Fraction(1, 1000), "metadata": {"title": str(i)}}
            for i, (start, end) in enumerate(((0, 100), (100, 400)))
        ])
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i, pts in enumerate((0, 1, 3, 4, 10, 11)):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 30, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return output.getvalue()


def test_metadata_matches_native_vfr_frames_and_default_pixels(vfr_video):
    with av.open(io.BytesIO(vfr_video)) as container:
        expected = [(frame.pts, frame.time_base) for frame in container.decode(video=0)]
    _, default = read_frames_from_stream(io.BytesIO(vfr_video))
    _, detailed = read_frames_from_stream(io.BytesIO(vfr_video), return_frame_metadata=True)
    for index, (array, (other, info)) in enumerate(zip(default, detailed, strict=True)):
        np.testing.assert_array_equal(array, other)
        assert info == FrameMetadata(index, *expected[index])
        assert info.time == expected[index][0] * expected[index][1]
        assert isinstance(info.time, Fraction)


def test_skipping_and_downsampling_keep_original_indices_and_timing(vfr_video):
    _, frames = read_frames_from_stream(
        io.BytesIO(vfr_video), skip_frames=1, fps=10, size=FrameSize((32, 24)), return_frame_metadata=True,
    )
    result = list(frames)
    assert [info.index for _, info in result] == [1, 3, 4]
    assert [info.time for _, info in result] == [Fraction(1, 30), Fraction(4, 30), Fraction(10, 30)]
    assert [array.shape for array, _ in result] == [(24, 32, 3)] * 3


def test_resize_without_metadata_keeps_array_output(vfr_video):
    _, frames = read_frames_from_stream(io.BytesIO(vfr_video), size=FrameSize((32, 24)))
    assert [frame.shape for frame in frames] == [(24, 32, 3)] * 6


def test_open_container_is_neither_rewound_nor_closed(vfr_video):
    with av.open(io.BytesIO(vfr_video)) as container:
        first = next(container.decode(video=0))
        _, frames = read_frames_from_stream(container, return_frame_metadata=True)
        result = list(frames)
        assert result[0][1].pts > first.pts
        assert result[0][1].index == 0
        assert len(result) == 5
        container.seek(0)
        assert next(container.decode(video=0)).pts == first.pts


@pytest.mark.parametrize(("pts", "time_base"), [(None, Fraction(1, 30)), (3, None)])
def test_missing_timestamps_remain_unknown(pts, time_base):
    assert FrameMetadata(0, pts, time_base).time is None


def test_invalid_size_is_rejected_before_opening():
    with pytest.raises(TypeError, match="FrameSize"):
        read_frames_from_stream(io.BytesIO(b""), size=(32, 24))


def test_chapter_selection_uses_vfr_timing_and_restarts_fps(vfr_video):
    _, frames = read_frames_from_stream(io.BytesIO(vfr_video))
    original = list(frames)
    result = [(start, list(frames)) for _, start, frames in read_chapters(io.BytesIO(vfr_video), fps=10)]
    assert [start for start, _ in result] == [0, Fraction(1, 10)]
    np.testing.assert_array_equal(result[0][1], [original[0]])
    np.testing.assert_array_equal(result[1][1], [original[2], original[4]])
