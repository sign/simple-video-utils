import io
import json
from fractions import Fraction

import av
import numpy as np
import pytest

from simple_video_utils.chapters import read_chapters
from simple_video_utils.frames import read_frames_batched, read_frames_exact, read_frames_from_stream
from simple_video_utils.slicing import slice_video, slice_video_stream
from simple_video_utils.timeline import Timeline


@pytest.fixture(params=[("packed", "libx264"), ("gaps", "libx264"), ("packed", "vp8"), ("gaps", "vp8")])
def chaptered(tmp_path, request):
    """The same source frames, stored using either historical or current writer timing."""
    path = tmp_path / "chapters.mkv"
    layout, codec = request.param
    starts = (0, 15) if layout == "packed" else (30, 90)
    with av.open(str(path), "w") as container:
        container.set_chapters([
            {"id": i, "start": start, "end": start + 15, "time_base": Fraction(1, 30),
             "metadata": {"PAYLOAD": json.dumps({"start_frame": original})}}
            for i, (start, original) in enumerate(zip(starts, (30, 90), strict=True))
        ])
        stream = container.add_stream(codec, rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.codec_context.gop_size = 15
        for start, original in zip(starts, (30, 90), strict=True):
            for offset in range(15):
                frame = av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), (original + offset) * 2, dtype=np.uint8), format="rgb24",
                )
                frame.pts, frame.time_base = start + offset, Fraction(1, 30)
                container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return str(path)


def test_mapping_round_trip_boundaries_gaps_and_reconstruction():
    timeline = Timeline([(0, Fraction(1, 2), 1), (Fraction(1, 2), 1, 3)])
    assert timeline.to_file(1.2) == Fraction(1, 5)
    assert timeline.to_source(Fraction(1, 5)) == Fraction(6, 5)
    assert timeline.to_file(2) == Fraction(1, 2)
    assert timeline.to_source(Fraction(1, 2)) == 3
    assert timeline.to_file(3.1) == Fraction(3, 5)
    assert timeline.to_file(3.5) == 1
    assert Timeline(timeline.chapters).to_source(Fraction(3, 5)) == Fraction(31, 10)
    np.testing.assert_allclose(timeline.to_source(np.array([[0.2, 0.6]])), [[1.2, 3.1]])
    np.testing.assert_allclose(timeline.to_file(np.array([1.2, 2, 3.1])), [0.2, 0.5, 0.6])


def test_native_gaps_and_unchaptered_times_are_identity():
    for timeline in (Timeline(), Timeline([(1, 1.5, 1), (3, 3.5, 3)])):
        for time in (0, 1.2, 2, 3.1, 4):
            assert timeline.to_file(time) == timeline.to_source(time) == Fraction(str(time))


def test_fractional_cadence_rounding_does_not_move_the_next_chapter_start():
    timeline = Timeline([(0, Fraction(334, 1000), 0), (Fraction(334, 1000), Fraction(667, 1000), Fraction(1001, 3000))])
    assert timeline.to_file(Fraction(1001, 3000)) == Fraction(334, 1000)
    assert timeline.to_source(Fraction(334, 1000)) == Fraction(1001, 3000)


@pytest.mark.parametrize("chapters", [[(0, 0, 1)], [(0, 1, 1), (0.5, 1.5, 2)], [(0, 1)]])
def test_invalid_native_chapters_are_rejected(chapters):
    with pytest.raises(ValueError, match="chapters must"):
        Timeline(chapters)


def test_backward_source_chapters_still_report_times_but_cannot_be_inverted():
    timeline = Timeline([(0, 1, 5), (1, 2, 3)])
    assert timeline.to_source(1.2) == Fraction(16, 5)
    with pytest.raises(ValueError, match="source order"):
        timeline.to_file(3.2)


def test_timeline_read_preserves_open_container_position(chaptered):
    with av.open(chaptered) as container:
        decoded = container.decode(video=0)
        first = next(decoded)
        timeline = Timeline.read(container)
        assert next(decoded).pts > first.pts
        assert timeline.chapters == Timeline.read(chaptered).chapters


def test_time_reads_and_batch_indices_use_original_video_seconds(chaptered):
    all_frames = list(read_frames_exact(chaptered))
    result = list(read_frames_exact(chaptered, start_time=3.1, end_time=3.3, return_indices=True))
    assert [index for _, index in result] == list(range(18, 25))
    np.testing.assert_array_equal([frame for frame, _ in result], all_frames[18:25])
    frames, indices = read_frames_batched(chaptered, start_time=3.1, end_time=3.3, return_indices=True)
    np.testing.assert_array_equal(frames, all_frames[18:25])
    np.testing.assert_array_equal(indices, np.arange(18, 25))
    np.testing.assert_array_equal(list(read_frames_exact(chaptered, start_time=2)), all_frames[15:])
    np.testing.assert_array_equal(list(read_frames_exact(chaptered, end_time=2)), all_frames[:15])
    assert list(read_frames_exact(chaptered, start_time=2, end_time=2.5)) == []


def test_metadata_exposes_source_time_and_retains_native_time(chaptered):
    with open(chaptered, "rb") as file:
        _, frames = read_frames_from_stream(file, return_frame_metadata=True)
        info = [metadata for _, metadata in frames]
    timeline = Timeline.read(chaptered)
    assert info[0].time == 1
    assert info[15].time == 3
    assert info[18].time == Fraction(31, 10)
    assert all(frame.file_time == frame.pts * frame.time_base for frame in info)
    assert info[18].file_time == timeline.to_file(Fraction(31, 10))
    assert [frame.index for frame in info] == list(range(30))


def test_chapter_iteration_uses_native_boundaries_and_source_starts(chaptered):
    result = [(start, len(list(frames))) for _, start, frames in read_chapters(chaptered)]
    assert result == [(1, 15), (3, 15)]


def test_fps_selection_uses_source_time_for_both_layouts(chaptered):
    with open(chaptered, "rb") as file:
        _, frames = read_frames_from_stream(file, fps=7, return_frame_metadata=True)
        indices = [info.index for _, info in frames]
    assert indices == [0, 5, 9, 13, 15, 20, 24, 28]
    selected = list(read_frames_exact(chaptered, fps=7, return_indices=True))
    assert [index for _, index in selected] == indices


@pytest.mark.parametrize("size", [None, 64])
def test_slices_use_source_times_and_do_not_include_the_next_chapter_from_a_gap(chaptered, size):
    for bounds, count, mean in (((3.1, 3.3), 7, 186), ((1.1, 2.5), 12, 66)):
        [clip] = slice_video(chaptered, [bounds], size=size)
        with av.open(io.BytesIO(clip)) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == count
        assert frames[0].time == 0
        assert frames[0].to_ndarray(format="rgb24").mean() == pytest.approx(mean, abs=3)
    with pytest.raises(ValueError, match="has no frames"):
        list(slice_video(chaptered, [(2, 2.5)], size=size))
    with pytest.raises(ValueError, match="out of range"):
        list(slice_video(chaptered, [(3, 4)], size=size))


def test_stream_windows_report_source_starts_and_skip_removed_gaps(chaptered):
    with open(chaptered, "rb") as file:
        clips = list(slice_video_stream(file, duration=0.5))
    assert [clip.start for clip in clips] == [1, 3]
    for clip in clips:
        with av.open(io.BytesIO(clip.data)) as container:
            assert sum(1 for _ in container.decode(video=0)) == 15
