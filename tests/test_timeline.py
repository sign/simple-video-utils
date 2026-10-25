from pathlib import Path

import numpy as np
import pytest

from simple_video_utils.chapters import ChapterClip, Timeline, write_chapters
from simple_video_utils.metadata import open_video

# Two clips of a 60 s source: 10–20 s at file 0–10 s, 40–45 s at file 10–15 s.
T = Timeline([(0, 10, 10), (10, 15, 40)])


def test_to_file_maps_chapters_and_collapses_gaps():
    assert T.to_file(12) == 2
    assert T.to_file(41) == 11
    assert T.to_file(30) == T.to_file(40) == 10  # inside the gap: the next chapter's start


def test_to_file_clamps_outside_the_chapters():
    assert T.to_file(5) == T.to_file(-1) == 0
    assert T.to_file(50) == T.to_file(1e9) == 15
    np.testing.assert_array_equal(T.to_file(np.array([0, 12, 30, 99])), [0, 2, 10, 15])


def test_window_is_none_without_video():
    assert T.window(12, 41) == (2, 11)
    assert T.window(25, 35) is None  # wholly in the gap
    assert T.window(0, 9) is None  # before the first chapter
    assert T.window(45, 50) is None  # past the last


def test_to_source_inverts_and_clamps():
    assert T.to_source(2) == 12
    np.testing.assert_array_equal(T.to_source(np.array([-1, 0, 9.5, 10, 14, 16])), [10, 10, 19.5, 40, 44, 45])


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_times_raise(time):
    with pytest.raises(ValueError, match="finite"):
        T.to_file(time)
    with pytest.raises(ValueError, match="finite"):
        Timeline().to_source(np.array([1.0, time]))


def test_without_chapters_times_map_to_themselves():
    assert Timeline().to_file(7.5) == Timeline().to_source(7.5) == 7.5
    assert Timeline().window(1, 2) == (1, 2)


@pytest.mark.parametrize("chapters", [[(0, 1, 5), (1, 2, 3)], [(1, 1, 0)]])
def test_backward_or_empty_chapters_raise(chapters):
    with pytest.raises(ValueError, match="chapters"):
        Timeline(chapters)


def test_read_from_written_chapters(tmp_path):
    source = Path(__file__).parent / "assets" / "example.mp4"  # 30 fps
    output = tmp_path / "chapters.mkv"
    write_chapters(source, output, [ChapterClip(30, 60, (0, 0, 64, 64)), ChapterClip(90, 120, (0, 0, 64, 64))])
    timeline = Timeline.read(output)
    np.testing.assert_allclose(timeline.chapters, [(0, 1, 1), (1, 2, 3)], atol=1e-3)
    assert timeline.to_file(2.5) == pytest.approx(1, abs=1e-3)
    with open_video(output, thread_type="NONE") as video:
        np.testing.assert_array_equal(Timeline.read(video).chapters, timeline.chapters)
    np.testing.assert_array_equal(Timeline(timeline.chapters).chapters, timeline.chapters)
    assert Timeline.read(source).chapters.shape == (0, 3)
