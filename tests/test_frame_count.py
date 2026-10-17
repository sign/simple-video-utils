from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils import frames as frames_module
from simple_video_utils.chapters import ChapterClip, write_chapters
from simple_video_utils.frames import FrameSize, read_frames_batched, read_frames_exact

ASSETS = Path(__file__).parent / "assets"


@pytest.mark.parametrize("filename", ["no_nb_frames.mkv", "no_nb_frames.webm"])
@pytest.mark.parametrize("use_container", [False, True])
def test_duration_hint_avoids_metadata_scan(monkeypatch, filename, use_container):
    path = str(ASSETS / filename)
    size = FrameSize((16, 16))
    expected = np.stack(list(read_frames_exact(path, size=size)))

    def fail_metadata(*args, **kwargs):
        pytest.fail("scanned the file for metadata")

    monkeypatch.setattr(frames_module, "video_metadata", fail_metadata)
    monkeypatch.setattr(frames_module, "video_metadata_from_container", fail_metadata)
    frames_module._count_and_rate_cached.cache_clear()
    with av.open(path) as container:
        assert not container.streams.video[0].frames
        result = read_frames_batched(container if use_container else path, size=size)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("hint", [1, 200])
def test_inaccurate_hint_keeps_every_frame(monkeypatch, hint):
    path = str(ASSETS / "no_nb_frames.mkv")
    size = FrameSize((16, 16))
    expected = np.stack(list(read_frames_exact(path, size=size)))
    frames_module._count_and_rate_cached.cache_clear()
    monkeypatch.setattr(frames_module, "_header_frames", lambda *args: hint)
    with av.open(path) as container:
        result = read_frames_batched(container, size=size)
    np.testing.assert_array_equal(result, expected)


def test_chapter_hint_counts_exported_frames(tmp_path):
    output = tmp_path / "chapters.mkv"
    write_chapters(ASSETS / "example.mp4", output, [
        ChapterClip(30, 32, (0, 0, 32, 32)),
        ChapterClip(90, 92, (0, 0, 32, 32)),
    ])
    path = str(output)
    hint, _ = frames_module._count_and_rate(path)
    assert 4 <= hint <= 5  # Allow Matroska's millisecond duration rounding.
    assert len(read_frames_batched(path)) == 4
