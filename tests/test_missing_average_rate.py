"""Containers without a header average rate (e.g. Chrome MediaRecorder WebM).

The fixture is a real Chrome-muxed canvas recording whose paint timing was
jittered on a 120Hz grid (a rAF-driven pump on a ProMotion display does this
naturally): no DefaultDuration, irregular cluster timestamps, so ffprobe
reports ``avg_frame_rate 0/0`` and PyAV's ``stream.average_rate`` is None.
"""

from pathlib import Path

import av
import pytest

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import video_metadata
from simple_video_utils.slicing import slice_video

FIXTURE = str(Path(__file__).parent / "assets" / "mediarecorder-no-average-rate.webm")


@pytest.fixture
def norate_path() -> str:
    with av.open(FIXTURE) as container:
        # the fixture must actually exercise the fallback
        assert container.streams.video[0].average_rate is None
    return FIXTURE


class TestMissingAverageRate:
    def test_metadata_derives_fps(self, norate_path):
        meta = video_metadata(norate_path)

        assert meta.duration is not None
        assert meta.nb_frames is not None
        assert meta.fps == pytest.approx(meta.nb_frames / meta.duration)

    def test_read_frames_exact_time_range(self, norate_path):
        meta = video_metadata(norate_path)

        frames = list(read_frames_exact(norate_path, start_time=0.0, end_time=meta.duration))

        assert len(frames) == meta.nb_frames

    def test_slice_video_reencodes(self, norate_path):
        meta = video_metadata(norate_path)

        [clip] = slice_video(norate_path, [(0.0, meta.duration)], size=64)

        with av.open(__import__("io").BytesIO(clip)) as container:
            stream = container.streams.video[0]
            assert stream.width == stream.height == 64
            assert sum(1 for _ in container.decode(video=0)) > 0
