"""Containers without a header average rate (e.g. Chrome MediaRecorder WebM).

``mediarecorder-no-average-rate.webm`` is a real Chrome-muxed canvas recording
whose paint timing was jittered on a 120Hz grid (a rAF-driven pump on a
ProMotion display does this naturally): no DefaultDuration, irregular cluster
timestamps, so ffprobe reports ``avg_frame_rate 0/0`` and PyAV's
``stream.average_rate`` is None.

``no_average_rate_5s.webm`` is synthetic (PyAV-muxed VP8, 100 frames jittered
on the same 120Hz grid, DefaultDuration overwritten with an EBML Void): the
real fixture is 2.27s, too short to reach the seek path
(``_seek_near.min_seek_seconds`` is 3.0). Its ``guessed_rate`` is 24 — wrong,
like the 1000 the real fixture reports — so seeked reads only pass if frames
are located with the derived rate.
"""

import io
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.frames import read_frames_batched, read_frames_exact, read_frames_from_stream
from simple_video_utils.metadata import video_metadata
from simple_video_utils.slicing import slice_video

ASSETS = Path(__file__).parent / "assets"
FIXTURE = str(ASSETS / "mediarecorder-no-average-rate.webm")
LONG_FIXTURE = str(ASSETS / "no_average_rate_5s.webm")


def _decoded_ground_truth(path: str) -> tuple[float, int]:
    """True cadence and count, independent of the code under test."""
    with av.open(path) as container:
        # the fixtures must actually exercise the fallback
        assert container.streams.video[0].average_rate is None
        times = [frame.time for frame in container.decode(video=0)]
    return (len(times) - 1) / (times[-1] - times[0]), len(times)


class TestMissingAverageRate:
    def test_metadata_derives_fps(self):
        true_fps, true_count = _decoded_ground_truth(FIXTURE)

        meta = video_metadata(FIXTURE)

        assert meta.fps == pytest.approx(true_fps)
        assert meta.nb_frames == true_count
        assert meta.duration is not None

    def test_read_frames_exact_time_range(self):
        meta = video_metadata(FIXTURE)

        frames = list(read_frames_exact(FIXTURE, start_time=0.0, end_time=meta.duration))

        assert len(frames) == meta.nb_frames

    def test_seeked_read_returns_the_right_frames(self):
        meta = video_metadata(LONG_FIXTURE)
        full = list(read_frames_exact(LONG_FIXTURE))
        start = int(4.0 * meta.fps)  # past min_seek_seconds, so _seek_near fires

        seeked = list(read_frames_exact(LONG_FIXTURE, start_time=4.0, end_time=meta.duration))

        assert len(full) == meta.nb_frames
        assert len(seeked) == len(full) - start
        for got, expected in zip(seeked, full[start:], strict=True):
            assert np.array_equal(got, expected)

    def test_stream_metadata_matches_path_metadata(self):
        path_meta = video_metadata(FIXTURE)

        stream_meta, frames = read_frames_from_stream(io.BytesIO(Path(FIXTURE).read_bytes()))

        assert stream_meta.fps == pytest.approx(path_meta.fps)
        assert stream_meta.nb_frames == path_meta.nb_frames
        assert sum(1 for _ in frames) == path_meta.nb_frames

    def test_read_frames_batched(self):
        meta = video_metadata(FIXTURE)

        batch = read_frames_batched(FIXTURE)

        assert batch.shape == (meta.nb_frames, meta.height, meta.width, 3)

    def test_slice_video_reencodes(self):
        meta = video_metadata(FIXTURE)

        [clip] = slice_video(FIXTURE, [(0.0, meta.duration)], size=64)

        with av.open(io.BytesIO(clip)) as container:
            stream = container.streams.video[0]
            assert stream.width == stream.height == 64
            assert sum(1 for _ in container.decode(video=0)) > 0
