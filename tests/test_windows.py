"""The training decode: open_video(thread_type="NONE"), metadata, then a read_frames_batched window."""

import io
import json
import math
import multiprocessing
import subprocess
from fractions import Fraction
from itertools import islice

import av
import numpy as np
import pytest

from simple_video_utils import frames as frames_module
from simple_video_utils.chapters import Timeline
from simple_video_utils.frames import FrameSize, read_frames_batched, read_frames_exact
from simple_video_utils.metadata import count_frames, open_video, video_metadata, video_metadata_from_container
from simple_video_utils.slicing import slice_video

SIZE = FrameSize((32, 32))


def _write(path, n, rate=25, *, gop=25, chapters=(), time_base=None):
    """``n`` distinct frames at ``rate``; ``chapters`` are packed (source start frame, length) clips."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(rate).limit_denominator(1001))
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = gop
        stream.options = {"preset": "ultrafast", "bf": "0"}
        if time_base:
            stream.time_base = time_base
        offset, marks = 0, []
        for start, length in chapters:
            ends = (round(offset * 1000 / rate), round((offset + length) * 1000 / rate))
            marks.append({"id": len(marks), "start": ends[0], "end": ends[1], "time_base": Fraction(1, 1000),
                          "metadata": {"payload": json.dumps({"start_frame": start, "end_frame": start + length})}})
            offset += length
        if marks:
            container.set_chapters(marks)
        for i in range(n):
            array = np.zeros((64, 64, 3), np.uint8)
            array[:, :, 0], array[: i % 64, :, 1], array[:, : i // 64 % 64, 2] = i * 37 % 256, 255, 255
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return str(path)


@pytest.fixture(scope="module")
def long_mkv(tmp_path_factory):
    """Four minutes at 25 fps: three packed chapters cut from source frames 0-2000, 2500-4500 and 5000-7000."""
    return _write(tmp_path_factory.mktemp("long") / "long.mkv", 6000,
                  chapters=[(0, 2000), (2500, 2000), (5000, 2000)])


@pytest.fixture(scope="module")
def long_mp4(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("long") / "long.mp4", 6000)


class _Counting(io.FileIO):
    read_bytes = 0

    def read(self, size=-1):
        data = super().read(size)
        self.read_bytes += len(data)
        return data


def _training_read(path, start, end, fps=None):
    with _Counting(path) as file, open_video(file, thread_type="NONE") as video:
        meta = video_metadata_from_container(video)
        metadata_bytes = file.read_bytes
        frames, indices = read_frames_batched(video, start_frame=start, end_frame=end, fps=fps,
                                              return_indices=True, size=SIZE)
        return meta, frames, indices, metadata_bytes, file.read_bytes


def _full(path):
    return list(read_frames_exact(path, size=SIZE))


@pytest.mark.parametrize("name", ["long_mkv", "long_mp4"])
def test_late_window_costs_what_an_early_one_does(name, request):
    """0.10.0 decoded from frame 0, so cost grew with the offset (edge 1); a window's reads must not scan the file."""
    path = request.getfixturevalue(name)
    size = len(open(path, "rb").read())
    full = _full(path)
    costs = []
    for start in (50, 2500, 5800):
        meta, frames, indices, metadata_bytes, total = _training_read(path, start, start + 99)
        assert meta.nb_frames == 6000
        assert metadata_bytes < size / 10  # the header, the final GOP and one probed frame
        assert indices.tolist() == list(range(start, start + 100))
        assert all(np.array_equal(got, want) for got, want in zip(frames, full[start:start + 100], strict=True))
        costs.append(total - metadata_bytes)
    assert max(costs) < 3 * min(costs) + 64 * 1024
    assert max(costs) < size / 10


@pytest.mark.parametrize("native", [15, 25, Fraction(30000, 1001), 30])
@pytest.mark.parametrize("suffix", ["mkv", "mp4"])
def test_target_at_or_above_native_keeps_every_frame(tmp_path, native, suffix):
    """Matroska's millisecond clock stores a 15 fps file as 0, 67, 133 ms; 0.9.8 dropped about a third at fps=15."""
    path = _write(tmp_path / f"v.{suffix}", 150, native)
    for fps in (float(native), float(native) + 1):
        assert len(list(read_frames_exact(path, fps=fps, size=SIZE))) == 150


def test_selection_runs_on_one_grid_across_packed_chapters(long_mkv):
    """Frame dropping does not restart at chapter boundaries (edge 4): a CFR window keeps int((N-1)*fps/native)+1."""
    for start, end in ((1900, 2600), (1000, 5500), (0, 5999)):
        _, frames, indices, _, _ = _training_read(long_mkv, start, end, fps=10)
        assert len(frames) == len(indices) == int((end - start) * 10 / 25) + 1
        assert set(np.diff(indices)) <= {2, 3}


def test_returned_indices_point_at_the_returned_frames(long_mkv):
    """After seeking and fps selection, each index names the frame returned beside it (edge 9)."""
    full = _full(long_mkv)
    _, frames, indices, _, _ = _training_read(long_mkv, 4321, 4999, fps=7)
    assert len(frames) == len(indices) > 100
    assert all(np.array_equal(frame, full[index]) for frame, index in zip(frames, indices, strict=True))


@pytest.mark.parametrize("name", ["long_mkv", "long_mp4"])
def test_time_range_is_the_frames_timed_inside_it(name, request):
    """Canonical time path (edge 6): frames ceil(start*fps) through floor(end*fps), identical to the frame path."""
    path = request.getfixturevalue(name)
    for start, end in ((0.7, 1.3), (0.71, 1.29), (100.02, 101.0), (199.96, 200.04)):
        first, last = math.ceil(round(start * 25, 6)), math.floor(round(end * 25, 6))
        by_time = read_frames_batched(path, start_time=start, end_time=end, size=SIZE)
        by_frame = read_frames_batched(path, start_frame=first, end_frame=last, size=SIZE)
        assert np.array_equal(by_time, by_frame)


def test_edit_list_mp4_hides_its_lead_in(long_mp4, tmp_path):
    """An ``ffmpeg -ss … -c copy`` trim keeps a keyframe lead-in behind an edit list (edge 7)."""
    trimmed = str(tmp_path / "trimmed.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "100.4", "-i", long_mp4, "-t", "20", "-c", "copy", trimmed],
                   check=True)
    full = _full(trimmed)
    meta = video_metadata(trimmed)
    assert meta.nb_frames == count_frames(trimmed) == len(full) == 500
    source = _full(long_mp4)
    assert all(np.array_equal(got, want) for got, want in zip(full, source[2510:3010], strict=True))
    _, frames, indices, _, _ = _training_read(trimmed, 300, 399)
    assert all(np.array_equal(got, want) for got, want in zip(frames, full[300:400], strict=True))


def test_selection_follows_timestamps_not_the_header_rate(tmp_path):
    """A header claiming 120 fps over 30 fps timestamps (fsboard, edge 8) must not change which frames are kept."""
    path = _write(tmp_path / "v.mkv", 300, 30)
    claimed = tmp_path / "claimed.mkv"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-c", "copy", "-r", "120", str(claimed)], check=True)
    with av.open(str(claimed)) as container:
        assert container.streams.video[0].average_rate != 30 or pytest.skip("ffmpeg kept the true rate")
    kept = list(read_frames_exact(str(claimed), fps=10, return_indices=True, size=SIZE))
    assert [index for _, index in kept] == [index for _, index in read_frames_exact(path, fps=10, return_indices=True)]


def _decode_in_worker(path):
    with open_video(path, thread_type="NONE") as video:
        video_metadata_from_container(video)
        return len(read_frames_batched(video, start_frame=2500, end_frame=2599, fps=10, size=SIZE))


def test_thread_type_none_holds_for_the_container_and_forked_workers(long_mkv):
    """open_video's policy survives a read issued before any metadata call, and forked workers decode (edge 10)."""
    with open_video(long_mkv, thread_type="NONE") as video:
        read_frames_batched(video, start_frame=2500, end_frame=2510)
        assert video.streams.video[0].thread_type.name == "NONE"
    list(read_frames_exact(long_mkv, start_frame=0, end_frame=50))  # threaded decoders alive in the parent
    with multiprocessing.get_context("fork").Pool(2) as pool:
        assert pool.map_async(_decode_in_worker, [long_mkv] * 4).get(timeout=60) == [40] * 4


def test_fps_drops_before_rgb_and_the_batch_is_preallocated(long_mkv, monkeypatch):
    """Only kept frames reach RGB conversion, and the header-sized hint covers the batch (edge 11)."""
    converted, hints = [], []
    to_rgb, stack = frames_module._frames_to_rgb, frames_module.stack_frames
    monkeypatch.setattr(frames_module, "_frames_to_rgb", lambda frames, size: to_rgb(
        (converted.append(1) or frame for frame in frames), size))
    monkeypatch.setattr(frames_module, "stack_frames", lambda frames, size_hint: hints.append(size_hint) or stack(
        frames, size_hint))
    batch = read_frames_batched(long_mkv, start_frame=3000, end_frame=3999, fps=10, size=SIZE)
    assert len(converted) == len(batch) == 400
    assert 0 <= hints[0] - len(batch) <= 1


def test_slices_use_file_seconds_and_both_paths_agree(long_mkv):
    """slice_video cuts file seconds (edge 12): map source spans with Timeline.window. Copy and re-encode start
    on the same frame, at t=0."""
    timeline = Timeline.read(long_mkv)
    start, end = timeline.window(110.04, 112.0)  # source frames 2751-2800: file frames 2251-2300
    assert (start, end) == pytest.approx((90.04, 92.0))
    copied, encoded = next(slice_video(long_mkv, [(start, end)])), next(slice_video(long_mkv, [(start, end)], size=64))
    for clip in (copied, encoded):
        with av.open(io.BytesIO(clip)) as container:
            times = [frame.time for frame in container.decode(video=0)]
        assert times[0] == pytest.approx(0, abs=1e-3)
    expected = read_frames_batched(long_mkv, start_time=start, end_time=end, size=SIZE)
    assert len(expected) == 50
    got = list(islice(read_frames_exact(io.BytesIO(copied), size=SIZE), 50))
    assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))
