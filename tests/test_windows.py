"""The training decode: open_video(thread_type="NONE"), metadata, then a read_frames_batched window.

Chaptered videos are read in source frames, so packed (before 0.10) and gapped (since 0.10) files read the same.
"""

import io
import json
import math
import multiprocessing
import subprocess
from fractions import Fraction

import av
import numpy as np
import pytest

from simple_video_utils import frames as frames_module
from simple_video_utils.frames import FrameSize, read_frames_batched, read_frames_exact
from simple_video_utils.metadata import count_frames, open_video, video_metadata, video_metadata_from_container
from simple_video_utils.slicing import slice_video

SIZE = FrameSize((32, 32))
CLIPS = [(0, 2000), (2500, 2000), (5000, 2000)]  # (source start frame, length) at 25 fps: 0-80 s, 100-180 s, 200-280 s


def _picture(source_index):
    array = np.zeros((64, 64, 3), np.uint8)
    array[:, :, 0] = source_index * 37 % 256
    array[: source_index % 64, :, 1] = 255
    array[:, : source_index // 64 % 64, 2] = 255
    return array


def _write(path, clips, rate=25, *, layout=None):
    """Frames picture their source index. ``layout`` "packed" or "gapped" stores ``clips`` as chapters."""
    rate = Fraction(rate).limit_denominator(1001)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 25
        stream.options = {"preset": "ultrafast", "bf": "0"}
        offset, marks, positions = 0, [], []
        for start, length in clips:
            first = start if layout == "gapped" else offset
            marks.append({"id": len(marks), "start": round(first * 1000 / rate),
                          "end": round((first + length) * 1000 / rate), "time_base": Fraction(1, 1000),
                          "metadata": {"payload": json.dumps({"start_frame": start, "end_frame": start + length})}})
            positions += [(first + k, start + k) for k in range(length)]
            offset += length
        if layout:
            container.set_chapters(marks)
        for position, source in positions:
            frame = av.VideoFrame.from_ndarray(_picture(source), format="rgb24")
            frame.pts, frame.time_base = position, 1 / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return str(path)


@pytest.fixture(scope="module", params=["packed", "gapped", "mp4"])
def video(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp(request.param)
    if request.param == "mp4":
        return _write(folder / "plain.mp4", [(0, 6000)])
    return _write(folder / f"{request.param}.mkv", CLIPS, layout=request.param)


@pytest.fixture(scope="module")
def truth(video):
    """Every frame the video holds, by source index, from one decode without seeking."""
    frames, indices = read_frames_batched(video, return_indices=True, size=SIZE)
    return dict(zip(indices.tolist(), frames, strict=True))


def _held(video, start, end):
    clips = [(0, 6000)] if video.endswith(".mp4") else CLIPS
    return [i for first, length in clips for i in range(first, first + length) if start <= i <= end]


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


def test_late_window_costs_what_an_early_one_does(video, truth):
    """0.10.0 decoded chaptered files from frame 0, so cost grew with the offset (edge 1); reads stay local (13)."""
    size = len(open(video, "rb").read())
    costs = []
    for start in (50, 2600, 6900 if video.endswith(".mkv") else 5800):
        meta, frames, indices, metadata_bytes, total = _training_read(video, start, start + 99)
        assert meta.nb_frames == 6000
        assert metadata_bytes < size / 10  # the header, the final GOP and one probed frame
        assert indices.tolist() == list(range(start, start + 100))
        assert all(np.array_equal(frame, truth[index]) for frame, index in zip(frames, indices, strict=True))
        costs.append(total - metadata_bytes)
    assert max(costs) < 3 * min(costs) + 64 * 1024
    assert max(costs) < size / 10


@pytest.mark.parametrize(("start", "end"), [(1950, 2549), (2100, 2400), (1990, 7100), (4400, 5100)])
def test_windows_hold_only_the_frames_the_chapters_keep(video, truth, start, end):
    """Source frames cropped out of a chaptered video are absent (edges 2, 3); packed and gapped agree."""
    held = _held(video, start, end)
    if not held:
        with pytest.raises(ValueError, match="no frames"):
            read_frames_batched(video, start_frame=start, end_frame=end, size=SIZE)
        return
    frames, indices = read_frames_batched(video, start_frame=start, end_frame=end, return_indices=True, size=SIZE)
    assert indices.tolist() == held
    assert all(np.array_equal(frame, truth[index]) for frame, index in zip(frames, indices, strict=True))


@pytest.mark.parametrize("native", [15, 25, Fraction(30000, 1001), 30])
@pytest.mark.parametrize("suffix", ["mkv", "mp4"])
def test_target_at_or_above_native_keeps_every_frame(tmp_path, native, suffix):
    """Matroska's millisecond clock stores a 15 fps file as 0, 67, 133 ms; 0.9.8 dropped about a third at fps=15."""
    path = _write(tmp_path / f"v.{suffix}", [(0, 150)], native)
    for fps in (float(native), float(native) + 1):
        assert len(list(read_frames_exact(path, fps=fps, size=SIZE))) == 150


@pytest.mark.parametrize(("start", "end"), [(1900, 2600), (1000, 5500), (0, 6999)])
def test_fps_selection_keeps_one_grid_across_chapters(video, truth, start, end):
    """Gaps don't restart the selection grid (edge 4): N held frames keep int((N-1)*fps/native)+1, and each
    returned index names its frame (edge 9)."""
    _, frames, indices, _, _ = _training_read(video, start, end, fps=10)
    assert len(frames) == len(indices) == int((len(_held(video, start, end)) - 1) * 10 / 25) + 1
    assert all(np.array_equal(frame, truth[index]) for frame, index in zip(frames, indices, strict=True))


def test_a_time_range_between_two_frames_is_empty(video):
    for start in (0.705, 100.705):  # frames at 25 fps sit at .68 and .72
        assert list(read_frames_exact(video, start_time=start, end_time=start + 0.01)) == []


def test_time_range_is_the_frames_timed_inside_it(video):
    """Canonical time path (edge 6): source frames ceil(start*fps) through floor(end*fps), as the frame path."""
    for start, end in ((0.7, 1.3), (0.71, 1.29), (79.98, 100.04), (199.96, 200.04)):
        first, last = math.ceil(round(start * 25, 6)), math.floor(round(end * 25, 6))
        by_time = read_frames_batched(video, start_time=start, end_time=end, size=SIZE)
        by_frame = read_frames_batched(video, start_frame=first, end_frame=last, size=SIZE)
        assert np.array_equal(by_time, by_frame)


def test_edit_list_mp4_hides_its_lead_in(tmp_path):
    """An ``ffmpeg -ss … -c copy`` trim keeps a keyframe lead-in behind an edit list (edge 7)."""
    source = _write(tmp_path / "source.mp4", [(0, 3500)])
    trimmed = str(tmp_path / "trimmed.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "100.4", "-i", source, "-t", "20", "-c", "copy", trimmed],
                   check=True)
    full = list(read_frames_exact(trimmed, size=SIZE))
    assert video_metadata(trimmed).nb_frames == count_frames(trimmed) == len(full) == 500
    expected = list(read_frames_exact(source, start_frame=2510, end_frame=3009, size=SIZE))
    assert all(np.array_equal(got, want) for got, want in zip(full, expected, strict=True))
    _, frames, _, _, _ = _training_read(trimmed, 300, 399)
    assert all(np.array_equal(got, want) for got, want in zip(frames, full[300:400], strict=True))


def test_selection_follows_timestamps_not_the_header_rate(tmp_path):
    """A header claiming 120 fps over 30 fps timestamps (fsboard, edge 8) must not change which frames are kept."""
    path = _write(tmp_path / "v.mkv", [(0, 300)], 30)
    claimed = tmp_path / "claimed.mkv"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-c", "copy", "-r", "120", str(claimed)], check=True)
    kept = list(read_frames_exact(str(claimed), fps=10, return_indices=True, size=SIZE))
    assert [index for _, index in kept] == [index for _, index in read_frames_exact(path, fps=10, return_indices=True)]


def _decode_in_worker(path):
    with open_video(path, thread_type="NONE") as video:
        video_metadata_from_container(video)
        return len(read_frames_batched(video, start_frame=2600, end_frame=2699, fps=10, size=SIZE))


def test_thread_type_none_holds_for_the_container_and_forked_workers(video):
    """open_video's policy survives a read issued before any metadata call, and forked workers decode (edge 10)."""
    with open_video(video, thread_type="NONE") as container:
        read_frames_batched(container, start_frame=2600, end_frame=2610)
        assert container.streams.video[0].thread_type.name == "NONE"
    list(read_frames_exact(video, start_frame=0, end_frame=50))  # threaded decoders alive in the parent
    with multiprocessing.get_context("fork").Pool(2) as pool:
        assert pool.map_async(_decode_in_worker, [video] * 4).get(timeout=60) == [40] * 4


def test_fps_drops_before_rgb_and_the_batch_is_preallocated(video, monkeypatch):
    """Only kept frames reach RGB conversion, and the hint covers the batch (edge 11)."""
    converted, hints = [], []
    to_rgb, stack = frames_module._frames_to_rgb, frames_module.stack_frames
    monkeypatch.setattr(frames_module, "_frames_to_rgb", lambda frames, size: to_rgb(
        (converted.append(1) or frame for frame in frames), size))
    monkeypatch.setattr(frames_module, "stack_frames", lambda frames, size_hint: hints.append(size_hint) or stack(
        frames, size_hint))
    batch = read_frames_batched(video, start_frame=3000, end_frame=3999, fps=10, size=SIZE)
    assert len(converted) == len(batch) == 400
    assert 0 <= hints[0] - len(batch) <= 1


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("layouts")
    return [_write(folder / f"{layout}.mkv", CLIPS, layout=layout) for layout in ("packed", "gapped")]


@pytest.mark.parametrize("size", [None, 64])
@pytest.mark.parametrize(("start", "end", "held"), [
    (110.04, 112.0, range(2751, 2801)),  # inside a chapter
    (78.0, 102.0, [*range(1950, 2000), *range(2500, 2551)]),  # across a gap: kept
    (85.0, 102.0, range(2500, 2551)),  # from inside a gap: dropped
    (70.0, 80.0, range(1750, 2000)),  # to a chapter's end: not the next chapter's first frame
])
def test_slices_take_source_seconds_on_either_layout(layouts, size, start, end, held):
    """Copied and re-encoded slices (edge 12) start at t=0 on their first frame, keep gaps between chapters and
    drop a leading one, identically on packed and gapped files."""
    for path in layouts:
        [clip] = slice_video(path, [(start, end)], size=size)
        with av.open(io.BytesIO(clip)) as container:
            times = [round(frame.time, 3) for frame in container.decode(video=0) if frame.time >= 0]
        assert times == [round((index - held[0]) / 25, 3) for index in held]


def test_a_slice_wholly_cropped_out_raises(layouts):
    for path in layouts:
        with pytest.raises(ValueError, match="no frames"):
            next(slice_video(path, [(82.0, 98.0)]))


def test_nb_frames_counts_only_held_frames(tmp_path):
    """A gapped file starting late and skipping ahead holds 200 frames, not its 40 s span."""
    path = _write(tmp_path / "late.mkv", [(500, 100), (900, 100)], layout="gapped")
    assert video_metadata(path).nb_frames == count_frames(path) == 200
