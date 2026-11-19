import io
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils import slicing
from simple_video_utils.chapters import read_chapters
from simple_video_utils.slicing import slice_video
from simple_video_utils.video import Video


def slice_video_chunks(src, start, end, *, chapters=False):
    return Video.from_slice(src, start, end, chapters=chapters).iter_bytes()


def _write_video(path, frames=300, starts=(0,), sources=None, fps=30):
    with av.open(str(path), mode="w") as output:
        if sources is not None:
            output.set_chapters([
                {"id": i, "start": start, "end": start + frames, "time_base": Fraction(1, fps),
                 "metadata": {"payload": json.dumps({"start_frame": source})}}
                for i, (start, source) in enumerate(zip(starts, sources, strict=True))
            ])
        stream = output.add_stream("h264", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        for start, source in zip(starts, sources or starts, strict=True):
            for offset in range(frames):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), (source + offset) * 2 % 256,
                                                          dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = start + offset, Fraction(1, fps)
                output.mux(stream.encode(frame))
        output.mux(stream.encode())


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mp4"
    _write_video(path)
    return str(path)


def test_slice_does_not_decode_or_encode_frames(source, monkeypatch):
    expected = next(slice_video(source, [(0.1, 2.1)]))

    def forbidden(*args, **kwargs):
        pytest.fail("slicing must only remux encoded packets")

    monkeypatch.setattr(slicing, "read_frames_exact", forbidden)
    monkeypatch.setattr(slicing, "_encode_clip", forbidden)
    monkeypatch.setattr("simple_video_utils.metadata._probe_rotation", forbidden)
    data = b"".join(slice_video_chunks(source, 0.1, 2.1))
    assert data == expected


def test_replay_is_deterministic_and_pipe_decodes_exact_frames(source):
    data = b"".join(slice_video_chunks(source, 0.1, 2.1))
    assert data == b"".join(slice_video_chunks(source, 0.1, 2.1))
    assert data.index(b"moov") < data.index(b"mdat")

    class Pipe:
        def __init__(self):
            self.stream = io.BytesIO(data)

        def read(self, size):
            return self.stream.read(size)

    with av.open(Pipe(), mode="r") as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 61
    assert frames[0].time == 0
    np.testing.assert_allclose([f.to_ndarray(format="rgb24").mean() for f in frames], np.arange(3, 64) * 2, atol=3)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("chapters", [False, True])
def test_packed_and_gapped_source_offsets(tmp_path, packed, chapters):
    path = tmp_path / "source.mkv"
    _write_video(path, frames=15, starts=(0, 15) if packed else (30, 90), sources=(30, 90))
    data = b"".join(slice_video_chunks(str(path), 1.2, 3.2, chapters=chapters))
    with av.open(io.BytesIO(data), mode="r") as container:
        frames = list(container.decode(video=0))
        assert bool(container.chapters()) == chapters
    expected = np.r_[np.arange(36, 45), np.arange(90, 97)] * 2
    np.testing.assert_allclose([f.to_ndarray(format="rgb24").mean() for f in frames], expected, atol=3)
    assert frames[0].time == 0
    if chapters:
        assert [(round(float(start), 3), len(list(items))) for _, start, items in read_chapters(io.BytesIO(data))] == [
            (1.2, 9), (3.0, 7)]
    else:
        assert frames[9].time == pytest.approx(1.8)


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (1, 1), (2, 1), (0, float("inf")), (float("nan"), 1),
                                            (10, 11), (0.001, 0.002)])
def test_invalid_or_empty_ranges_fail(source, start, end):
    with pytest.raises(ValueError, match="finite|range|frames"):
        list(slice_video_chunks(source, start, end))


def test_slice_inside_source_gap_fails(tmp_path):
    path = tmp_path / "source.mkv"
    _write_video(path, frames=15, starts=(30, 90), sources=(30, 90))
    with pytest.raises(ValueError, match="no frames"):
        list(slice_video_chunks(str(path), 2, 2.5))


@pytest.mark.parametrize("name", ["example.webm", "rotated90.mp4", "mediarecorder-no-average-rate.webm"])
def test_existing_codecs_rotation_and_missing_rate(name):
    path = str(Path(__file__).parent / "assets" / name)
    data = b"".join(slice_video_chunks(path, 0, 0.3))
    with av.open(io.BytesIO(data), mode="r") as container:
        assert sum(1 for _ in container.decode(video=0)) > 0


def test_slice_fingerprint_replays_identically(source):
    import hashlib

    passes = []

    def produce():
        chunks = []
        passes.append(chunks)
        for chunk in slice_video_chunks(source, 0, 2):
            chunks.append(chunk)
            yield chunk

    body = Video(produce, 2)
    digest = body.fingerprint()
    for _ in range(10):
        data = b"".join(body.iter_bytes())
        expected = b"".join(passes[0])
        differences = [(index, left, right) for index, (left, right) in enumerate(zip(expected, data, strict=False))
                       if left != right][:16]
        assert digest == hashlib.md5(data).hexdigest(), (len(expected), len(data), differences)
        assert body.size == len(data)


def test_video_slice_duration_includes_boundary_frame(source):
    body = Video.from_slice(source, 0, 5)
    assert body.duration is None  # binding the slice must not decode frames
    assert body.duration_seconds() == pytest.approx(151 / 30, abs=.001)
    assert body.duration > 5
    assert body.digest is None  # duration must not require hashing the whole video


@pytest.mark.parametrize("chapters", [False, True])
def test_video_slice_duration_uses_output_timestamps_across_gaps(tmp_path, chapters):
    path = tmp_path / "source.mkv"
    _write_video(path, frames=15, starts=(30, 300), sources=(30, 300))
    body = Video.from_slice(str(path), 1.2, 10.2, chapters=chapters)
    expected = 16 / 30 if chapters else 271 / 30
    assert body.duration_seconds() == pytest.approx(expected, abs=.001)
    data = b"".join(body.iter_bytes())
    with av.open(io.BytesIO(data), mode="r") as container:
        frames = list(container.decode(video=0))
        assert frames[-1].time + 1 / 30 == pytest.approx(expected, abs=.001)


@pytest.mark.parametrize(("name", "start", "end"), [
    ("example.mp4", 2.7, 3.03), ("example-short.mp4", .1, .68), ("mpeg4.mp4", .1, .3),
    ("rotated90.mp4", .1, .3),
])
def test_streamed_slice_is_identical_to_existing_lossless_copy(name, start, end):
    path = str(Path(__file__).parent / "assets" / name)
    assert b"".join(slice_video_chunks(path, start, end, chapters=True)) == next(
        slice_video(path, [(start, end)], chapters=True))


def test_packet_payload_and_codec_are_preserved(source):
    data = b"".join(slice_video_chunks(source, 0, 2))
    with av.open(source, mode="r") as original, av.open(io.BytesIO(data), mode="r") as sliced:
        assert original.streams.video[0].codec_context.name == sliced.streams.video[0].codec_context.name
        packets = [bytes(p) for p in original.demux(video=0) if p.size]
        copied = [bytes(p) for p in sliced.demux(video=0) if p.size]
    assert copied == packets[:len(copied)]


def test_header_preparation_discards_half_a_gigabyte_of_media():
    import tracemalloc

    output = slicing._SliceOutput(8192, record=True)
    tracemalloc.start()
    try:
        for _ in range(128):
            output.write(bytes(4 * 1024 * 1024))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert output.length == 512 * 1024 * 1024
    assert output.header is not None
    assert len(output.header) == 8192
    assert not output.pending
    assert peak < 20 * 1024 * 1024


@pytest.fixture
def bframes(tmp_path):
    paths = []
    for format in ("mp4", "matroska"):
        path = tmp_path / ("bframes." + ("mp4" if format == "mp4" else "mkv"))
        with av.open(str(path), mode="w", format=format) as output:
            stream = output.add_stream("libx264", rate=30, options={
                "bf": "2", "g": "30", "keyint_min": "30", "sc_threshold": "0",
                "b-adapt": "0", "preset": "ultrafast"})
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(300):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index % 256, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 30)
                output.mux(stream.encode(frame))
            output.mux(stream.encode())
        paths.append(str(path))
    return paths


@pytest.mark.parametrize(("start", "end"), [(1, 6), (1.5, 6.6)])
def test_slice_duration_and_size_match_identical_bytes_with_bframes(bframes, start, end):
    body = Video.from_slice(bframes[0], start, end)
    duration, size = body.duration_seconds(), body.size_bytes()
    assert body.digest is None
    data = b"".join(body.iter_bytes())
    assert duration == Video(data).duration_seconds()
    assert size == len(data)


def test_matroska_seek_keeps_bframe_keyframe_lead_in(bframes):
    frames = []
    for path in bframes:
        body = Video.from_slice(path, 1.37, 4.2)
        data = b"".join(body.iter_bytes())
        assert data == next(slice_video(path, [(1.37, 4.2)]))
        with av.open(io.BytesIO(data)) as container:
            decoded = list(container.decode(video=0))
        assert decoded[0].time == 0
        frames.append(len(decoded))
    assert frames == [87, 87]

@pytest.mark.parametrize("packed", [False, True])
def test_streamed_chapter_tail_clamps_to_last_source_frame(tmp_path, packed):
    path = tmp_path / "tail.mkv"
    _write_video(path, frames=15, starts=(0, 15) if packed else (30, 90), sources=(30, 90))
    exact = Video.from_slice(str(path), 3.2, 3.5, chapters=True)
    overshoot = Video.from_slice(str(path), 3.2, 4, chapters=True)
    assert overshoot.duration_seconds() == exact.duration_seconds()
    assert b"".join(overshoot.iter_bytes()) == b"".join(exact.iter_bytes())
    with pytest.raises(ValueError, match="no frames"):
        list(Video.from_slice(str(path), 3.5, 4, chapters=True).iter_bytes())
