import io
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils import Video, slicing
from simple_video_utils.chapters import read_chapters
from simple_video_utils.slicing import slice_video_chunks


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


def test_yields_before_decoding_whole_slice_and_closes(source, monkeypatch):
    read, closed = [], []
    original = slicing.read_frames_exact

    def tracked(*args, **kwargs):
        try:
            for frame in original(*args, **kwargs):
                read.append(frame[1])
                yield frame
        finally:
            closed.append(True)

    monkeypatch.setattr(slicing, "read_frames_exact", tracked)
    chunks = slice_video_chunks(source, 0, 9)
    assert read == []
    assert next(chunks)
    assert len(read) < 271
    chunks.close()
    assert closed == [True]


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
                                            (0, 11), (0.001, 0.002)])
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


@pytest.mark.anyio
async def test_slice_and_async_body_fingerprint_replay(source):
    import asyncio
    import hashlib

    passes = []

    def produce():
        chunks = []
        passes.append(chunks)
        for chunk in slice_video_chunks(source, 0, 2):
            chunks.append(chunk)
            yield chunk

    body = Video(produce, 2)
    digest = await body.fingerprint()
    await asyncio.sleep(1.1)  # A new wall-clock second must not change container metadata.
    for _ in range(10):
        data = b"".join([part async for part in body.chunks()])
        expected = b"".join(passes[0])
        differences = [(index, left, right) for index, (left, right) in enumerate(zip(expected, data, strict=False))
                       if left != right][:16]
        assert digest == hashlib.md5(data).hexdigest(), (len(expected), len(data), differences)
        assert body.size == len(data)



@pytest.fixture
def anyio_backend():
    return "asyncio"
