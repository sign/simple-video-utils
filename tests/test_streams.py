import tracemalloc

import pytest

from simple_video_utils.video import Video


def test_hashing_half_a_gigabyte_does_not_retain_the_video():
    def source():
        for _ in range(128):
            yield bytes(4 * 1024 * 1024)

    body = Video(source, 600)
    tracemalloc.start()
    try:
        body.fingerprint()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert body.size == 512 * 1024 * 1024
    assert peak < 20 * 1024 * 1024


def test_replay_and_fingerprint_cache_no_bytes():
    runs, closed = [], []

    def source():
        runs.append(True)
        try:
            yield b"head"
            yield b"tail"
        finally:
            closed.append(True)

    body = Video(source, 1)
    assert body.size is None
    assert body.fingerprint() == "9eff9643fdf694cf59b821c37310f26c"
    assert body.size == 8
    assert body.fingerprint() == body.digest
    assert runs == [True]
    for _ in range(2):
        assert b"".join(body.iter_bytes()) == b"headtail"
    assert runs == closed == [True] * 3


def test_source_failure_closes_generator_and_can_retry():
    closed = []

    def source():
        try:
            yield b"head"
            message = "source failed"
            raise OSError(message)
        finally:
            closed.append(True)

    body = Video(source, 1)
    for _ in range(2):
        with pytest.raises(OSError, match="source failed"):
            body.fingerprint()
        assert body.digest is body.size is None
    assert closed == [True, True]


@pytest.mark.parametrize("kind", ["bytes", "io"])
def test_bytes_and_seekable_io_use_the_same_video_type(kind):
    import io

    source = b"headtail" if kind == "bytes" else io.BytesIO(b"headtail")
    body = Video(source, duration=1)
    assert body.replayable
    assert body.fingerprint() == "9eff9643fdf694cf59b821c37310f26c"
    for _ in range(2):
        assert b"".join(body.iter_bytes()) == b"headtail"
    if kind == "io":
        assert not source.closed


def test_non_seekable_io_is_one_shot_and_cannot_be_consumed_by_hashing():
    import io

    class Pipe(io.BytesIO):
        def seekable(self):
            return False

    source = Pipe(b"video")
    body = Video(source, duration=1)
    assert not body.replayable
    with pytest.raises(ValueError, match="replayable"):
        body.fingerprint()
    assert source.tell() == 0
    assert b"".join(body.iter_bytes()) == b"video"
    with pytest.raises(ValueError, match="replayed"):
        next(body.iter_bytes())
    assert not source.closed


def test_seekable_io_replays_the_entire_video():
    import io

    source = io.BytesIO(b"prefixvideo")
    source.seek(6)
    body = Video(source, duration=1)
    assert b"".join(body.iter_bytes()) == b"prefixvideo"
    assert b"".join(body.iter_bytes()) == b"prefixvideo"
    assert not source.closed


def test_io_reads_are_bounded():
    import io

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 65536
            return super().read(size)

    body = Video(Bounded(bytes(65536 * 3)), duration=1)
    assert [len(chunk) for chunk in body.iter_bytes()] == [65536] * 3


@pytest.mark.parametrize("kind", ["bytes", "io"])
def test_duration_is_measured_for_bytes_and_io(kind):
    import io
    from pathlib import Path

    from simple_video_utils.metadata import video_metadata

    path = Path(__file__).parent / "assets" / "example-short.mp4"
    source = path.read_bytes() if kind == "bytes" else io.BytesIO(path.read_bytes())
    body = Video(source)
    assert body.duration_seconds() == pytest.approx(video_metadata(str(path)).duration)
    assert b"".join(body.iter_bytes()) == path.read_bytes()
    if kind == "io":
        assert not source.closed


def test_duration_probe_rejects_a_generated_source_without_consuming_it():
    opened = []

    def source():
        opened.append(True)
        yield b"video"

    body = Video(source)
    with pytest.raises(ValueError, match="Supply duration"):
        body.duration_seconds()
    assert opened == []


def test_io_duration_probe_preserves_callers_position():
    import io
    from pathlib import Path

    source = io.BytesIO((Path(__file__).parent / "assets" / "example-short.mp4").read_bytes())
    body = Video(source)
    source.seek(10)
    assert body.duration_seconds() > 0
    assert source.tell() == 10


def test_io_cannot_be_probed_or_read_by_two_consumers():
    import io

    body = Video(io.BytesIO(bytes(65537)))
    reader = body.iter_bytes()
    assert len(next(reader)) == 65536
    with pytest.raises(ValueError, match="active reader"):
        next(body.iter_bytes())
    with pytest.raises(ValueError, match="active reader"):
        body.duration_seconds()
    reader.close()
    assert len(b"".join(body.iter_bytes())) == 65537


def test_factory_is_coalesced_and_closed_on_early_stop():
    closed = []

    def source():
        try:
            for _ in range(65537):
                yield b"a"
        finally:
            closed.append(True)

    reader = Video(source).iter_bytes()
    assert len(next(reader)) == 65536
    reader.close()
    assert closed == [True]
