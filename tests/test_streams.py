import asyncio
import threading
import tracemalloc

import pytest

from simple_video_utils import Video


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_hashing_half_a_gigabyte_does_not_retain_the_video():
    def source():
        for _ in range(128):
            yield bytes(4 * 1024 * 1024)

    body = Video(source, 600)
    tracemalloc.start()
    try:
        await body.fingerprint()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert body.size == 512 * 1024 * 1024
    assert peak < 20 * 1024 * 1024


@pytest.mark.anyio
@pytest.mark.parametrize("raw_task_cancel", [False, True])
async def test_cancellation_joins_generator_before_closing_it(raw_task_cancel):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def source():
        try:
            entered.set()
            assert release.wait(5)
            yield b"fragment"
        finally:
            closed.set()

    body = Video(source, 600)
    chunks = body.chunks()

    async def consume():
        import anyio
        with anyio.CancelScope() as scope:
            scope_ready.set_result(scope)
            try:
                await anext(chunks)
                await anyio.sleep_forever()
            finally:
                await chunks.aclose()

    scope_ready = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(consume())
    scope = await scope_ready
    await asyncio.to_thread(entered.wait, 5)
    if raw_task_cancel:
        task.cancel()
    else:
        scope.cancel()
    release.set()
    if raw_task_cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    else:
        await asyncio.wait_for(task, 5)
    assert closed.is_set()


@pytest.mark.anyio
async def test_replay_and_fingerprint_cache_no_bytes():
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
    assert await body.fingerprint() == "9eff9643fdf694cf59b821c37310f26c"
    assert body.size == 8
    assert await body.fingerprint() == body.digest
    assert runs == [True]
    for _ in range(2):
        assert b"".join([part async for part in body.chunks()]) == b"headtail"
    assert runs == closed == [True] * 3


@pytest.mark.anyio
async def test_source_failure_closes_generator_and_can_retry():
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
            await body.fingerprint()
        assert body.digest is body.size is None
    assert closed == [True, True]


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["bytes", "io"])
async def test_bytes_and_seekable_io_use_the_same_video_type(kind):
    import io

    source = b"headtail" if kind == "bytes" else io.BytesIO(b"headtail")
    body = Video(source, duration=1)
    assert body.replayable
    assert await body.fingerprint() == "9eff9643fdf694cf59b821c37310f26c"
    for _ in range(2):
        assert b"".join([part async for part in body.chunks()]) == b"headtail"
    if kind == "io":
        assert not source.closed


@pytest.mark.anyio
async def test_non_seekable_io_is_one_shot_and_cannot_be_consumed_by_hashing():
    import io

    class Pipe(io.BytesIO):
        def seekable(self):
            return False

    source = Pipe(b"video")
    body = Video(source, duration=1)
    assert not body.replayable
    with pytest.raises(ValueError, match="replayable"):
        await body.fingerprint()
    assert source.tell() == 0
    assert b"".join([part async for part in body.chunks()]) == b"video"
    with pytest.raises(ValueError, match="replayed"):
        await anext(body.chunks())
    assert not source.closed


def test_seekable_io_rejects_concurrent_readers_and_keeps_initial_position():
    import io

    source = io.BytesIO(b"prefixvideo")
    source.seek(6)
    body = Video(source, duration=1)
    first, second = body.iter_bytes(), body.iter_bytes()
    assert next(first) == b"video"
    with pytest.raises(RuntimeError, match="already being consumed"):
        next(second)
    first.close()
    assert b"".join(body.iter_bytes()) == b"video"
    assert not source.closed


def test_io_reads_are_bounded():
    import io

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 65536
            return super().read(size)

    body = Video(Bounded(bytes(65536 * 3)), duration=1)
    assert [len(chunk) for chunk in body.iter_bytes()] == [65536] * 3


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["bytes", "io"])
async def test_duration_is_measured_for_bytes_and_io(kind):
    import io
    from pathlib import Path

    from simple_video_utils.metadata import video_metadata

    path = Path(__file__).parent / "assets" / "example-short.mp4"
    source = path.read_bytes() if kind == "bytes" else io.BytesIO(path.read_bytes())
    body = Video(source)
    assert await body.duration_seconds() == pytest.approx(video_metadata(str(path)).duration)
    assert b"".join(body.iter_bytes()) == path.read_bytes()
    if kind == "io":
        assert not source.closed


@pytest.mark.anyio
async def test_duration_probe_rejects_a_generated_source_without_consuming_it():
    opened = []

    def source():
        opened.append(True)
        yield b"video"

    body = Video(source)
    with pytest.raises(ValueError, match="Supply duration"):
        await body.duration_seconds()
    assert opened == []
