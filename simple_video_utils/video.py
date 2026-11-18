"""Replayable video bodies with bounded async reads and content fingerprints."""

import asyncio
import hashlib
import io
import threading
from _thread import LockType
from collections.abc import AsyncGenerator, Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO, cast

import anyio


@dataclass
class Video:
    """One video backed by bytes, binary I/O, or a replayable byte-generator factory.

    Caller-owned I/O stays open. Seekable I/O replays from its initial position;
    non-seekable I/O supports one read. A generator factory must reproduce the
    same bytes on every pass. Close readers when stopping consumption early.
    """

    source: bytes | BinaryIO | Callable[[], Iterator[bytes]]
    duration: float | None = None
    media_type: str = "video/mp4"
    size: int | None = None
    digest: str | None = None

    _start: int | None = field(default=None, init=False, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)
    _duration_from_source: bool = field(default=False, init=False, repr=False)
    _io_lock: LockType = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.source, bytes):
            self.size = len(self.source)
        if not isinstance(self.source, bytes) and not callable(self.source) and self.source.seekable():
            self._start = self.source.tell()

    @classmethod
    def from_slice(cls, src: str, start: float, end: float, *, chapters: bool = False) -> "Video":
        """Bind a lazy slice whose duration comes from its encoded frame timestamps."""
        from simple_video_utils.slicing import slice_video_chunks

        def produce():
            body.duration = yield from slice_video_chunks(src, start, end, chapters=chapters)

        body = cls(produce)
        body._duration_from_source = True
        return body

    @property
    def replayable(self) -> bool:
        return isinstance(self.source, bytes) or callable(self.source) or self._start is not None

    def iter_bytes(self) -> Generator[bytes, None, None]:
        """Read bounded pieces; a factory gives concurrent readers independent decoders."""
        if isinstance(self.source, bytes):
            for start in range(0, len(self.source), 64 * 1024):
                yield self.source[start:start + 64 * 1024]
        elif callable(self.source):
            yield from self.source()
        else:
            if not self._io_lock.acquire(blocking=False):
                message = "Video I/O is already being consumed; use independent sources for concurrent readers"
                raise RuntimeError(message)
            try:
                if self._start is not None:
                    self.source.seek(self._start)
                elif self._consumed:
                    message = "Non-seekable video cannot be replayed"
                    raise ValueError(message)
                self._consumed = True
                while chunk := self.source.read(64 * 1024):
                    yield chunk
            finally:
                self._io_lock.release()

    async def duration_seconds(self) -> float:
        if self.duration is None:
            if self._duration_from_source:
                await self.fingerprint()
            else:
                self.duration = await _run_sync(self._measure_duration)
        assert self.duration is not None
        return self.duration

    def _measure_duration(self) -> float:
        import av

        from simple_video_utils.metadata import video_metadata_from_container

        if not isinstance(self.source, bytes) and self._start is None:
            message = "Supply duration for a generated or non-seekable video"
            raise ValueError(message)
        shared_io = not isinstance(self.source, bytes)
        if shared_io and not self._io_lock.acquire(blocking=False):
            message = "Video I/O is already being consumed"
            raise RuntimeError(message)
        try:
            source = io.BytesIO(self.source) if isinstance(self.source, bytes) else cast(BinaryIO, self.source)
            position = source.tell()
            source.seek(0 if self._start is None else self._start)
            try:
                with av.open(source, mode="r") as container:
                    meta = video_metadata_from_container(container)
            finally:
                source.seek(position)
        finally:
            if shared_io:
                self._io_lock.release()
        if meta.duration is not None:
            return meta.duration
        if meta.nb_frames and meta.fps:
            return meta.nb_frames / meta.fps
        message = "Could not determine video duration"
        raise ValueError(message)

    async def chunks(self) -> AsyncGenerator[bytes, None]:
        iterator = self.iter_bytes()
        end = object()
        try:
            while (chunk := await _run_sync(next, iterator, end)) is not end:
                yield cast(bytes, chunk)
        finally:
            with anyio.CancelScope(shield=True):
                await _run_sync(iterator.close)

    async def fingerprint(self) -> str:
        if not self.replayable:
            message = "Fingerprinting requires a replayable video source"
            raise ValueError(message)
        if self.digest is None:
            digest, size = hashlib.md5(), 0
            async for chunk in self.chunks():
                digest.update(chunk)
                size += len(chunk)
            self.digest, self.size = digest.hexdigest(), size
        return self.digest


async def _run_sync(function, *args):
    # Both ASGI scope cancellation and TaskGroup cancellation must join decoder work.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    finally:
        with anyio.CancelScope(shield=True):
            await task
