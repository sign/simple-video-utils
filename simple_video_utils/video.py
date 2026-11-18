"""A video backed by bytes, binary I/O or a replayable slice."""

import hashlib
import io
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO, cast


@dataclass(eq=False)
class Video:
    """One video backed by bytes, binary I/O, or a replayable byte-generator factory.

    Caller-owned I/O stays open and has one shared cursor. Seekable I/O replays from byte zero;
    non-seekable I/O supports one read. A generator factory must reproduce the
    same bytes on every pass. Close readers when stopping consumption early.
    """

    source: bytes | BinaryIO | Callable[[], Iterator[bytes]]
    duration: float | None = None
    media_type: str = "video/mp4"
    size: int | None = None
    digest: str | None = None

    _seekable: bool = field(default=False, init=False, repr=False)
    _reading: bool = field(default=False, init=False, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.source, bytes):
            self.size = len(self.source)
        if not isinstance(self.source, bytes) and not callable(self.source) and self.source.seekable():
            self._seekable = True

    @classmethod
    def from_slice(cls, src: str, start: float, end: float, *, chapters: bool = False) -> "Video":
        """Bind a lossless MP4 slice; read it lazily without retaining its media.

        The source must remain unchanged. MP4-unsupported codecs raise rather than
        re-encode. Only MP4 headers/sample tables and the current packet are buffered.
        """
        from simple_video_utils.slicing import _Slice

        return cls(_Slice(src, start, end, chapters))

    @property
    def replayable(self) -> bool:
        return isinstance(self.source, bytes) or callable(self.source) or self._seekable

    def iter_bytes(self) -> Generator[bytes, None, None]:
        """Read bounded pieces; a factory gives concurrent readers independent decoders."""
        if isinstance(self.source, bytes):
            for start in range(0, len(self.source), 64 * 1024):
                yield self.source[start:start + 64 * 1024]
        elif callable(self.source):
            chunks = self.source()
            try:
                yield from _coalesce(chunks)
            finally:
                if close := getattr(chunks, "close", None):
                    close()
        else:
            if self._reading:
                message = "Video I/O already has an active reader"
                raise ValueError(message)
            self._reading = True
            try:
                if self._seekable:
                    self.source.seek(0)
                elif self._consumed:
                    message = "Non-seekable video cannot be replayed"
                    raise ValueError(message)
                self._consumed = True
                while chunk := self.source.read(64 * 1024):
                    yield chunk
            finally:
                self._reading = False

    def duration_seconds(self) -> float:
        if self.duration is None:
            from simple_video_utils.slicing import _Slice

            if isinstance(self.source, _Slice):
                self.duration = self.source.duration_seconds()
                self.size = self.source.size_bytes()
            else:
                self.duration = self._measure_duration()
        assert self.duration is not None
        return self.duration

    def _measure_duration(self) -> float:
        import av

        from simple_video_utils.metadata import video_metadata_from_container

        if self._reading:
            message = "Video I/O already has an active reader"
            raise ValueError(message)
        if not isinstance(self.source, bytes) and not self._seekable:
            message = "Supply duration for a generated or non-seekable video"
            raise ValueError(message)
        source = io.BytesIO(self.source) if isinstance(self.source, bytes) else cast(BinaryIO, self.source)
        position = source.tell()
        self._reading = True
        try:
            source.seek(0)
            with av.open(source, mode="r") as container:
                meta = video_metadata_from_container(container, rotation=0)
        finally:
            try:
                source.seek(position)
            finally:
                self._reading = False
        if meta.duration is not None:
            return meta.duration
        if meta.nb_frames and meta.fps:
            return meta.nb_frames / meta.fps
        message = "Could not determine video duration"
        raise ValueError(message)

    def size_bytes(self) -> int:
        """Measure length without retaining media; slices need no content hash."""
        if self.size is None:
            from simple_video_utils.slicing import _Slice

            if isinstance(self.source, _Slice):
                self.size = self.source.size_bytes()
            else:
                self.fingerprint()
        assert self.size is not None
        return self.size

    def fingerprint(self) -> str:
        if not self.replayable:
            message = "Fingerprinting requires a replayable video source"
            raise ValueError(message)
        if self.digest is None or self.size is None:
            digest, size = hashlib.md5(), 0
            for chunk in self.iter_bytes():
                digest.update(chunk)
                size += len(chunk)
            self.digest, self.size = digest.hexdigest(), size
        return self.digest


def _coalesce(chunks):
    """Keep transport chunks bounded without one scheduler hop per media packet."""
    pending = bytearray()
    for chunk in chunks:
        view = memoryview(chunk)
        while view:
            count = min(len(view), 64 * 1024 - len(pending))
            pending.extend(view[:count])
            view = view[count:]
            if len(pending) == 64 * 1024:
                yield bytes(pending)
                pending.clear()
    if pending:
        yield bytes(pending)
