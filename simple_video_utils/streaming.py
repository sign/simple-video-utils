"""Split an incoming video byte stream into standalone MP4 clips."""

import asyncio
import io
import math
import queue
from collections.abc import AsyncGenerator, AsyncIterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from threading import Event
from typing import Any

import av

_INPUT_QUEUE_SIZE = 8
_OUTPUT_QUEUE_SIZE = 2
_END = object()


@dataclass(frozen=True)
class _Packet:
    data: bytes
    pts: int
    dts: int
    duration: int
    time_base: Fraction
    is_keyframe: bool


class _StreamingBody(io.RawIOBase):
    """Blocking file view over an asynchronous byte stream."""

    def __init__(self, chunks: queue.Queue, stopped: Event) -> None:
        self._chunks = chunks
        self._stopped = stopped
        self._buffer = bytearray()
        self._eof = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer) -> int:  # noqa: ANN001 - RawIOBase signature
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        while not self._buffer and not self._eof:
            self._pull()
        if size < 0:
            while not self._eof:
                self._pull()
            size = len(self._buffer)
        data = self._buffer[:size]
        del self._buffer[:size]
        return bytes(data)

    def _pull(self) -> None:
        while not self._stopped.is_set():
            try:
                item = self._chunks.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        else:
            item = None
        if item is None:
            self._eof = True
        elif isinstance(item, BaseException):
            raise item
        else:
            self._buffer.extend(item)


async def _put(out: queue.Queue, item: Any, stopped: Event) -> None:
    while not stopped.is_set():
        try:
            out.put_nowait(item)
        except queue.Full:
            await asyncio.sleep(0.01)
        else:
            return


async def _feed(chunks: AsyncIterable[bytes], out: queue.Queue, stopped: Event) -> None:
    end: BaseException | None = None
    try:
        async for chunk in chunks:
            if chunk:
                await _put(out, chunk, stopped)
    except BaseException as e:  # noqa: BLE001 - forward request failures to the demuxer
        end = e
    await _put(out, end, stopped)


def _blocking_put(out: queue.Queue, item: Any, stopped: Event) -> None:
    while not stopped.is_set():
        try:
            out.put(item, timeout=0.1)
        except queue.Full:
            pass
        else:
            return


def _clip_packets(packets: list[_Packet], start: int, end: int | None) -> list[_Packet]:
    keyframe = max(
        (i for i, packet in enumerate(packets) if packet.is_keyframe and packet.pts <= start),
        default=0,
    )
    selected = packets[keyframe:]
    return selected if end is None else [packet for packet in selected if packet.dts <= end]


def _mux(stream: av.VideoStream, packets: list[_Packet], start: int) -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        out_stream = container.add_stream_from_template(stream)
        for source in packets:
            packet = av.Packet(source.data)
            packet.pts = source.pts - start
            packet.dts = source.dts - start
            packet.duration = source.duration
            packet.time_base = source.time_base
            packet.is_keyframe = source.is_keyframe
            packet.stream = out_stream
            container.mux(packet)
    return output.getvalue()


def _split(body: _StreamingBody, duration: float) -> Iterator[bytes]:
    with av.open(body, metadata_errors="replace", buffer_size=4096) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        origin = stream.start_time or 0
        packets: list[_Packet] = []
        clip_index = 0

        def boundary(index: int) -> int:
            return round(index * duration / time_base) + origin

        start, end = boundary(0), boundary(1)
        for packet in container.demux(stream):
            if packet.pts is None or packet.dts is None or not packet.size:
                continue
            packets.append(_Packet(
                bytes(packet), packet.pts, packet.dts, packet.duration or 0,
                packet.time_base, packet.is_keyframe,
            ))
            while packet.dts > end:
                selected = _clip_packets(packets, start, end)
                if any(item.pts >= start for item in selected):
                    yield _mux(stream, selected, start)
                clip_index += 1
                start, end = boundary(clip_index), boundary(clip_index + 1)
                keyframe = max(
                    (i for i, item in enumerate(packets) if item.is_keyframe and item.pts <= start),
                    default=0,
                )
                packets = packets[keyframe:]

        selected = _clip_packets(packets, start, None)
        if any(item.pts >= start for item in selected):
            yield _mux(stream, selected, start)


def _worker(body: _StreamingBody, out: queue.Queue, stopped: Event, duration: float) -> None:
    try:
        for clip in _split(body, duration):
            _blocking_put(out, clip, stopped)
            if stopped.is_set():
                break
    except BaseException as e:  # noqa: BLE001 - forward demux failures to the response stream
        _blocking_put(out, e, stopped)
    finally:
        _blocking_put(out, _END, stopped)


async def _get(source: queue.Queue) -> Any:
    while True:
        try:
            return source.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)


async def slice_video_stream(
    chunks: AsyncIterable[bytes],
    duration: float = 0.5,
) -> AsyncGenerator[bytes, None]:
    """Yield independently playable MP4 clips while video bytes are arriving.

    The compressed packets are copied. Each clip retains the keyframe before
    its requested start and rebases timestamps so an MP4 edit list hides that
    lead-in, matching ``slice_video`` without seeking or re-encoding.
    """
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)

    stopped = Event()
    input_queue: queue.Queue = queue.Queue(maxsize=_INPUT_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=_OUTPUT_QUEUE_SIZE)
    body = _StreamingBody(input_queue, stopped)
    feeder = asyncio.create_task(_feed(chunks, input_queue, stopped))
    worker = asyncio.create_task(asyncio.to_thread(_worker, body, output_queue, stopped, duration))
    try:
        while (item := await _get(output_queue)) is not _END:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()
        feeder.cancel()
        with suppress(asyncio.CancelledError):
            await feeder
        await worker
