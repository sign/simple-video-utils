"""Split an incoming video byte stream into clips without re-encoding."""

import io
import math
from collections.abc import AsyncGenerator, AsyncIterable, Iterator
from dataclasses import dataclass

import av


@dataclass(frozen=True)
class _Packet:
    data: bytes
    pts: int
    dts: int
    duration: int
    time_base: object
    keyframe: bool


def _mux(stream: av.VideoStream, packets: list[_Packet], start: int) -> bytes:
    output = io.BytesIO()
    format_name = "webm" if stream.codec_context.name in {"vp8", "vp9"} else "mp4"
    with av.open(output, mode="w", format=format_name) as container:
        destination = container.add_stream_from_template(stream)
        for source in packets:
            packet = av.Packet(source.data)
            packet.pts = source.pts - start
            packet.dts = source.dts - start
            packet.duration = source.duration
            packet.time_base = source.time_base
            packet.is_keyframe = source.keyframe
            packet.stream = destination
            container.mux(packet)
    return output.getvalue()


def _clip(packets: list[_Packet], start: int, end: int | None) -> list[_Packet]:
    keyframe = max(
        (index for index, packet in enumerate(packets)
         if packet.keyframe and packet.pts <= start),
        default=0,
    )
    selected = packets[keyframe:]
    return selected if end is None else [packet for packet in selected if packet.dts <= end]


def _split(data: bytes, duration: float) -> Iterator[bytes]:
    with av.open(io.BytesIO(data), metadata_errors="replace") as container:
        source = container.streams.video[0]
        assert source.time_base is not None
        origin = source.start_time or 0
        packets: list[_Packet] = []
        index = 0

        def boundary(number: int) -> int:
            return round(number * duration / float(source.time_base)) + origin

        start, end = boundary(0), boundary(1)
        for packet in container.demux(source):
            if packet.pts is None or packet.dts is None or not packet.size:
                continue
            packets.append(_Packet(
                bytes(packet), packet.pts, packet.dts, packet.duration or 0,
                packet.time_base, packet.is_keyframe,
            ))
            while packet.dts > end:
                selected = _clip(packets, start, end)
                if any(item.pts >= start for item in selected):
                    yield _mux(source, selected, start)
                index += 1
                start, end = boundary(index), boundary(index + 1)

        selected = _clip(packets, start, None)
        if any(item.pts >= start for item in selected):
            yield _mux(source, selected, start)


async def slice_video_stream(
    chunks: AsyncIterable[bytes],
    duration: float = 0.5,
) -> AsyncGenerator[bytes, None]:
    """Yield packet-copied clips after the input stream reaches EOF."""
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)

    data = b"".join([chunk async for chunk in chunks])
    for clip in _split(data, duration):
        yield clip
