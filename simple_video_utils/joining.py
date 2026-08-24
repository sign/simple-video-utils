"""Join videos, preserving their encoded packets when they overlap."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import av

_DIMENSIONS_ERROR = "videos must have the same dimensions"
_EMPTY_ERROR = "at least one video is required"


def _container_format(stream: av.VideoStream) -> str:
    return "webm" if stream.codec_context.name in {"vp8", "vp9"} else "mp4"


@dataclass(frozen=True)
class _Packet:
    data: bytes
    pts: Fraction
    dts: Fraction
    duration: Fraction
    is_keyframe: bool


def _packets(container: av.InputContainer) -> list[_Packet]:
    stream = container.streams.video[0]
    time_base = stream.time_base
    assert time_base is not None
    return [
        _Packet(
            bytes(packet), packet.pts * time_base, packet.dts * time_base,
            (packet.duration or 0) * time_base, packet.is_keyframe,
        )
        for packet in container.demux(stream)
        if packet.pts is not None and packet.dts is not None and packet.size
    ]


def _overlap(first: list[_Packet], second: list[_Packet]) -> int:
    """Number of encoded packets shared by ``first``'s tail and ``second``'s head."""
    limit = min(len(first), len(second))
    for size in range(limit, 0, -1):
        if all(a.data == b.data for a, b in zip(first[-size:], second[:size], strict=True)):
            return size
    return 0


def _copy_join(videos: Sequence[bytes]) -> bytes | None:
    containers = [av.open(io.BytesIO(video), mode="r") for video in videos]
    try:
        packets = _packets(containers[0])
        for container in containers[1:]:
            incoming = _packets(container)
            overlap = _overlap(packets, incoming)
            if not overlap:
                return None
            shift = packets[-overlap].dts - incoming[0].dts
            packets.extend(_Packet(
                packet.data, packet.pts + shift, packet.dts + shift,
                packet.duration, packet.is_keyframe,
            ) for packet in incoming[overlap:])

        output = io.BytesIO()
        stream = containers[0].streams.video[0]
        with av.open(output, mode="w", format=_container_format(stream)) as destination:
            time_base = stream.time_base
            assert time_base is not None
            out_stream = destination.add_stream_from_template(stream)
            for source in packets:
                packet = av.Packet(source.data)
                packet.pts = round(source.pts / time_base)
                packet.dts = round(source.dts / time_base)
                packet.duration = round(source.duration / time_base)
                packet.time_base = time_base
                packet.is_keyframe = source.is_keyframe
                packet.stream = out_stream
                destination.mux(packet)
        return output.getvalue()
    finally:
        for container in containers:
            container.close()


def _encode_join(videos: Sequence[bytes]) -> bytes:
    inputs = [av.open(io.BytesIO(video), mode="r") for video in videos]
    try:
        first = inputs[0].streams.video[0]
        rate = first.average_rate or first.guessed_rate or 24
        width, height = first.codec_context.width, first.codec_context.height
        output = io.BytesIO()
        with av.open(output, mode="w", format="mp4") as destination:
            stream = destination.add_stream("h264", rate=rate, options={"crf": "18"})
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            reformatter = av.video.reformatter.VideoReformatter()
            for container in inputs:
                source = container.streams.video[0]
                if (source.codec_context.width, source.codec_context.height) != (width, height):
                    raise ValueError(_DIMENSIONS_ERROR)
                for frame in container.decode(source):
                    frame = reformatter.reformat(frame, width=width, height=height, format="yuv420p")
                    frame.pts = None
                    destination.mux(stream.encode(frame))
            destination.mux(stream.encode())
        return output.getvalue()
    finally:
        for container in inputs:
            container.close()


def join_videos(videos: Sequence[bytes]) -> bytes:
    """Join videos in order.

    Slices copied from the same encoded stream share packets around their
    boundary. Those packets are de-duplicated and remuxed without quality loss;
    unrelated videos are decoded and joined into an MP4 with H.264 CRF 18.
    """
    if not videos:
        raise ValueError(_EMPTY_ERROR)
    joined = videos[0]
    for video in videos[1:]:
        joined = _copy_join((joined, video)) or _encode_join((joined, video))
    return joined
