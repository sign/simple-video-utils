"""Join videos, preserving their encoded packets when they overlap."""

from __future__ import annotations

import io
from collections.abc import Sequence

import av


def _copy_packet(packet: av.Packet) -> av.Packet:
    copy = av.Packet(bytes(packet))
    copy.pts = packet.pts
    copy.dts = packet.dts
    copy.duration = packet.duration
    copy.time_base = packet.time_base
    copy.is_keyframe = packet.is_keyframe
    return copy


def _packets(container: av.InputContainer, time_base) -> list[av.Packet]:  # noqa: ANN001 - PyAV's time base type
    stream = container.streams.video[0]
    packets = []
    for packet in container.demux(stream):
        if packet.pts is None or packet.dts is None or not packet.size:
            continue
        packet = _copy_packet(packet)
        assert packet.time_base is not None
        if packet.time_base != time_base:
            packet.pts = round(packet.pts * packet.time_base / time_base)
            packet.dts = round(packet.dts * packet.time_base / time_base)
            packet.duration = round((packet.duration or 0) * packet.time_base / time_base)
        packet.time_base = time_base
        packets.append(packet)
    return packets


def _overlap(first: list[av.Packet], second: list[av.Packet]) -> int:
    """Number of encoded packets shared by ``first``'s tail and ``second``'s head."""
    limit = min(len(first), len(second))
    for size in range(limit, 0, -1):
        if all(bytes(a) == bytes(b) for a, b in zip(first[-size:], second[:size], strict=True)):
            return size
    return 0


def _copy_join(videos: Sequence[bytes]) -> bytes | None:
    containers = [av.open(io.BytesIO(video), mode="r") for video in videos]
    try:
        stream = containers[0].streams.video[0]
        assert stream.time_base is not None
        packets = _packets(containers[0], stream.time_base)
        for container in containers[1:]:
            incoming = _packets(container, stream.time_base)
            overlap = _overlap(packets, incoming)
            if not overlap:
                return None
            shift = packets[-overlap].dts - incoming[0].dts
            for packet in incoming[overlap:]:
                packet.pts += shift
                packet.dts += shift
            packets.extend(incoming[overlap:])

        output = io.BytesIO()
        format_name = "webm" if stream.codec_context.name in {"vp8", "vp9"} else "mp4"
        with av.open(output, mode="w", format=format_name) as destination:
            out_stream = destination.add_stream_from_template(stream)
            for source in packets:
                packet = _copy_packet(source)
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
                    message = "videos must have the same dimensions"
                    raise ValueError(message)
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
        message = "at least one video is required"
        raise ValueError(message)
    joined = videos[0]
    for video in videos[1:]:
        joined = _copy_join((joined, video)) or _encode_join((joined, video))
    return joined
