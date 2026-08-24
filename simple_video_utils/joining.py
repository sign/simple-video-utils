"""Join videos, preserving their encoded packets when they overlap."""

import io
from collections.abc import Sequence
from fractions import Fraction

import av

from simple_video_utils.metadata import _open_video
from simple_video_utils.slicing import _MP4_COPY_CODECS, _copy_packet, _iter_packets, _mux_packets


def _packets(container: av.container.InputContainer, time_base: Fraction) -> list[av.Packet]:
    """All video packets, copied and rescaled onto ``time_base``."""
    packets = []
    for source in _iter_packets(container, container.streams.video[0]):
        packet = _copy_packet(source)
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
    """Remux without re-encoding, or None when any boundary shares no packets."""
    containers = [_open_video(io.BytesIO(video)) for video in videos]
    try:
        stream = containers[0].streams.video[0]
        if stream.codec_context.name not in _MP4_COPY_CODECS:
            return None
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
        return _mux_packets(stream, packets, 0)
    finally:
        for container in containers:
            container.close()


def _encode_join(videos: Sequence[bytes]) -> bytes:
    inputs = [_open_video(io.BytesIO(video)) for video in videos]
    try:
        streams = [container.streams.video[0] for container in inputs]
        first = streams[0].codec_context
        if any((s.codec_context.width, s.codec_context.height) != (first.width, first.height)
               for s in streams):
            message = "videos must have the same dimensions"
            raise ValueError(message)
        rate = streams[0].average_rate or streams[0].guessed_rate or 24
        output = io.BytesIO()
        with av.open(output, mode="w", format="mp4") as destination:
            stream = destination.add_stream("h264", rate=rate, options={"crf": "18"})
            stream.width, stream.height, stream.pix_fmt = first.width, first.height, "yuv420p"
            reformatter = av.video.reformatter.VideoReformatter()
            for container, source in zip(inputs, streams, strict=True):
                for frame in container.decode(source):
                    video_frame = reformatter.reformat(frame, width=first.width,
                                                       height=first.height, format="yuv420p")
                    video_frame.pts = None
                    destination.mux(stream.encode(video_frame))
            destination.mux(stream.encode())
        return output.getvalue()
    finally:
        for container in inputs:
            container.close()


def join_videos(videos: Sequence[bytes]) -> bytes:
    """Join videos in order.

    Clips copied from the same encoded stream share packets around their
    boundaries; those are de-duplicated and the join is remuxed without
    quality loss. Anything else — a boundary with no shared packets, or a
    codec MP4 can't carry — falls back to decoding everything and encoding
    one H.264 MP4 (CRF 18), so inputs are re-encoded at most once.
    """
    if not videos:
        message = "at least one video is required"
        raise ValueError(message)
    if len(videos) == 1:
        return videos[0]
    return _copy_join(videos) or _encode_join(videos)
