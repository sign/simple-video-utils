"""Join videos, preserving their encoded packets when they overlap."""

import io
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from fractions import Fraction

import av

from simple_video_utils.metadata import _open_video
from simple_video_utils.slicing import _MP4_COPY_CODECS, _faststart, _iter_packets, _mux_packets


def _packets(container: av.container.InputContainer, time_base: Fraction) -> list[av.Packet]:
    """Decodable video packets, rescaled onto ``time_base``."""
    packets = list(_iter_packets(container, container.streams.video[0]))
    for packet in packets:
        if packet.time_base != time_base:
            packet.pts = round(packet.pts * packet.time_base / time_base)
            packet.dts = round(packet.dts * packet.time_base / time_base)
            packet.duration = round((packet.duration or 0) * packet.time_base / time_base)
            packet.time_base = time_base
    return packets


def _overlap(first: list[av.Packet], second: list[av.Packet]) -> int:
    """Number of encoded packets shared by ``first``'s tail and ``second``'s head."""
    limit = min(len(first), len(second))
    tail = [bytes(packet) for packet in first[len(first) - limit:]]
    head = [bytes(packet) for packet in second[:limit]]
    return next((size for size in range(limit, 0, -1) if tail[limit - size:] == head[:size]), 0)


def _copy_join(videos: Sequence[bytes]) -> bytes | None:
    """Remux without re-encoding; None when the codec can't be copied into MP4
    or a clip doesn't continue where the previous ones left off."""
    with ExitStack() as stack:
        containers = [stack.enter_context(_open_video(io.BytesIO(video))) for video in videos]
        stream = containers[0].streams.video[0]
        if stream.codec_context.name not in _MP4_COPY_CODECS:
            return None
        packets = _packets(containers[0], stream.time_base)
        for container in containers[1:]:
            incoming = _packets(container, stream.time_base)
            overlap = _overlap(packets, incoming)
            # The overlap must cover the incoming clip's whole hidden lead-in
            # (negative pts): a shorter match means a gap between the clips,
            # and the lead-in bridging it would surface as visible content.
            lead_in = sum(1 for packet in incoming if packet.pts < 0)
            if not overlap or overlap < lead_in:
                return None
            shift = packets[-overlap].dts - incoming[0].dts
            for packet in incoming[overlap:]:
                packet.pts += shift
                packet.dts += shift
            packets.extend(incoming[overlap:])
        return _mux_packets(stream, packets, 0)


def _encode_join(videos: Sequence[bytes]) -> bytes:
    with ExitStack() as stack:
        inputs = [stack.enter_context(_open_video(io.BytesIO(video))) for video in videos]
        streams = [container.streams.video[0] for container in inputs]
        first = streams[0]
        if any((s.width, s.height) != (first.width, first.height) for s in streams):
            message = "videos must have the same dimensions"
            raise ValueError(message)
        rate = first.average_rate or first.guessed_rate or 30
        output = io.BytesIO()
        with av.open(output, mode="w", format="mp4") as destination:
            stream = destination.add_stream("h264", rate=rate, options={"crf": "18"})
            stream.width, stream.height, stream.pix_fmt = first.width, first.height, "yuv420p"
            reformatter = av.video.reformatter.VideoReformatter()
            index = 0
            for container in inputs:
                for frame in container.decode(video=0):
                    if frame.pts is not None and frame.pts < 0:
                        continue  # edit-list-hidden keyframe lead-in of a copied clip
                    video_frame = reformatter.reformat(frame, format="yuv420p")
                    # explicit uniform cadence: pts=None would tick in the
                    # source time_base and collapse every frame onto t=0
                    video_frame.pts = index
                    video_frame.time_base = Fraction(1) / rate
                    video_frame.pict_type = 0  # let the encoder choose frame types
                    index += 1
                    destination.mux(stream.encode(video_frame))
            destination.mux(stream.encode())
        return _faststart(output.getvalue())


def join_videos(videos: Iterable[bytes]) -> bytes:
    """Join videos in order.

    Clips copied from the same encoded stream share packets around their
    boundaries; those are de-duplicated and the join is remuxed without
    quality loss. Overlaps merge on the source timeline — a clip fully
    contained in the previous ones adds nothing. Anything else — a gap
    between clips, no shared packets, or a codec MP4 can't carry — falls
    back to decoding everything and encoding one H.264 MP4 (CRF 18), so
    joining adds at most one encode generation.
    """
    videos = list(videos)
    if not videos:
        message = "at least one video is required"
        raise ValueError(message)
    if len(videos) == 1:
        return videos[0]
    return _copy_join(videos) or _encode_join(videos)
