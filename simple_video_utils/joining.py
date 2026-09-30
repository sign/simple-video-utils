"""Join videos, preserving their encoded packets when they overlap."""

import io
import json
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np

from simple_video_utils.metadata import _open_video
from simple_video_utils.slicing import _MP4_COPY_CODECS, _iter_packets, _mux_packets


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
        return output.getvalue()


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


GAPPED_TRACKS_TAG = "gapped_tracks"
GAPPED_TRACKS_VERSION = "1"


@dataclass
class VideoTrack:
    """One clip of a source video: frames [start_frame, end_frame) as uint8 RGB of height x width."""

    start_frame: int
    end_frame: int
    width: int
    height: int
    frames: Iterable[np.ndarray]
    payload: dict | None = None


def _track_packets(stream: av.VideoStream, track: VideoTrack) -> Iterable[av.Packet]:
    count = 0
    for rgb in track.frames:
        if rgb.shape != (track.height, track.width, 3) or rgb.dtype != np.uint8:
            message = f"track {track.start_frame}-{track.end_frame} frames must be uint8 RGB of its height x width"
            raise ValueError(message)
        padded = np.pad(rgb, ((0, stream.height - track.height), (0, stream.width - track.width), (0, 0)))
        frame = av.VideoFrame.from_ndarray(padded, format="rgb24")
        # each track sits at its source time, so a player shows the clip where it happened
        frame.pts = track.start_frame + count
        yield from stream.encode(frame)
        count += 1
    if count != track.end_frame - track.start_frame:
        expected = track.end_frame - track.start_frame
        message = f"track {track.start_frame}-{track.end_frame} declares {expected} frames, got {count}"
        raise ValueError(message)
    yield from stream.encode()


def write_video_tracks(
    tracks: Sequence[VideoTrack], output: str | Path, *, fps: float | Fraction, source_frames: int
) -> None:
    """Write each clip as its own H.264 track of a Matroska file that merge_video_tracks reads back.

    Tracks are titled "start-end" (end exclusive) and carry their payload as JSON. The file carries
    source_fps, source_frames and a gapped_tracks version tag. Odd sizes are padded right and bottom
    to even for H.264.
    """
    if not tracks:
        message = "at least one track is required"
        raise ValueError(message)
    rate = Fraction(fps).limit_denominator(1000)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        artifact = Path(directory) / output.name
        with av.open(str(artifact), mode="w", format="matroska") as container:
            container.metadata[GAPPED_TRACKS_TAG] = GAPPED_TRACKS_VERSION
            container.metadata["source_fps"] = str(rate)
            container.metadata["source_frames"] = str(source_frames)

            # matroska writes its header on the first packet, so every track is registered before any is encoded
            streams = []
            for track in tracks:
                stream = container.add_stream("libx264", rate=rate)
                stream.width, stream.height = track.width + track.width % 2, track.height + track.height % 2
                stream.pix_fmt = "yuv420p"
                stream.metadata["title"] = f"{track.start_frame}-{track.end_frame}"
                stream.metadata["payload"] = json.dumps(track.payload or {})
                streams.append(stream)
            for stream, track in zip(streams, tracks, strict=True):
                for packet in _track_packets(stream, track):
                    container.mux(packet)
        artifact.replace(output)


def _gapped_tracks(video: av.container.InputContainer) -> tuple[Fraction, int, list[tuple[int, int, av.VideoStream]]]:
    metadata = {key.lower(): value for key, value in video.metadata.items()}
    if metadata.get(GAPPED_TRACKS_TAG) != GAPPED_TRACKS_VERSION:
        message = f"video is not written by write_video_tracks (no {GAPPED_TRACKS_TAG}={GAPPED_TRACKS_VERSION} tag)"
        raise ValueError(message)
    if not video.streams.video or "source_fps" not in metadata or "source_frames" not in metadata:
        message = "video needs tracks, source_fps and source_frames metadata"
        raise ValueError(message)
    rate = Fraction(metadata["source_fps"])
    total = int(metadata["source_frames"])
    if rate <= 0 or total < 0:
        message = "source_fps must be positive and source_frames nonnegative"
        raise ValueError(message)

    clips = []
    for stream in video.streams.video:
        title = stream.metadata.get("title", "")
        try:
            start, end = map(int, title.split("-"))
        except ValueError as exc:
            message = f"invalid track title: {title!r}"
            raise ValueError(message) from exc
        clips.append((start, end, stream))
    clips.sort(key=lambda clip: clip[0])
    if any(
        start < 0 or end <= start or end > total or (i and start < clips[i - 1][1])
        for i, (start, end, _) in enumerate(clips)
    ):
        message = "track ranges must be disjoint and within source_frames"
        raise ValueError(message)
    return rate, total, clips


def merge_video_tracks(source: str | Path, output: str | Path, *, gap_frame: np.ndarray | None = None) -> None:
    """Merge frame-range-named video tracks into an MP4 on the source timeline.

    The source must carry source_fps and source_frames metadata. Tracks have
    titles like "191-212" (end exclusive). Clips are centered on a canvas
    sized to the largest track (or to gap_frame, if supplied). gap_frame
    must be large enough for every track and be uint8 RGB. Output metadata
    contains the ordered clip payloads.
    """
    with av.open(str(source)) as video:
        rate, total, clips = _gapped_tracks(video)

        width = max(stream.width for _, _, stream in clips)
        height = max(stream.height for _, _, stream in clips)
        if gap_frame is None:
            gap_frame = np.zeros((height, width, 3), dtype=np.uint8)
        elif (
            gap_frame.ndim != 3
            or gap_frame.shape[2] != 3
            or gap_frame.dtype != np.uint8
            or gap_frame.shape[0] < height
            or gap_frame.shape[1] < width
        ):
            message = "gap_frame must be uint8 RGB and at least as large as every track"
            raise ValueError(message)
        height, width = gap_frame.shape[:2]
        if width % 2 or height % 2:
            message = "output dimensions must be even for H.264"
            raise ValueError(message)

        payloads = [
            {"start_frame": start, "end_frame": end, "payload": json.loads(stream.metadata.get("PAYLOAD", "{}"))}
            for start, end, stream in clips
        ]
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as directory:
            artifact = Path(directory) / output.name
            with av.open(str(artifact), mode="w", format="mp4") as destination:
                destination.metadata["comment"] = json.dumps(payloads)
                stream = destination.add_stream("libx264", rate=rate, options={"crf": "18"})
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                index = 0

                def encode(rgb: np.ndarray) -> None:
                    nonlocal index
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts = index
                    frame.time_base = Fraction(1) / rate
                    destination.mux(stream.encode(frame))
                    index += 1

                for start, end, clip in clips:
                    while index < start:
                        encode(gap_frame)
                    video.seek(0)
                    count = 0
                    for frame in video.decode(clip):
                        rgb = frame.to_ndarray(format="rgb24")
                        canvas = np.zeros_like(gap_frame)
                        left, top = (width - clip.width) // 2, (height - clip.height) // 2
                        canvas[top : top + clip.height, left : left + clip.width] = rgb
                        encode(canvas)
                        count += 1
                    if count != end - start:
                        message = f"track {start}-{end} declares {end - start} frames, decoded {count}"
                        raise ValueError(message)
                while index < total:
                    encode(gap_frame)
                destination.mux(stream.encode())
            artifact.replace(output)
