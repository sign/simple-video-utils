"""Join videos, preserving their encoded packets when they overlap."""

import io
import json
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np

from simple_video_utils.metadata import _open_video, video_metadata_from_container
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
    """One clip: frames [start_frame, end_frame) as uint8 RGB arrays or PyAV video frames."""

    start_frame: int
    end_frame: int
    width: int
    height: int
    frames: Iterable[np.ndarray | av.VideoFrame]
    payload: dict | None = None


@dataclass
class VideoCrop:
    """A source-frame range [start_frame, end_frame) and a crop in stored pixel coordinates."""

    start_frame: int
    end_frame: int
    x: int
    y: int
    width: int
    height: int
    payload: dict | None = None


def _crop_frames(frames: Iterable[av.VideoFrame], crop: VideoCrop) -> Iterable[av.VideoFrame]:
    graph = None
    for frame in frames:
        if graph is None:
            graph = av.filter.Graph()
            # Subsampled chroma cannot represent odd crop origins exactly.
            filters = []
            if any(value % 2 for value in (crop.x, crop.y, crop.width, crop.height)):
                filters.append(graph.add("format", "pix_fmts=yuv444p"))
            graph.link_nodes(
                graph.add_buffer(template=frame),
                *filters,
                graph.add("crop", f"{crop.width}:{crop.height}:{crop.x}:{crop.y}:exact=1"),
                graph.add("buffersink"),
            )
            graph.configure()
        graph.push(frame)
        yield graph.pull()


def write_video_crops(source: str | Path, crops: Sequence[VideoCrop], output: str | Path) -> None:
    """Crop source ranges into gapped H.264 Matroska tracks without converting frames to RGB.

    Crops use stored coordinates (before display rotation), have exclusive end frames,
    and must be disjoint and within the source. Odd sizes are padded right and bottom.
    Decoding runs once in presentation order, including intervening gaps, and only crop
    frames are filtered and re-encoded. Memory is bounded independently of clip length.
    The output uses the source's nominal frame rate, as does merge_video_tracks.
    """
    with _open_video(str(source)) as video:
        stream = video.streams.video[0]
        metadata = video_metadata_from_container(video)
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(metadata.fps).limit_denominator(1000)
        if metadata.nb_frames is None:
            message = "source frame count is unavailable"
            raise ValueError(message)
        _validate_track_ranges(((crop.start_frame, crop.end_frame) for crop in crops), metadata.nb_frames)
        for crop in crops:
            if (
                crop.x < 0 or crop.y < 0 or crop.width <= 0 or crop.height <= 0
                or crop.x + crop.width > stream.width or crop.y + crop.height > stream.height
            ):
                message = "crop must have positive dimensions and lie within the stored source frame"
                raise ValueError(message)
        decoded = video.decode(stream)
        tracks = []
        previous_end = 0
        for crop in sorted(crops, key=lambda crop: crop.start_frame):
            frames = islice(decoded, crop.start_frame - previous_end, crop.end_frame - previous_end)
            tracks.append(VideoTrack(
                crop.start_frame, crop.end_frame, crop.width, crop.height, _crop_frames(frames, crop), crop.payload
            ))
            previous_end = crop.end_frame
        write_video_tracks(tracks, output, fps=rate, source_frames=metadata.nb_frames)


def _validate_track_ranges(ranges: Iterable[tuple[int, int]], total: int) -> None:
    previous_end = 0
    for start, end in sorted(ranges):
        if start < previous_end or end <= start or end > total:
            message = "track ranges must be disjoint and within source_frames"
            raise ValueError(message)
        previous_end = end


def _track_packets(stream: av.VideoStream, track: VideoTrack, rate: Fraction) -> Iterable[av.Packet]:
    count = 0
    padding = None
    for item in track.frames:
        if isinstance(item, av.VideoFrame):
            frame = item
            valid = (frame.width, frame.height) == (track.width, track.height)
        else:
            valid = item.shape == (track.height, track.width, 3) and item.dtype == np.uint8
            if valid:
                padded = np.pad(item, ((0, stream.height - track.height), (0, stream.width - track.width), (0, 0)))
                frame = av.VideoFrame.from_ndarray(padded, format="rgb24")
        if not valid:
            message = f"track {track.start_frame}-{track.end_frame} needs frames matching its dimensions"
            raise ValueError(message)
        frame.time_base = Fraction(1) / rate
        if isinstance(item, av.VideoFrame) and (track.width % 2 or track.height % 2):
            # pad rounds subsampled input sizes down; 4:4:4 retains the last odd row/column.
            frame = frame.reformat(format="yuv444p")
            if padding is None:
                padding = av.filter.Graph()
                padding.link_nodes(
                    padding.add_buffer(template=frame),
                    padding.add("pad", f"{stream.width}:{stream.height}:0:0:black"),
                    padding.add("buffersink"),
                )
                padding.configure()
            padding.push(frame)
            frame = padding.pull()
        # each track sits at its source time, so a player shows the clip where it happened
        frame.pts = track.start_frame + count
        frame.pict_type = 0
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
    to even for H.264. Ranges must be disjoint and within source_frames; tracks may be in any order.
    """
    if not tracks:
        message = "at least one track is required"
        raise ValueError(message)
    rate = Fraction(fps) if not isinstance(fps, float) else Fraction(fps).limit_denominator(1000)
    if rate <= 0 or source_frames < 0:
        message = "source_fps must be positive and source_frames nonnegative"
        raise ValueError(message)
    _validate_track_ranges(((track.start_frame, track.end_frame) for track in tracks), source_frames)
    if any(track.width <= 0 or track.height <= 0 for track in tracks):
        message = "track dimensions must be positive"
        raise ValueError(message)
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
                stream = container.add_stream("libx264", rate=rate, options={"crf": "18"})
                stream.width, stream.height = track.width + track.width % 2, track.height + track.height % 2
                stream.pix_fmt = "yuv420p"
                stream.metadata["title"] = f"{track.start_frame}-{track.end_frame}"
                stream.metadata["payload"] = json.dumps(track.payload or {})
                streams.append(stream)
            for stream, track in zip(streams, tracks, strict=True):
                for packet in _track_packets(stream, track, rate):
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
    _validate_track_ranges(((start, end) for start, end, _ in clips), total)
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
                    # A container-wide seek uses the first track, which may start after this clip.
                    video.seek(clip.start_time or 0, stream=clip)
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
