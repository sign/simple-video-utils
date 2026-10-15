"""
Chapter files: clips of a video, each cut square around a box, kept as the chapters of one Matroska video track.

Each clip sits at its source time, with gaps between clips, at its own frame size, and its chapter carries a JSON
``payload`` tag saying where the clip came from and how it was cut. ``write_chapters`` writes one, ``list_chapters``
lists its chapters, and the frame readers in ``simple_video_utils.frames`` read it frame by frame on the source
timeline.
"""

import json
import math
import operator
from collections.abc import Callable, Generator, Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import av

from simple_video_utils.metadata import open_container, video_metadata

# a crop's edges and a square's side and bars lie on whole H.264 macroblocks: the source's macroblocks then land whole
# in the chapter, and FFmpeg's decoder can keep the previous chapter's size when two sizes cover the same macroblocks
# and differ only in their crop, as 470 and 472 do
MACROBLOCK = 16
# a square larger than this side is scaled down to it
MAX_SIDE = 768
# x264's own default quality
CRF = 23


class Chapter(NamedTuple):
    """A chapter's [start, end) in seconds and the JSON object in its ``payload`` tag, None without one."""

    start: Fraction
    end: Fraction
    payload: dict | None


class Square(NamedTuple):
    """How a box is cut square: the crop around it, left, top, right and bottom in source pixels; the black bars
    squaring the crop, in the same order; the square's side; and the side it is scaled to."""

    crop: list[int]
    pad: list[int]
    side: int
    size: int


class ChapterClip(NamedTuple):
    """Source frames [start_frame, end_frame) cut square around ``box``, left, top, right and bottom in source pixels,
    with more JSON for the chapter's payload tag."""

    start_frame: int
    end_frame: int
    box: Sequence[float]
    payload: dict | None = None


def list_chapters(src: str | Path | av.container.InputContainer) -> list[Chapter]:
    """Every chapter of ``src`` by start time."""
    with open_container(str(src) if isinstance(src, Path) else src) as container:
        found = []
        for chapter in container.chapters():
            tags = {key.lower(): value for key, value in chapter["metadata"].items()}
            time_base = chapter["time_base"]
            payload = json.loads(tags["payload"]) if "payload" in tags else None
            found.append(Chapter(chapter["start"] * time_base, chapter["end"] * time_base, payload))
    return sorted(found, key=operator.attrgetter("start"))


def _centre_bar(spare: int) -> int:
    """The bar before a crop with ``spare`` pixels to share between its two bars: whole macroblocks, as near half as
    they come."""
    return min(math.floor(spare / 2 / MACROBLOCK + 0.5) * MACROBLOCK, spare)


def square(box: Sequence[float], frame_size: tuple[int, int], max_side: int = MAX_SIDE) -> Square:
    """
    The crop holding ``box`` whole, each edge moved out to the frame's next macroblock, so the chapter's macroblocks
    hold the source's own; the black bars squaring it, those before it whole macroblocks too, off centre by up to
    half of one; the square's side, whole macroblocks up to ``max_side``; and the side it is scaled to.
    """
    width, height = frame_size
    # an edge past the frame's last whole macroblock stops on its last even pixel, as yuv420p needs
    right_limit, bottom_limit = width - width % 2, height - height % 2
    left = min(math.floor(box[0] / MACROBLOCK) * MACROBLOCK, right_limit)
    top = min(math.floor(box[1] / MACROBLOCK) * MACROBLOCK, bottom_limit)
    right = min(math.ceil(box[2] / MACROBLOCK) * MACROBLOCK, right_limit)
    bottom = min(math.ceil(box[3] / MACROBLOCK) * MACROBLOCK, bottom_limit)
    if right <= left or bottom <= top:
        msg = f"empty crop for box {list(box)}"
        raise ValueError(msg)
    longer = max(right - left, bottom - top)
    side = math.ceil(longer / MACROBLOCK) * MACROBLOCK if longer <= max_side else longer
    x, y = _centre_bar(side - (right - left)), _centre_bar(side - (bottom - top))
    pad = [x, y, side - (right - left) - x, side - (bottom - top) - y]
    return Square([left, top, right, bottom], pad, side, min(side, max_side))


def _milliseconds(frame: int, rate: Fraction) -> int:
    """``frame``'s time as Matroska stores a block's, rounded half up to the millisecond, so a seek to a chapter's
    start lands on its first frame."""
    return math.floor(frame * 1000 / rate + Fraction(1, 2))


def _graph(stream: av.VideoStream, payload: dict) -> av.filter.Graph:
    """Cut, bar and scale a chapter's frames inside FFmpeg."""
    left, top, right, bottom = payload["crop"]
    side, size = payload["side"], payload["resolution"]["width"]
    graph = av.filter.Graph()
    graph.link_nodes(
        graph.add_buffer(template=stream),
        graph.add("crop", f"{right - left}:{bottom - top}:{left}:{top}"),
        graph.add("pad", f"{side}:{side}:{payload['pad'][0]}:{payload['pad'][1]}:black"),
        *([graph.add("scale", f"{size}:{size}")] if size < side else []),
        graph.add("format", "yuv420p"),
        graph.add("buffersink"),
    )
    graph.configure()
    return graph


def _encoder(size: int, rate: Fraction, crf: int) -> av.VideoCodecContext:
    """A chapter's own encoder: outside the container it writes its headers into the stream, so one track can change
    size at every chapter."""
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = encoder.height = size
    encoder.pix_fmt = "yuv420p"
    encoder.time_base, encoder.framerate = 1 / rate, rate
    encoder.options = {"crf": str(crf)}
    return encoder


def _payloads(clips: Sequence[ChapterClip], frame_size: tuple[int, int], max_side: int) -> list[dict]:
    """Each clip's payload: where it comes from and how it is cut square, then the clip's own payload."""
    if not clips:
        msg = "no clips to write"
        raise ValueError(msg)
    if any(clip.end_frame <= clip.start_frame for clip in clips) or any(
        later.start_frame < earlier.end_frame for earlier, later in zip(clips, clips[1:], strict=False)
    ):
        msg = "clips must be non-empty, in order and not overlap"
        raise ValueError(msg)
    payloads = []
    for clip in clips:
        cut = square(box=clip.box, frame_size=frame_size, max_side=max_side)
        payloads.append({
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "box": [round(float(value), 2) for value in clip.box],
            "crop": cut.crop,
            "pad": cut.pad,
            "side": cut.side,
            "scale": round(cut.size / cut.side, 6),
            "resolution": {"width": cut.size, "height": cut.size},
            **(clip.payload or {}),
        })
    return payloads


def write_chapters(
    src: str | Path,
    out: str | Path,
    clips: Sequence[ChapterClip],
    *,
    crf: int = CRF,
    max_side: int = MAX_SIDE,
    on_frame: Callable[[int, av.VideoFrame], None] | None = None,
) -> list[dict]:
    """
    Write each clip of ``src`` as a chapter of ``out``, a Matroska file as long as ``src``: its frames cut square
    around its box (see ``square``), at their source times, encoded by x264 at ``crf``.

    Each chapter's ``payload`` tag holds the clip's ``start_frame``, ``end_frame`` and ``box`` (rounded to 0.01 px),
    its ``crop``, ``pad``, ``side``, ``scale`` and ``resolution``, then the clip's own ``payload``. The file's tags
    hold the source's ``source_fps``, ``source_frames``, ``source_width`` and ``source_height``.

    ``on_frame(clip_index, frame)`` sees every source frame inside a clip as decoded, before it is cut, in the same
    single pass over ``src``.

    Returns:
        Each chapter's payload, in order.

    Raises:
        ValueError: If there are no clips, the clips are out of order or overlap, a box leaves an empty crop, ``src``
            is rotated or reports no frame rate or frame count, or ``src`` ends inside a clip.
    """
    meta = video_metadata(str(src))
    if meta.rotation:
        # decoded frames stay unrotated while boxes are in shown pixels, so crops would land elsewhere
        msg = f"{src} is rotated by {meta.rotation} degrees"
        raise ValueError(msg)
    if not meta.fps or not meta.nb_frames:
        msg = f"no frame rate or frame count reported for {src}"
        raise ValueError(msg)
    payloads = _payloads(clips=clips, frame_size=(meta.width, meta.height), max_side=max_side)
    # NTSC rates are n / 1001
    rate = Fraction(meta.fps).limit_denominator(1001)

    with av.open(str(src)) as source:
        stream = source.streams.video[0]
        stream.thread_type = "AUTO"
        with av.open(str(out), mode="w", format="matroska") as container:
            container.metadata.update(
                source_fps=str(rate),
                source_frames=str(meta.nb_frames),
                source_width=str(meta.width),
                source_height=str(meta.height),
            )
            container.set_chapters([
                {
                    "id": k + 1,
                    "start": _milliseconds(frame=payload["start_frame"], rate=rate),
                    "end": _milliseconds(frame=payload["end_frame"], rate=rate),
                    "time_base": Fraction(1, 1000),
                    "metadata": {
                        "title": f"{payload['start_frame']}-{payload['end_frame']}",
                        "payload": json.dumps(payload),
                    },
                }
                for k, payload in enumerate(payloads)
            ])
            # the track's header holds the first chapter's size; every chapter carries its own in the stream
            track = container.add_stream("libx264", rate=rate)
            track.width = track.height = payloads[0]["resolution"]["width"]
            track.pix_fmt = "yuv420p"
            written = _encode(
                source=source,
                track=track,
                payloads=payloads,
                rate=rate,
                crf=crf,
                source_frames=meta.nb_frames,
                on_frame=on_frame or (lambda clip_index, frame: None),
            )
            if written < len(payloads):
                msg = f"{src} ended inside the clip from frame {payloads[written]['start_frame']}"
                raise ValueError(msg)
    return payloads


def _encode(
    source: av.container.InputContainer,
    track: av.VideoStream,
    payloads: list[dict],
    rate: Fraction,
    crf: int,
    source_frames: int,
    on_frame: Callable[[int, av.VideoFrame], None],
) -> int:
    """
    Mux every chapter's frames into ``track`` in one pass over ``source``: frames inside a chapter go through its
    filters and its encoder, at their source time, and the last one lasts until the source ends. Returns how many
    chapters were written whole.
    """
    stream = source.streams.video[0]
    k = 0
    for index, frame in enumerate(source.decode(stream)):
        payload = payloads[k]
        if index < payload["start_frame"]:
            continue
        if index == payload["start_frame"]:
            graph = _graph(stream=stream, payload=payload)
            encoder = _encoder(size=payload["resolution"]["width"], rate=rate, crf=crf)
        on_frame(k, frame)
        graph.push(frame)
        cut = graph.pull()
        cut.pts, cut.time_base = index, 1 / rate
        packets = encoder.encode(cut)
        if index == payload["end_frame"] - 1:
            packets += encoder.encode(None)
            k += 1
        if k == len(payloads):
            # with lookahead on, the encoder holds the last frame until it is flushed
            last = max(packets, key=lambda packet: packet.pts)
            if last.pts != index:
                msg = f"the encoder released frame {index} before its flush"
                raise RuntimeError(msg)
            last.duration = source_frames - index
        for packet in packets:
            packet.stream = track
            track.container.mux(packet)
        if k == len(payloads):
            break
    return k


def _frames_on_timeline(
    frames: Iterable[av.VideoFrame], fps: float, start: int, end: int | None
) -> Generator[tuple[int, av.VideoFrame], None, None]:
    """
    (index, frame) pairs whose index, the frame's timestamp in frames, lies in [start, end].

    A chapter file keeps each frame at its source time with gaps between chapters, so a frame's
    position in the file is not its index; its timestamp is.
    """
    for frame in frames:
        if frame.time is None:
            continue
        index = round(frame.time * fps)
        if end is not None and index > end:
            return
        if index >= start:
            yield index, frame


def _decode_from(
    container: av.container.InputContainer, stream: av.VideoStream, start: float
) -> Generator[av.VideoFrame, None, None]:
    """
    Decode ``stream``, but not the non-reference frames shown before ``start`` seconds.

    No frame refers to them, so the frames from ``start`` on decode the same, and a read that
    seeks to the keyframe before its range skips much of the run up to it. Frames must be
    located by their timestamps, as the skipped ones never come out.
    """
    codec = stream.codec_context
    start_pts = start / stream.time_base
    try:
        for packet in container.demux(stream):
            lead_in = packet.pts is not None and packet.pts < start_pts
            # the decoder reads skip_frame per packet, frame threads included
            if lead_in != (codec.skip_frame == "NONREF"):
                codec.skip_frame = "NONREF" if lead_in else "DEFAULT"
            yield from packet.decode()
    finally:
        # a reused container's next read must decode everything
        codec.skip_frame = "DEFAULT"


def _with_chapters(
    items: Iterable[tuple[int, av.VideoFrame]], chapters: list[Chapter]
) -> Generator[tuple[int, av.VideoFrame, dict | None], None, None]:
    """Each (index, frame) with the payload of the chapter holding the frame's timestamp, {} for a chapter without
    one, None outside every chapter."""
    k = 0
    for index, frame in items:
        payload = None
        if frame.pts is not None:
            time = frame.pts * frame.time_base
            while k < len(chapters) and time >= chapters[k].end:
                k += 1
            if k < len(chapters) and chapters[k].start <= time:
                payload = {} if chapters[k].payload is None else chapters[k].payload
        yield index, frame, payload
