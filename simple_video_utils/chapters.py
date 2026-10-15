"""
Chapter files: clips of a video, each run through its own FFmpeg filters and played back to back as the chapters of
one Matroska video track.

Each chapter keeps the frame size its filters give it and carries a JSON ``payload`` tag naming the source frames it
came from. ``write_chapters`` writes one and ``list_chapters`` lists its chapters; its frames read like any video's,
with ``simple_video_utils.frames``.
"""

import json
import math
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from itertools import accumulate, islice
from pathlib import Path
from typing import NamedTuple

import av
from av.video.frame import PictureType

from simple_video_utils.metadata import open_video


class Chapter(NamedTuple):
    """A chapter's [start, end) in seconds and the JSON object in its ``payload`` tag, None without one."""

    start: Fraction
    end: Fraction
    payload: dict | None


class ChapterClip(NamedTuple):
    """Source frames [start_frame, end_frame), each run through ``filters``, an FFmpeg filter chain such as
    ``"crop=96:64:16:32,pad=96:96:0:16"``, with more JSON for the chapter's payload tag."""

    start_frame: int
    end_frame: int
    filters: str = "null"
    payload: dict | None = None


def list_chapters(src: str | Path) -> Iterator[Chapter]:
    """Every chapter of ``src``, in the order the file keeps them."""
    with open_video(str(src)) as container:
        for chapter in container.chapters():
            tags = {key.lower(): value for key, value in chapter["metadata"].items()}
            payload = json.loads(tags["payload"]) if "payload" in tags else None
            yield Chapter(chapter["start"] * chapter["time_base"], chapter["end"] * chapter["time_base"], payload)


def write_chapters(
    src: str | Path,
    out: str | Path,
    clips: Sequence[ChapterClip],
    *,
    crf: int = 23,
    thread_type: str = "AUTO",
    on_frame: Callable[[int, av.VideoFrame], None] | None = None,
) -> None:
    """
    Write each clip of ``src`` as a chapter of ``out``, a Matroska file playing them back to back at the source's
    frame rate: each clip's frames run through its filters and are encoded by x264 at ``crf``, in the pixel format the
    filters leave.

    Each chapter's ``payload`` tag holds the clip's ``start_frame`` and ``end_frame``, then the clip's own ``payload``.
    Filters see frames as decoded, unrotated, and must give one frame for each frame.

    ``on_frame(clip_index, frame)`` sees every source frame of a clip as decoded, before its filters, in the same
    single pass over ``src``.

    Raises:
        ValueError: If there are no clips, the clips are out of order or overlap, ``src`` reports no frame rate or
            ends inside a clip, or a chapter's size covers the previous one's macroblocks without matching it.
    """
    if not clips:
        msg = "no clips to write"
        raise ValueError(msg)
    if any(clip.end_frame <= clip.start_frame for clip in clips) or any(
        later.start_frame < earlier.end_frame for earlier, later in zip(clips, clips[1:], strict=False)
    ):
        msg = "clips must be non-empty, in order and not overlap"
        raise ValueError(msg)

    with open_video(str(src), thread_type=thread_type) as source:
        stream = source.streams.video[0]
        rate = stream.average_rate
        if not rate:
            msg = f"no frame rate reported for {src}"
            raise ValueError(msg)
        # each chapter's first and last frame in the file
        bounds = [0, *accumulate(clip.end_frame - clip.start_frame for clip in clips)]
        with av.open(str(out), mode="w", format="matroska") as container:
            container.set_chapters(
                [
                    {
                        "id": k + 1,
                        "start": _milliseconds(frame=start, rate=rate),
                        "end": _milliseconds(frame=end, rate=rate),
                        "time_base": Fraction(1, 1000),
                        "metadata": {
                            "title": f"{clip.start_frame}-{clip.end_frame}",
                            "payload": json.dumps(
                                {"start_frame": clip.start_frame, "end_frame": clip.end_frame, **(clip.payload or {})}
                            ),
                        },
                    }
                    for k, (clip, start, end) in enumerate(zip(clips, bounds, bounds[1:], strict=False))
                ]
            )
            track = container.add_stream("libx264", rate=rate)
            _encode(
                stream=stream,
                track=track,
                clips=clips,
                rate=rate,
                crf=crf,
                on_frame=on_frame or (lambda clip_index, frame: None),
            )


def _milliseconds(frame: int, rate: Fraction) -> int:
    """``frame``'s time as Matroska stores a block's, rounded half up to the millisecond, so a seek to a chapter's
    start lands on its first frame."""
    return math.floor(frame * 1000 / rate + Fraction(1, 2))


def _graph(stream: av.VideoStream, filters: str) -> av.filter.Graph:
    """A clip's filter chain, inside FFmpeg."""
    graph = av.filter.Graph()
    # ponytail: split on "," and each filter's first "=", so no filter's arguments may hold a ","
    chain = [graph.add(name, args or None) for name, _, args in (f.partition("=") for f in filters.split(","))]
    graph.link_nodes(graph.add_buffer(template=stream), *chain, graph.add("buffersink"))
    graph.configure()
    return graph


def _encoder(frame: av.VideoFrame, rate: Fraction, crf: int) -> av.VideoCodecContext:
    """A chapter's own encoder, at its first frame's size and pixel format: outside the container it writes its
    headers into the stream, so one track can change size at every chapter."""
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = frame.width, frame.height, frame.format.name
    encoder.time_base, encoder.framerate = 1 / rate, rate
    encoder.options = {"crf": str(crf)}
    return encoder


def _mux(track: av.VideoStream, packets: list[av.Packet]) -> None:
    for packet in packets:
        packet.stream = track
        track.container.mux(packet)


def _encode(
    stream: av.VideoStream,
    track: av.VideoStream,
    clips: Sequence[ChapterClip],
    rate: Fraction,
    crf: int,
    on_frame: Callable[[int, av.VideoFrame], None],
) -> None:
    """Mux every clip's frames into ``track``, back to back, in one pass over ``stream``: each clip through its
    filters and its own encoder."""
    frames, read, written, size, blocks = stream.container.decode(stream), 0, 0, None, None
    for k, clip in enumerate(clips):
        graph, encoder, first = _graph(stream=stream, filters=clip.filters), None, written
        for frame in islice(frames, clip.start_frame - read, clip.end_frame - read):
            on_frame(k, frame)
            graph.push(frame)
            cut = graph.pull()
            # x264 picks its own frame types, not the source's
            cut.pts, cut.time_base, cut.pict_type = written, 1 / rate, PictureType.NONE
            written += 1
            if encoder is None:
                previous, size = size, (cut.width, cut.height)
                previous_blocks, blocks = blocks, [math.ceil(side / 16) for side in size]
                # FFmpeg's decoder can keep the previous chapter's size when the new one covers the same 16-px
                # macroblocks and differs only in its crop, as 470 and 472 do
                if size != previous and blocks == previous_blocks:
                    msg = f"the clip from frame {clip.start_frame} is {size}, on the macroblocks of the previous "
                    msg += f"chapter's {previous}: decoders would show it at that size"
                    raise ValueError(msg)
                encoder = _encoder(frame=cut, rate=rate, crf=crf)
                if not k:
                    # the track's header holds the first chapter's size; every chapter carries its own in the stream
                    track.width, track.height, track.pix_fmt = cut.width, cut.height, cut.format.name
            _mux(track=track, packets=encoder.encode(cut))
        if written - first != clip.end_frame - clip.start_frame:
            msg = f"{stream.container.name} ended inside the clip from frame {clip.start_frame}"
            raise ValueError(msg)
        _mux(track=track, packets=encoder.encode(None))
        read = clip.end_frame
