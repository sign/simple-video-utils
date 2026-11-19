"""One video: its metadata, frames and chapters, over one PyAV container."""

import io
import json
import math
import operator
import os
import ssl
import struct
import sys
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from fractions import Fraction
from functools import cached_property, lru_cache
from itertools import groupby, pairwise
from typing import BinaryIO, NamedTuple

import av
import certifi
import numpy as np
from av.video.reformatter import Interpolation, VideoReformatter

# Below this many pixels swscale's thread dispatch costs more than the RGB conversion itself (2x at 320x240, even at
# 640x360, a win from ~960x540). The crossover depends on the machine, so it is a knob. Pixels are the same either way.
SINGLE_THREAD_MAX_PIXELS = 300_000


class VideoMetadata(NamedTuple):
    width: int  # as displayed: rotation applied
    height: int
    fps: float  # 0.0 when the header has no rate and the source cannot be rewound to measure one
    nb_frames: int | None  # best effort: the header, checked against duration x fps, decoded when they disagree
    time_base: str | None
    duration: float | None  # seconds
    rotation: int = 0  # degrees, from the container's display matrix


class Frames(NamedTuple):
    frames: np.ndarray  # (N, H, W, 3) uint8 RGB
    indices: np.ndarray  # each frame's index in the video


class Chapter(NamedTuple):
    """A stretch of frames with their source timing; the seconds between chapters hold no frames."""

    id: int
    start: Fraction  # seconds in this file
    end: Fraction
    source: Fraction  # where `start` sits on the video this was cut from: the same, except in a slice
    payload: dict


class PackedChaptersError(ValueError):
    """The file plays its chapters back to back, as versions before 0.10 wrote them. Only gapped files are read."""


def _open(src, **options) -> av.container.InputContainer:
    try:
        if sys.platform == "linux" and isinstance(src, str) and src.startswith("https://"):
            # PyAV's Linux wheels carry no trust store
            options["options"] = {"ca_file": ssl.get_default_verify_paths().cafile or certifi.where()}
        return av.open(io.BytesIO(src) if isinstance(src, bytes) else src, **options)
    except Exception as e:
        raise RuntimeError("Failed to open video") from e


def rotation_filters(graph: av.filter.Graph, rotation: int) -> list:
    """Filters that turn stored pixels upright; 90 is counterclockwise, as in the display matrix."""
    if rotation == 180:
        return [graph.add("hflip"), graph.add("vflip")]
    return [graph.add("transpose", "cclock" if rotation == 90 else "clock")] if rotation in (90, 270) else []


def _rotation_graph(rotation: int, width: int, height: int) -> av.filter.Graph:
    graph = av.filter.Graph()
    source = graph.add_buffer(width=width, height=height, format="rgb24", time_base=Fraction(1, 1000))
    graph.link_nodes(source, *rotation_filters(graph, rotation), graph.add("buffersink"))
    graph.configure()
    return graph


def _pick(items: Iterable[tuple], fps: float, keep_last: bool) -> Iterator[tuple]:
    """Thin frames out to about `fps`: the first of each 1/fps bucket, counted from each chapter's first frame.

    Frames are dropped, never repeated, so a target at or above the source rate keeps them all. `keep_last` also
    keeps the last frame of each chapter, which would otherwise be dropped when it shares a bucket.
    """
    anchor = bucket = dropped = None
    current = -1
    for item in items:
        _, chapter, frame = item
        if (time := frame.time) is None:
            continue
        if anchor is None or chapter != current:
            if dropped:
                yield dropped
            anchor, bucket, current = time, None, chapter
        # one tick of slack: containers round times to their clock (a 15 fps mkv stores 0, 67, 133 ms)
        if (now := int((time - anchor + float(frame.time_base)) * fps)) != bucket:
            bucket, dropped = now, None
            yield item
        elif keep_last:
            dropped = item
    if dropped:
        yield dropped


def _rgb(items: Iterable[tuple], size: tuple[int, int] | None) -> Iterator[tuple]:
    """Resize the native planes, convert to RGB, then turn upright: rotating first would change the chroma."""
    scale, convert = VideoReformatter(), VideoReformatter()  # each keeps its swscale context for the whole read
    graph = lru_cache(maxsize=1)(_rotation_graph)  # stateful, so one per read
    for index, chapter, frame in items:
        rotation = frame.rotation % 360
        if size:
            width, height = size[::-1] if rotation in (90, 270) else size
            frame = scale.reformat(frame, width=width, height=height, interpolation=Interpolation.AREA, threads=1)
        threads = 1 if frame.width * frame.height < SINGLE_THREAD_MAX_PIXELS else 0
        rgb = convert.reformat(frame, format="rgb24", threads=threads)
        if rotation in (90, 180, 270):
            turn = graph(rotation, rgb.width, rgb.height)
            turn.push(rgb)
            rgb = turn.pull()
        # padded lines come out as strided views, which image libraries reject
        yield index, chapter, np.ascontiguousarray(rgb.to_ndarray())


def stack_frames(frames: Iterable[np.ndarray], size_hint: int = 0) -> np.ndarray:
    """`np.stack(list(frames))`, copying each array into the batch as it arrives, while it is still in cache.

    With an exact `size_hint` the batch is allocated once and returned as is.
    """
    chunks, buffer, count, size = [], None, 0, size_hint or 64
    for array in frames:
        if buffer is None:
            buffer, count, size = np.empty((size, *array.shape), array.dtype), 0, size * 2
        if array.shape != buffer.shape[1:]:
            raise ValueError(f"frames have different shapes: {buffer.shape[1:]} and {array.shape}")
        buffer[count] = array
        count += 1
        if count == len(buffer):
            chunks.append(buffer)
            buffer = None
    if buffer is not None:
        if not chunks:
            buffer.resize((count, *buffer.shape[1:]), refcheck=False)  # a hint that ran long: give the rest back
            return buffer
        chunks.append(buffer[:count])
    if not chunks:
        raise ValueError("no frames to stack")
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class Video:
    """A video opened once: a path, a URL, bytes, or a binary stream (rewound between reads when it can seek).

    `thread_type` is PyAV's ("AUTO", "FRAME", "SLICE", "NONE"); pass "NONE" in processes that fork, where an
    inherited threaded decoder can deadlock. A smaller `buffer_size` yields frames sooner on a live stream.
    """

    def __init__(self, src: str | os.PathLike | bytes | BinaryIO, thread_type: str = "AUTO", buffer_size: int = 32768):
        self.src = os.fspath(src) if isinstance(src, os.PathLike) else src
        self.seekable = isinstance(self.src, (str, bytes)) or getattr(src, "seekable", lambda: False)()
        self.container = _open(self.src, buffer_size=buffer_size)
        if not self.container.streams.video:
            self.container.close()
            raise RuntimeError("Failed to open video: it has no video stream")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = thread_type
        self._moved = False  # whether a read has left the container away from its start

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.container.close()

    def rewind(self) -> None:
        """Back to the first frame, for reading `container` directly after a read has moved it."""
        if self._moved:
            if not self.seekable:
                raise ValueError("this stream cannot seek, so it can be read only once")
            self.container.seek(0)
        self._moved = True

    @cached_property
    def rotation(self) -> int:
        matrix = self.stream.codec_context.coded_side_data.get("display_matrix")
        if not matrix:
            return 0
        a, b = struct.unpack_from("2i", matrix)
        return round(-math.degrees(math.atan2(b, a))) % 360

    @cached_property
    def duration(self) -> float | None:
        # the video stream's own length when it has one: the container's follows its longest stream, often audio
        stream = self.stream
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        return self.container.duration / av.time_base if self.container.duration else None

    @cached_property
    def chapters(self) -> list[Chapter]:
        """The chapters this library wrote, in order. Other chapter markers are not chapters here."""
        found, ahead = [], []
        for chapter in sorted(self.container.chapters(), key=lambda c: c["start"] * c["time_base"]):
            tags = {key.lower(): value for key, value in chapter["metadata"].items()}
            start, end = (chapter[key] * chapter["time_base"] for key in ("start", "end"))
            try:
                payload = json.loads(tags.get("payload") or tags.get("title") or "")  # MP4 keeps only a title
                rate = Fraction(payload.get("rate") or self.stream.average_rate)
                source = Fraction(payload["start_frame"]) / rate
                if "end_frame" in payload and "payload" not in tags:
                    # MP4 ends a chapter where the next one starts, gap included; Matroska keeps the real end
                    end = min(end, start + (payload["end_frame"] - payload["start_frame"]) / rate)
            except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError):
                continue  # an ordinary marker: its title is not our JSON
            found.append(Chapter(chapter["id"], start, end, source, payload))
            # how far the file's clock runs from the source's, in frames: nothing in a whole file (a Matroska payload
            # tag), one amount throughout a slice (an MP4 title). A packed file closes its gaps, so it drifts.
            base = 0 if "payload" in tags else found[0].source - found[0].start
            ahead.append((source - start - base) * rate)
        if any(abs(frames) > Fraction(1, 2) for frames in ahead):
            raise PackedChaptersError("chapters are packed back to back; rewrite the file with version 0.10 or later")
        if any(c.end <= c.start for c in found) or any(a.end > b.start for a, b in pairwise(found)):
            raise ValueError("chapters must not overlap and must have positive duration")
        return found

    @cached_property
    def _decoded(self) -> tuple[float | None, int | None]:
        """(fps, frame count) from decoding every frame: the truth when the header has no rate or miscounts."""
        if not self.seekable:
            return None, None
        self.rewind()
        first = last = None
        count = 0
        try:
            for frame in self.container.decode(self.stream):
                count += 1
                if frame.time is not None:
                    first, last = frame.time if first is None else first, frame.time
        except (av.FFmpegError, OSError):
            return None, None
        return ((count - 1) / (last - first) if last is not None and last > first else None), count

    @cached_property
    def fps(self) -> float:
        # a slice says its rate in its chapters: gaps throw off the rate MP4 derives from frames over duration
        rates = {c.payload["rate"] for c in self.chapters if "rate" in c.payload}
        rate = Fraction(rates.pop()) if len(rates) == 1 else self.stream.average_rate
        # browser recordings (MediaRecorder WebM) carry no rate at all
        return float(rate or self._decoded[0] or 0)

    def _tail_count(self) -> int | None:
        """Frame count from the last packet's time: reads the final keyframe's worth, not the file."""
        stream, fps, duration = self.stream, self.fps, self.duration
        if not fps or not duration or not stream.time_base:
            return None
        origin = stream.start_time or 0
        self._moved = True
        try:
            self.container.seek(origin + int(duration / stream.time_base), stream=stream)
            last = max((p.pts for p in self.container.demux(stream) if p.pts is not None and p.size), default=None)
        except (av.FFmpegError, OSError):
            return None
        gaps = sum(b.start - a.end for a, b in pairwise(self.chapters))
        return None if last is None else round(((last - origin) * stream.time_base - gaps) * fps) + 1

    @cached_property
    def meta(self) -> VideoMetadata:
        stream, fps, duration = self.stream, self.fps, self.duration
        header = stream.frames or None  # Matroska and WebM often have none; some MOV headers overcount
        expected = round(duration * fps) if duration and fps else None
        if not stream.average_rate and self._decoded[1] is not None:
            frames = self._decoded[1]
        elif header is None:
            frames = (self._tail_count() if self.seekable else None) or expected
        elif not self.seekable or expected is None or abs(header - expected) <= 1:
            frames = header
        else:
            frames = header if self._decoded[1] is None else self._decoded[1]
        width, height = (stream.height, stream.width) if self.rotation % 180 == 90 else (stream.width, stream.height)
        time_base = str(stream.time_base) if stream.time_base else None
        return VideoMetadata(width, height, fps, frames, time_base, duration, self.rotation)

    def _range(self, start, stop, seconds, fps, size, keep_last=False) -> tuple:
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        if keep_last and fps is None:
            raise ValueError("keep_last only applies with fps")
        if size is not None and (len(size) != 2 or min(map(operator.index, size)) <= 0):
            raise ValueError("size must be two positive integers, (width, height)")
        if seconds:  # a negative start reads from the beginning: padded windows are common
            if stop is not None and stop < max(start, 0):
                raise ValueError("invalid time range")
            return start, stop
        start, stop = operator.index(start), None if stop is None else operator.index(stop)
        if start < 0 or (stop is not None and stop < start):
            raise ValueError("invalid frame range")
        return start, stop

    def _decode(self, start, stop, seconds=False) -> Iterator[tuple]:  # noqa: C901 - one loop, one indexing rule
        """(index, chapter number, frame) for frames [start, stop): frame indices, or with `seconds` their times.

        Frames are counted. After a seek the first one is placed by its timestamp; so is the first of each chapter,
        which makes the frames missing between chapters count too.
        """
        stream, fps, chapters = self.stream, self.fps, self.chapters
        if stop is not None and stop <= start:
            return
        if (start > 0 or chapters) and not fps:
            raise ValueError("Video has no FPS information")
        first = float((stream.start_time or 0) * stream.time_base)
        # seconds from a frame's index time to its place in this file: a slice starts mid-source
        shift = [float(c.start - c.source) for c in chapters] or [first]
        starts, ends, half = [float(c.start) for c in chapters], [float(c.end) for c in chapters], 0.5 / (fps or 1)
        tick = float(stream.time_base) / 2  # files round times to their clock, so a frame can sit half a tick early
        at = (start if seconds else start / fps if fps else 0) + shift[0]
        if seeking := self.seekable and at - first >= 3:  # nearer than that, decoding from the start is as fast
            self.container.seek(int((at - 1) / stream.time_base), stream=stream)
            self._moved = True
        else:
            self.rewind()
        # a header rate can drift from the timestamps (#26). guessed_rate follows them, but is garbage without one
        locate = float(stream.guessed_rate) if stream.average_rate and stream.guessed_rate else fps
        index, chapter = (None if seeking else -1), None
        try:
            for frame in self.container.decode(stream):
                if chapters:
                    time = frame.time + half  # half a frame of slack: Matroska keeps milliseconds
                    number = bisect_right(starts, time) - 1
                    if number < 0 or time >= ends[number]:
                        continue  # the hidden edges of a slice
                    if number != chapter:
                        chapter, index = number, round((frame.time - shift[number]) * fps) - 1
                elif index is None:  # after a seek, the frame's time says which one it is
                    if frame.time is None:
                        continue
                    index = round((frame.time - first) * locate) - 1
                    if not seconds:  # and the first to reach `start` is `start`, however the rate has drifted
                        if index + 1 < start:
                            index = None
                            continue
                        index = start - 1
                index += 1
                # a timed read goes by each frame's own time: an average rate cannot place the frames of a variable one
                position = frame.time - shift[chapter or 0] + tick if seconds else index
                if stop is not None and position >= stop:
                    return
                if position >= start:
                    yield index, chapter, frame
        except av.FFmpegError as e:
            raise RuntimeError("Failed to decode video") from e

    def _frames(self, start, stop, seconds, fps, size, keep_last) -> Iterator[tuple]:
        items = self._decode(start, stop, seconds)
        return _rgb(_pick(items, fps, keep_last) if fps else items, size)

    def frames(self, start=0, stop=None, *, seconds=False, fps=None, keep_last=False, size=None,
               indices=False) -> Iterator:
        """RGB (H, W, 3) uint8 arrays for frames [start, stop), decoded as they are asked for.

        `seconds=True` takes times instead of frame indices, and selects by each frame's own timestamp. `fps` drops
        frames down to about that rate; it always keeps the first frame, and with `keep_last=True` the last one too.
        `size` is (width, height) as displayed. `indices=True` yields (array, frame index) pairs. In a file with
        chapters, indices and times are the source's, so the frames between chapters are skipped, not renumbered, and
        `fps` treats each chapter as its own range. A stream that cannot seek is read once.
        """
        start, stop = self._range(start, stop, seconds, fps, size, keep_last)
        frames = self._frames(start, stop, seconds, fps, size, keep_last)
        return ((array, index) for index, _, array in frames) if indices else (array for _, _, array in frames)

    def read(self, start=0, stop=None, *, seconds=False, fps=None, keep_last=False, size=None) -> Frames:
        """`frames()` as one batch: (frames, indices). Raises ValueError when the range is empty."""
        start, stop = self._range(start, stop, seconds, fps, size, keep_last)
        # the header's count, else an estimate: it only sizes the batch
        held = sum(float(c.end - c.start) for c in self.chapters) or self.duration or 0
        total = self.stream.frames or math.ceil(held * self.fps)
        low, high = (max(start, 0), stop) if not seconds else (math.ceil(max(start, 0) * self.fps),
                                                               None if stop is None else math.ceil(stop * self.fps))
        wanted = (total if high is None else high) - low  # a chaptered read asks in source frames, more than it holds
        hint = max(min(wanted, total or wanted), 0)
        if hint and fps and fps < self.fps:
            hint = int(hint * fps / self.fps) + 1 + keep_last * max(len(self.chapters), 1)
        numbers = []
        frames = self._frames(start, stop, seconds, fps, size, keep_last)
        frames = stack_frames((numbers.append(i) or a for i, _, a in frames), hint)
        return Frames(frames, np.array(numbers, dtype=np.int64))

    def chapter_frames(self, fps=None, size=None, keep_last=False) -> Iterator[tuple[Chapter | None, Iterator]]:
        """(chapter, its RGB frames) per chapter, from one decode, so it works on a stream that cannot seek.

        A file without chapters yields (None, frames) once. Each iterator ends when the next chapter is asked for.
        """
        frames = self._frames(*self._range(0, None, False, fps, size, keep_last), False, fps, size, keep_last)
        for number, group in groupby(frames, key=operator.itemgetter(1)):
            yield (None if number is None else self.chapters[number]), (array for _, _, array in group)

    def keyframes(self) -> list[int]:
        """Frame indices of the keyframes, in presentation order. Packets are read, nothing is decoded."""
        self.rewind()
        packets = [(p.pts if p.pts is not None else p.dts, p.is_keyframe)
                   for p in self.container.demux(self.stream) if p.size]
        order = range(len(packets))
        if all(time is not None for time, _ in packets):  # B-frames arrive out of presentation order
            order = sorted(order, key=lambda i: packets[i][0])
        return [index for index, i in enumerate(order) if packets[i][1]]

    def slice(self, start: float, stop: float):
        """Seconds [start, stop) as a lossless MP4 that is read lazily: see `Slice`."""
        from simple_video_utils.slicing import Slice

        return Slice(self.src, start, stop)


@lru_cache(maxsize=8)
def video_metadata(path: str | os.PathLike) -> VideoMetadata:
    with Video(path) as video:
        return video.meta
