"""Cut a video into clips by time range.

When no pixel change is needed — the target ``size`` is None, or the source is
already ``size`` x ``size`` and unrotated — packets are stream-copied: fast and
lossless. The file still carries the lead-in from the keyframe at/before
``start`` (required for decoding), but an edit list hides it, so playback
starts at ``start``; a few frames past ``end`` may remain visible (B-frame
reordering). Otherwise
frames are decoded, center-cropped to a square, resized, and re-encoded.

Every clip is a streamable MP4 (``moov`` before ``mdat``): a progressive reader
— a pipe, a socket — can demux it without seeking.
"""

import io
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import takewhile
from typing import BinaryIO, NamedTuple

import av
import numpy as np

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import _open_video, video_metadata, video_metadata_from_container
from simple_video_utils.timeline import Timeline

# MP4 output hides each clip's keyframe lead-in behind an edit list, keeping
# packet copy frame-accurate for players — but only for codecs MP4 can carry.
# Everything else is re-encoded so clips stay exact: WebM has no edit list (a
# copied VP8/VP9 clip would visibly start at the previous keyframe), and the eager API keeps its H.264
# fallback for VP9. The ideal source is H.264 in a streamable
# MP4 (faststart/fragmented) or MPEG-TS: opens without seeking and always
# takes the lossless copy path.
_MP4_COPY_CODECS = {"h264", "hevc", "av1", "mpeg4"}


class Clip(NamedTuple):
    """One window of a sliced stream, and where it sits on the source timeline.

    ``start`` is what a consumer cannot recover on its own: windows that yield
    no packets are skipped, so counting the clips it receives does not tell it
    how far into the video each one begins.
    """

    start: float  # seconds from the start of the stream
    data: bytes


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _mp4(buffer: io.BytesIO, samples: int, marks: Sequence[dict] = ()) -> av.container.OutputContainer:
    """An MP4 muxer that puts ``moov`` first: libav reserves the room up front and seeks back into it
    on close (``faststart`` would need a real file). The room bounds the sample tables: ~13 bytes
    per sample for H.264 plus ~700 fixed, so 32 per sample plus 4 KiB is a comfortable ceiling."""
    output = av.open(buffer, mode="w", format="mp4", options={"moov_size": str(4096 + 32 * samples)})
    if marks:
        output.set_chapters(marks)
    return output


def _encode_clip(src: str, start: float, end: float, fps: float, size: int | None, marks: Sequence[dict] = ()) -> bytes:
    frames = list(read_frames_exact(src, start_time=start, end_time=end, return_indices=True))
    if not frames:
        return b""
    height, width = (size, size) if size else frames[0][0].shape[:2]
    buffer = io.BytesIO()
    rate = Fraction(fps).limit_denominator(1000)
    with _mp4(buffer, len(frames), marks) as output:
        stream = output.add_stream("h264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        # One reformatter per clip so the swscale context is built once, not
        # per frame (same fix as _frames_to_rgb) — byte-identical output.
        reformatter = av.video.reformatter.VideoReformatter()
        for position, (frame, index) in enumerate(frames):
            array = _center_crop_square(frame) if size else frame
            video_frame = reformatter.reformat(av.VideoFrame.from_ndarray(array, format="rgb24"), width=width,
                                               height=height)
            # source frame indices: frames cropped out of a chaptered video stay a gap, unless chapters say where
            video_frame.pts = (position if marks else index - frames[0][1])
            video_frame.time_base = 1 / rate
            output.mux(stream.encode(video_frame))
        output.mux(stream.encode())
    return buffer.getvalue()


def _copy_packet(source, destination, start, clock):
    # A packet may be reused by the next slice's keyframe lead-in.
    packet = av.Packet(bytes(source))
    packet.pts = _rebase(source.pts, start, source.time_base, clock)
    packet.dts = _rebase(source.dts, start, source.time_base, clock)
    packet.duration, packet.time_base = source.duration, source.time_base
    packet.is_keyframe, packet.stream = source.is_keyframe, destination
    return packet


def _mux_packets(
    stream: av.VideoStream, packets: list[av.Packet], start: int, clock: Callable | None = None,
    marks: Sequence[dict] = (),
) -> bytes:
    output = io.BytesIO()
    with _mp4(output, len(packets), marks) as container:
        destination = container.add_stream_from_template(stream)
        # mux a copy: the source packets are shared between clips (the keyframe
        # lead-in), so rebasing them in place would corrupt the next clip
        for source in packets:
            container.mux(_copy_packet(source, destination, start, clock))
    return output.getvalue()


def _rebase(pts: int, start: int, time_base: Fraction, clock: Callable | None) -> int:
    """``pts`` relative to ``start`` on ``clock``, which maps file seconds (to the source, or past gaps)."""
    if clock is None:
        return pts - start
    return round((clock(pts * time_base) - clock(start * time_base)) / time_base)


def _stream_pts(seconds: float, stream: av.VideoStream) -> int:
    """``seconds`` on the stream's absolute pts timeline.

    round, not int: a frame landing exactly on a boundary must not be
    excluded by float noise in the division.
    """
    return round(seconds / stream.time_base) + (stream.start_time or 0)


def _window(packets: list[av.Packet], start: int) -> list[av.Packet]:
    """Trim ``packets`` (in place) to the keyframe lead-in needed to decode from
    ``start`` and return the clip, or [] when nothing reaches ``start``."""
    keyframe = next(
        (index for index in reversed(range(len(packets)))
         if packets[index].is_keyframe and packets[index].pts <= start),
        0,
    )
    del packets[:keyframe]
    return packets if any(packet.pts >= start for packet in packets) else []


def _iter_packets(container: av.container.InputContainer, stream: av.VideoStream) -> Iterator[av.Packet]:
    # Matroska may omit the first decode timestamps after a seek. Keep its
    # short reorder prefix until a known DTS lets us recover them backwards.
    pending = []
    for packet in container.demux(stream):
        if packet.pts is None or not packet.size:
            continue
        if packet.dts is None:
            pending.append(packet)
            if len(pending) > 32:
                message = "Cannot recover video decode timestamps within 32 packets"
                raise ValueError(message)
            continue
        dts = packet.dts
        for previous in reversed(pending):
            step = previous.duration or packet.duration
            if not step:
                message = "Cannot recover video decode timestamps without packet durations"
                raise ValueError(message)
            dts -= step
            previous.dts = dts
        yield from pending
        pending.clear()
        yield packet
    if pending:
        message = "Video ended before its decode timestamps could be recovered"
        raise ValueError(message)


def _copy_clip(
    src: str, start: float, end: float, timeline: Timeline, source_end: float, marks: Sequence[dict] = (),
) -> bytes:
    """
    Remux [start, end] seconds without re-encoding.

    The copied packets start on the keyframe at/before ``start`` (the lead-in
    is needed to decode) and may run a few frames past ``end``: packets arrive
    in decode order, so a B-frame with pts <= end can follow a packet with
    pts > end — cutting on pts would drop it. Cutting on dts (monotonic,
    dts <= pts) keeps every frame in range at the cost of a few trailing ones.

    Timestamps are rebased so ``start`` is t=0, putting the lead-in at
    negative pts — the mp4 muxer records that as an edit list, so players
    begin playback at ``start`` and the reported duration excludes the
    lead-in. Consumers that enumerate raw decoded frames still see the
    lead-in (with pts < 0); only presentation skips it.
    """
    with _open_video(src) as source:
        in_stream = source.streams.video[0]
        start_pts = _stream_pts(start, in_stream)
        end_pts = _stream_pts(end, in_stream)
        source.seek(start_pts, stream=in_stream, backward=True)
        # on the source clock: a packed chapter's end is the next chapter's first frame
        packets = list(takewhile(lambda p: p.dts <= end_pts and (not timeline.chapters.size or timeline.to_source(
            p.dts * p.time_base) <= source_end + p.time_base / 2), _iter_packets(source, in_stream)))
        clip = _window(packets, start_pts)
        # gaps between chapters stay gaps, unless chapters say where each stretch came from
        clock = (timeline.squeeze if marks else timeline.to_source) if timeline.chapters.size else None
        return _mux_packets(in_stream, clip, start_pts, clock, marks) if clip else b""


def slice_video(
    src: str,
    slices: Sequence[tuple[float, float]],
    size: int | None = None,
    *,
    chapters: bool = False,
) -> Iterator[bytes]:
    """Yield one MP4 clip (bytes) per (start, end) second range, in order.

    Yields lazily so a long slice list never holds every clip in memory. ``size``
    center-crops each frame to a square and resizes to ``size`` x ``size``; a
    source that is already that size (and unrotated) is stream-copied instead.
    Every slice must be within the video (``0 <= start <= end <= duration``);
    an out-of-range or empty slice raises ``ValueError``.

    A chaptered video takes source seconds: a slice holds the frames its chapters
    keep, starts at t=0 on the first of them, and keeps gaps between chapters.
    A slice wholly cropped out has no frames. ``chapters=True`` gives each clip the
    chapters it spans, timed on the clip, so readers map its frames back to source
    frames (svu's readers and swfast both do).

    Stream-copied clips keep the lead-in from the keyframe before ``start``
    and a few trailing frames past ``end``; an edit list hides them, so
    players are unaffected, but readers that enumerate raw decoded frames —
    including ``read_frames_exact`` and ``read_frames_from_stream`` — see
    them. Pass ``size`` to force re-encoding when exact frames matter.
    Sources whose codec MP4 can't carry (e.g. VP8/VP9 WebM) are always
    re-encoded, at the source resolution when ``size`` is None.
    """
    meta = video_metadata(src)
    with _open_video(src) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        timeline = Timeline.read(container)
        origin = float((stream.start_time or 0) * stream.time_base)
    held = timeline.chapters
    duration = held[-1, 2] + held[-1, 1] - held[-1, 0] if held.size else meta.duration
    should_copy = codec in _MP4_COPY_CODECS and (
        size is None or (meta.width == meta.height == size and meta.rotation == 0)
    )

    for start, end in slices:
        if start < 0 or end <= start or (duration is not None and end > duration):
            msg = f"slice ({start}, {end}) out of range [0, {duration}]"
            raise ValueError(msg)
        window = timeline.window(start, end)
        marks = _clip_chapters(timeline, window, end, meta.fps, should_copy) if chapters and window else ()
        if window is None:
            clip = b""
        elif should_copy:
            # chapter seconds are on the file's clock; _stream_pts counts from its first frame
            clip = _copy_clip(src, *(t - origin if held.size else t for t in window), timeline, end, marks)
        else:
            clip = _encode_clip(src, start, end, meta.fps, size, marks)
        if not clip:
            msg = f"slice ({start}, {end}) has no frames"
            raise ValueError(msg)
        yield clip


def _clip_chapters(timeline: Timeline, window: tuple[float, float], end: float, fps: float, copied: bool) -> list[dict]:
    """The source chapters a slice spans, on its clip, whose frames play back to back: a copy starts at the slice
    start, a re-encode on its first frame. MP4 keeps only chapter titles, so the payload rides in the title.
    """
    first = math.ceil(timeline.to_source(window[0]) * fps - 1e-6)
    rate = str(Fraction(fps).limit_denominator(1001))  # exact: MP4 derives the clip's rate from its duration
    def at(frame):  # a source frame, on the clock that skips gaps
        return timeline.squeeze(timeline.to_file(frame / fps))
    zero = timeline.squeeze(window[0]) if copied else at(first)
    marks = []
    for file_start, file_end, source in timeline.chapters:
        low = max(first, math.ceil(source * fps - 1e-6))
        high = min(math.floor(end * fps + 1e-6), round((source + file_end - file_start) * fps) - 1)  # last frame
        if low <= high:
            begin = at(low) - zero
            marks.append({"id": len(marks), "start": round(begin * 1000),
                          "end": round((begin + (high - low + 1) / fps) * 1000), "time_base": Fraction(1, 1000),
                          "metadata": {"title": json.dumps({"start_frame": low, "rate": rate})}})
    return marks


def _copy_windows(container: av.container.InputContainer, source: av.VideoStream,
                  duration: float) -> Iterator[Clip]:
    """Packet-copied windows: lossless, keyframe lead-in hidden by an edit list."""
    packets: list[av.Packet] = []
    index = 0
    end = _stream_pts(duration, source)
    for packet in _iter_packets(container, source):
        while packet.dts > end:
            start = _stream_pts(index * duration, source)
            if clip := _window(packets, start):
                yield Clip(index * duration, _mux_packets(source, clip, start))
            index += 1
            end = _stream_pts((index + 1) * duration, source)
        packets.append(packet)
    start = _stream_pts(index * duration, source)
    if clip := _window(packets, start):
        yield Clip(index * duration, _mux_packets(source, clip, start))


def _encode_window(frames: list[av.VideoFrame], source: av.VideoStream, start: int) -> bytes:
    output = io.BytesIO()
    with _mp4(output, len(frames)) as container:
        # rate is nominal metadata; the rebased frame pts carry the real timing,
        # so a source without an average_rate only gets a mislabeled fps
        stream = container.add_stream("h264", rate=source.average_rate or 30)
        stream.width, stream.height, stream.pix_fmt = source.width, source.height, "yuv420p"
        reformatter = av.video.reformatter.VideoReformatter()
        for frame in frames:
            video_frame = reformatter.reformat(frame, format="yuv420p")
            video_frame.pts = frame.pts - start
            video_frame.time_base = frame.time_base
            video_frame.pict_type = 0  # let the encoder choose frame types
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())
    return output.getvalue()


def _encode_windows(container: av.container.InputContainer, source: av.VideoStream,
                    duration: float) -> Iterator[Clip]:
    """Frame-exact windows for codecs MP4 can't carry: decode and re-encode."""
    frames: list[av.VideoFrame] = []
    index = 0
    end = _stream_pts(duration, source)
    for frame in container.decode(source):
        while frame.pts >= end:
            if frames:
                yield Clip(index * duration,
                           _encode_window(frames, source, _stream_pts(index * duration, source)))
                frames = []
            index += 1
            end = _stream_pts((index + 1) * duration, source)
        frames.append(frame)
    if frames:
        yield Clip(index * duration,
                   _encode_window(frames, source, _stream_pts(index * duration, source)))


def slice_video_stream(
    stream: BinaryIO,
    duration: float = 0.5,
    buffer_size: int = 32768,  # PyAV default; reduce for lower latency on live sources
) -> Iterator[Clip]:
    """Yield one ``Clip`` per ``duration`` seconds, as the stream arrives.

    Each clip is yielded as soon as its window has been read, so a live source
    produces a clip roughly every ``duration`` seconds (plus up to
    ``buffer_size`` bytes of read-ahead — lower it for lower latency). Sources
    whose codec MP4 can carry are packet-copied: lossless, keeping the keyframe
    lead-in each clip needs to decode, hidden from playback by an edit list.
    Anything else (e.g. VP8/VP9 WebM) is decoded and re-encoded into exact
    windows instead — see ``_MP4_COPY_CODECS``. Memory holds one window plus,
    when copying, the lead-in back to the last keyframe.

    Each clip carries its ``start`` on the source timeline, which a consumer
    cannot derive from the order it receives them: a window with no packets
    yields nothing while the timeline still advances past it.
    """
    if not math.isfinite(duration) or duration <= 0:
        message = "duration must be a positive finite number"
        raise ValueError(message)
    with _open_video(stream, buffer_size=buffer_size) as container:
        source = container.streams.video[0]
        windows = _copy_windows if source.codec_context.name in _MP4_COPY_CODECS else _encode_windows
        yield from windows(container, source, duration)


class _SliceOutput(io.RawIOBase):
    """Keep final MP4 headers on the first pass; emit only sequential media on replay."""

    def __init__(self, header_size: int, *, record: bool):
        self.position = self.length = 0
        self.header_size = header_size
        self.header = bytearray(header_size) if record else None
        self.pending = bytearray()

    def writable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        return self.position

    def write(self, data):
        end = self.position + len(data)
        if self.header is not None:
            if self.position < self.header_size:
                self.header[self.position:min(end, self.header_size)] = data[:self.header_size - self.position]
        elif end > self.header_size:
            if max(self.position, self.header_size) < self.length:
                message = "MP4 muxer attempted to rewrite already-streamed media"
                raise OSError(message)
            self.pending.extend(memoryview(data)[max(0, self.header_size - self.position):])
        self.position, self.length = end, max(self.length, end)
        return len(data)

    def drain(self, *, final=False):
        # Batch small muxer writes within a slice. A caller's own byte stream
        # controls when its chunks are ready for consumption.
        if len(self.pending) >= 64 * 1024 or (final and self.pending):
            data = bytes(self.pending)
            self.pending.clear()
            yield data


def _slice_packets(source, stream, begin, stop, timeline, end, keyframe=None):
    source.seek(begin if keyframe is None else keyframe, stream=stream, backward=True)
    ready = keyframe is None
    for packet in _iter_packets(source, stream):
        if packet.dts > stop or (timeline.chapters.size and timeline.to_source(
                packet.dts * packet.time_base) > end + packet.time_base / 2):
            break
        if not ready:
            ready = packet.is_keyframe and packet.pts == keyframe
        if ready:
            yield packet


def _write_slice(output, stream, packets, samples, begin, clock, marks):
    with _mp4(output, samples, marks) as container:
        destination = container.add_stream_from_template(stream)
        count = 0
        for source in packets:
            count += 1
            container.mux(_copy_packet(source, destination, begin, clock))
            yield from output.drain()
        if count != samples:
            message = "Video source changed between slice reads"
            raise OSError(message)
    yield from output.drain(final=True)


def _slice_stats(packets, begin, stop):
    samples, keyframe, visible = 0, None, False
    for packet in packets:
        if packet.is_keyframe and packet.pts <= begin:
            samples, keyframe, visible = 0, packet.pts, False
        samples += 1
        visible |= begin <= packet.pts <= stop
    if not visible:
        message = "slice has no frames"
        raise ValueError(message)
    return samples, keyframe


@dataclass
class _Slice:
    """A replayable lossless slice. Cache scalars, never headers or encoded media."""

    src: str
    start: float
    end: float
    chapters: bool = False
    _stats: tuple | None = field(default=None, init=False, repr=False)
    _length: int | None = field(default=None, init=False, repr=False)
    _duration: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            message = "slice times must be finite and satisfy 0 <= start < end"
            raise ValueError(message)

    def _plan(self, source):
        stream = source.streams.video[0]
        timeline = Timeline.read(source)
        held = timeline.chapters
        duration = (float(held[-1, 2] + held[-1, 1] - held[-1, 0]) if held.size else
                    float(stream.duration * stream.time_base) if stream.duration is not None else
                    source.duration / av.time_base if source.duration is not None else None)
        if duration is not None and self.end > duration:
            message = f"slice ({self.start}, {self.end}) out of range [0, {duration}]"
            raise ValueError(message)
        window = timeline.window(self.start, self.end)
        if window is None:
            message = f"slice ({self.start}, {self.end}) has no frames"
            raise ValueError(message)
        rate = timeline.rate or stream.average_rate or stream.guessed_rate
        if self.chapters and held.size and not rate:
            message = "Chapter mapping requires a source frame rate"
            raise ValueError(message)
        marks = (_clip_chapters(timeline, window, self.end, float(rate), copied=True)
                 if self.chapters and held.size else [])
        origin = float((stream.start_time or 0) * stream.time_base)
        low, high = (time - origin if held.size else time for time in window)
        begin, stop = _stream_pts(low, stream), _stream_pts(high, stream)
        clock = (timeline.squeeze if marks else timeline.to_source) if held.size else None
        if self._stats is None:
            self._stats = _slice_stats(
                _slice_packets(source, stream, begin, stop, timeline, self.end), begin, stop)
        return stream, timeline, begin, stop, clock, marks

    def duration_seconds(self):
        if self._duration is None:
            with _open_video(self.src) as source:
                self._header(source, *self._plan(source))
        return self._duration

    def size_bytes(self):
        self.duration_seconds()
        return self._length

    def _header(self, source, stream, timeline, begin, stop, clock, marks):
        samples, keyframe = self._stats
        # _mp4 reserves 4096 + 32 bytes/sample; the extra covers ftyp, mdat and its size patch.
        header_size = 4096 + 32 * samples + 1024
        recorded = _SliceOutput(header_size, record=True)
        packets = _slice_packets(source, stream, begin, stop, timeline, self.end, keyframe)
        for _ in _write_slice(recorded, stream, packets, samples, begin, clock, marks):
            pass
        assert recorded.header is not None
        header = bytes(recorded.header[:min(recorded.length, header_size)])
        length, header_size = recorded.length, len(header)
        del recorded
        if self._length is not None and length != self._length:
            message = "Video source changed between slice reads"
            raise OSError(message)
        self._length = length
        with av.open(io.BytesIO(header), mode="r") as container:
            self._duration = video_metadata_from_container(container, rotation=0).duration
        if self._duration is None:
            message = "Could not determine slice duration"
            raise ValueError(message)
        return header

    def __call__(self):
        with _open_video(self.src) as source:
            plan = self._plan(source)
            stream, timeline, begin, stop, clock, marks = plan
            header = self._header(source, *plan)
            length, header_size = self._length, len(header)
            samples, keyframe = self._stats
            yield header
            del header
            if header_size < length:
                output = _SliceOutput(header_size, record=False)
                yield from _write_slice(output, stream,
                                        _slice_packets(source, stream, begin, stop, timeline, self.end, keyframe),
                                        samples, begin, clock, marks)
                if output.length != length:
                    message = "Video source changed while replaying the slice"
                    raise OSError(message)
