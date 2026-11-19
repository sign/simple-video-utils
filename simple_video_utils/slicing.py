"""Cut seconds out of a video without re-encoding, as an MP4 that is never held in memory whole."""

import json
import math
import os
from bisect import bisect_left
from collections.abc import Iterator
from fractions import Fraction
from itertools import islice

import av

from simple_video_utils.video import SLACK, Video


class _Sink:
    """The file libav's MP4 muxer writes to: keeps the first `head` bytes, which the muxer returns to on close to
    fill in the moov, and queues or drops the media after them."""

    def __init__(self, head: int, queue: bool):
        self.head, self.queue, self.media = bytearray(head), queue, bytearray()
        self.position = self.size = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0, self.position, self.size)[whence]
        return self.position

    def write(self, data):
        end, head = self.position + len(data), len(self.head)
        if self.position < head:
            self.head[self.position:min(end, head)] = data[:head - self.position]
        if end > head and self.queue:
            if max(self.position, head) < self.size:
                raise OSError("MP4 muxer rewrote media that was already read")
            self.media += memoryview(data)[max(head - self.position, 0):]
        self.position, self.size = end, max(self.size, end)
        return len(data)

    def drain(self, least: int = 1) -> Iterator[bytes]:
        if len(self.media) >= least:
            yield bytes(self.media)
            self.media.clear()


def _with_dts(packets) -> Iterator[av.Packet]:
    # Matroska can omit the first decode timestamps after a seek: hold that short reordered run until a known one
    # lets them be counted backwards
    pending = []
    for packet in packets:
        if packet.pts is None or not packet.size:
            continue
        if packet.dts is None:
            pending.append(packet)
            if len(pending) > 32:
                raise ValueError("Cannot recover video decode timestamps within 32 packets")
            continue
        dts = packet.dts
        for previous in reversed(pending):
            if not (step := previous.duration or packet.duration):
                raise ValueError("Cannot recover video decode timestamps without packet durations")
            previous.dts = dts = dts - step
        yield from pending
        pending.clear()
        yield packet
    if pending:
        raise ValueError("Video ended before its decode timestamps could be recovered")


def _packets(video: Video, begin: int, end: int, keyframe: tuple | None = None) -> Iterator[av.Packet]:
    """Packets in decode order, from where a seek to `begin` lands, until one decodes at `end`.

    With `keyframe`, a (pts, dts), from that packet instead: sought by its decode time, which is where MPEG-TS seeks.
    """
    video.container.seek(begin if keyframe is None else min(keyframe), stream=video.stream)
    ready = keyframe is None
    for packet in _with_dts(video.container.demux(video.stream)):
        if packet.dts >= end:
            return
        ready = ready or packet.pts == keyframe[0]
        if ready:
            yield packet


def _keyframe_header(stream: av.VideoStream, packet: av.Packet) -> tuple[bytes, int, int] | None:
    """(avcC, width, height) of the H.264 parameter sets `packet` carries, when they are not the stream's own.

    A file whose chapters differ in size has one header, the first chapter's, and each later keyframe carries its own.
    """
    own = stream.codec_context.extradata
    if stream.codec_context.name != "h264" or not own or own[0] != 1:
        return None
    size, data, sets, at = (own[4] & 3) + 1, bytes(packet), {7: [], 8: []}, 0  # 7: sequence sets, 8: picture sets
    while at + size <= len(data):
        length = int.from_bytes(data[at:at + size])
        unit = data[at + size:at + size + length]
        at += size + length
        if unit and unit[0] & 31 in sets:
            sets[unit[0] & 31].append(unit)
    if not sets[7] or not sets[8] or sets[7][0] in own:
        return None
    header = bytes([1, *sets[7][0][1:4], 0xFC | (size - 1)])
    for kind, mask in ((7, 0xE0), (8, 0)):
        header += bytes([mask | len(sets[kind])]) + b"".join(len(unit).to_bytes(2) + unit for unit in sets[kind])
    decoder = av.CodecContext.create("h264", "r")  # the size is in the sequence set: let libav read it
    decoder.extradata = header
    frame = (decoder.decode(av.Packet(data)) + decoder.decode(None))[0]
    return header, frame.width, frame.height


class Slice:
    """Seconds [start, stop) of a video as a streamable MP4 (moov first), produced as it is read.

    Iterate for its bytes; iterate again to replay it. Packets are copied, so nothing is re-encoded and codecs MP4
    cannot carry raise. Times are matched at the precision of the file's clock (a millisecond in Matroska). The clip
    starts on its first frame and holds the frames from the keyframe before that one, hidden behind an edit list.
    With B-frames it also holds the later frames its last ones are predicted from, and those stay visible: a few
    frames at or past `stop`, not always consecutive. Gaps between chapters stay gaps, and the chapters it spans
    come along, so `Video` reads exactly the frames in range from a chaptered slice. Where chapters differ in size
    the clip's header describes the chapter it starts in. The source must not change between reads.
    """

    def __init__(self, src: str | os.PathLike | bytes, start: float, stop: float):
        if not isinstance(src, (str, os.PathLike, bytes)):
            raise TypeError("a slice reopens its source for every read: give a path or bytes")
        if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
            raise ValueError(f"slice ({start}, {stop}) out of range: need finite 0 <= start < stop")
        self.src, self.start, self.stop = src, start, stop
        self._plan = self._size = self._duration = None  # numbers and chapter marks: never headers or media

    @property
    def size(self) -> int:
        """Bytes it will yield."""
        self._measure()
        return self._size

    @property
    def duration(self) -> float:
        """Seconds of video, as its MP4 header says."""
        self._measure()
        return self._duration

    def _measure(self) -> None:
        if self._size is None:
            chunks = iter(self)
            next(chunks)
            chunks.close()

    def _plan_packets(self, video: Video, begin: int, end: int, seek: int | None = None) -> tuple:
        """(packets to copy, the keyframe they start at, the first frame's pts, chapter marks, room for the moov)."""
        walk = [(p.pts, p.is_keyframe, p.dts) for p in _packets(video, begin if seek is None else seek, end)]
        inside = [number for number, (pts, *_) in enumerate(walk) if begin <= pts < end]
        if not inside:
            raise ValueError(f"slice ({self.start}, {self.stop}) has no frames")
        shown = sorted(walk[number][0] for number in inside)
        # decoding starts at the last keyframe not after the first frame: a seek can land earlier, a whole chapter
        # earlier when the slice starts in a gap. It ends with the last frame before `stop`: with B-frames, what
        # decodes after that belongs to later frames.
        lead = max((number for number, (pts, key, _) in enumerate(walk) if key and pts <= shown[0]), default=None)
        if lead is None and seek is None:
            # an open GOP: MP4 and FLV seek by decode time, to the keyframe decoded before these frames but shown
            # after them. They need the keyframe before it.
            return self._plan_packets(video, begin, end, walk[0][2] - 1)
        if lead is None:  # MPEG-TS seeks to a packet, not a keyframe: nothing here could be decoded from its start
            raise ValueError(f"slice ({self.start}, {self.stop}) has no keyframe to start from")
        clock, marks = video.stream.time_base, []
        for chapter in video.chapters:
            rate = video.rate
            # counted from the packets, with the reader's half frame of slack, so marks and frames cannot disagree
            edges = ((edge - Fraction(1, 2) / rate) / clock for edge in (chapter.start, chapter.end))
            low, high = (bisect_left(shown, edge) for edge in edges)
            if low < high:
                first = round((shown[low] * clock - chapter.start + chapter.source) * rate)
                start = round((shown[low] - shown[0]) * clock * 1000)
                # MP4 keeps only a chapter's title, and ends each chapter where the next one starts
                title = json.dumps({"start_frame": first, "end_frame": first + high - low, "rate": str(rate)})
                marks.append({"id": len(marks), "start": start, "end": start + round((high - low) * 1000 / rate),
                              "time_base": Fraction(1, 1000), "metadata": {"title": title}})
        samples = inside[-1] + 1 - lead
        # A sample takes up to ~20 bytes of the moov (~13 in H.264), a chapter mark its title and 21 more (under 128
        # for any frame number), the rest ~1400 and the codec's header (2.4 KB from x265): a comfortable ceiling.
        room = 4096 + 32 * samples + 128 * len(marks)
        return samples, walk[lead][::2], shown[0], marks, room

    def _mux(self, sink: _Sink, video: Video, begin: int, end: int) -> Iterator[bytes]:
        samples, keyframe, first, marks, room = self._plan
        # moov first, so a reader needs no seeking: libav reserves the room and fills it in on close. The source's
        # clock, not the finer one the muxer would pick: its tick says how far the times were rounded.
        options = {"moov_size": str(room), "video_track_timescale": str(video.stream.time_base.denominator)}
        with av.open(sink, mode="w", format="mp4", options=options) as output:
            output.set_chapters(marks)
            # opaque: packets are only copied, so the decoder's own codec will do. PyAV otherwise looks for an encoder
            # of the decoder's name, and AV1's decoder (libdav1d) has none.
            track = output.add_stream_from_template(video.stream, opaque=True)
            if video.stream.codec_context.codec_tag == "hvc1":  # the muxer would write hev1, which Apple refuses
                track.codec_context.codec_tag = "hvc1"
            for packet in islice(_packets(video, begin, end, keyframe), samples):
                if samples == self._plan[0] and (header := _keyframe_header(video.stream, packet)):
                    # the stream's header is its first chapter's: Apple's players refuse a clip that contradicts it
                    context = track.codec_context
                    context.extradata, context.width, context.height = header
                samples -= 1
                packet.pts, packet.dts, packet.stream = packet.pts - first, packet.dts - first, track
                output.mux(packet)
                yield from sink.drain(64 * 1024)
        yield from sink.drain()
        if samples or sink.size != (self._size or sink.size):
            raise OSError("Video source changed between slice reads")

    def __iter__(self) -> Iterator[bytes]:
        with Video(self.src) as video:
            stream = video.stream
            base = 0 if video.chapters else stream.start_time or 0  # chapters sit on the source's clock
            rate = stream.average_rate  # the header's will do: it only keeps the slack under half a frame
            slack = min(SLACK, 0.5 / float(rate * stream.time_base)) if rate else SLACK
            begin, end = (math.ceil(t / stream.time_base - slack) + base for t in (self.start, self.stop))
            self._plan = self._plan or self._plan_packets(video, begin, end)
            # the reserved moov, plus room for ftyp, the mdat header and its size, which the muxer patches on close
            head = _Sink(self._plan[4] + 1024, queue=False)
            for _ in self._mux(head, video, begin, end):
                pass
            self._size, header = head.size, bytes(head.head[:head.size])
            with Video(header) as clip:
                self._duration = clip.duration
            yield header
            del header
            if head.size > len(head.head):
                yield from self._mux(_Sink(len(head.head), queue=True), video, begin, end)
