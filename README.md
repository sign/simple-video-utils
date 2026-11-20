# simple-video-utils

Read, write, slice and crop videos for sign language processing. A thin layer over [PyAV](https://pyav.org), about
1,100 lines, so that nothing using it has to import PyAV itself.

```bash
pip install simple-video-utils  # Python 3.12+
```

```python
from simple_video_utils import Video, video_metadata

video_metadata("sign.mp4")
# VideoMetadata(width=320, height=240, fps=30.0, nb_frames=130, time_base='1/15360', duration=4.333333333333333, rotation=0)

with Video("sign.mp4") as video:                      # a path, a URL, bytes, or a binary stream
    video.meta                                        # the same metadata, for any source
    frames, indices = video.read()                    # (N, H, W, 3) uint8 RGB, and each frame's index
    window = video.read(30, 60).frames                # frames 30..59
    second = video.read(1.0, 2.0, seconds=True)       # the frames timed in [1 s, 2 s)
    small = video.read(fps=15, size=(256, 256))
    for frame in video.frames(fps=15):                # the same arguments, decoded as you go
        ...
```

## Metadata

`width` and `height` are as displayed, rotation applied. `nb_frames` is checked rather than trusted: a file with
chapters counts what its chapters hold. Otherwise the header's count, less the samples an edit list hides (a slice's
lead-in, a recording trimmed in place), is compared with duration x fps and the frames are decoded when they disagree;
files with no count (Matroska, WebM) read their last packet instead, which takes the header's frame rate to hold for
the whole file. `fps` is measured from the frames when the header has none (browser recordings), and `video.rate` is
the same rate as an exact `Fraction`. A stream that cannot seek gets the header's values only, and `fps=0.0` when there
is no rate.

`video_metadata(path)` caches its last 8 paths.

`video.codec` (`"h264"`), `video.pix_fmt`, `video.sample_aspect_ratio`, `video.tags` (the container's metadata) and
`video.timestamps()` (each frame's time in seconds, read from the packets) cover what the header and the packets say.

## Errors

Whatever PyAV refuses, opening, decoding or writing, raises `VideoError`, a `RuntimeError`. When the data itself is
not a video (garbage, a file cut off inside its header, no video stream, a codec with no decoder) it is an
`InvalidVideoError`, a `VideoError`. A file cut off between frames is a shorter video, and reads as one; cut off
inside a frame, it raises with `thread_type="NONE"` or `"SLICE"` and reads as a shorter video with the default,
because FFmpeg's frame threads drop that error. Arguments that
make no sense raise `ValueError` or `TypeError`. A destination that cannot be written, for want of a directory or of
space, raises the `OSError` it is.

## Frames

`Video.frames()` yields RGB arrays lazily. `Video.read()` returns the same frames as one batch,
`Frames(frames, indices)`: `torch.from_numpy(batch.frames)` wraps it without a copy. Both take the same arguments:

| Argument | Meaning |
|---|---|
| `start`, `stop` | Frame indices, `stop` excluded. With `seconds=True`, times: the frames timestamped in `[start, stop)`. |
| `fps` | Drop frames down to about this rate. Never repeats a frame; keeps the first and the last of the range. |
| `keep_last` | `False` lets `fps` drop the last frame when it falls between steps. |
| `size` | `(width, height)` as displayed. Resized before the RGB conversion. |
| `format` | The arrays' pixel format: `"rgb24"` by default, `"gray"` for `(H, W)`, `"rgba"` for `(H, W, 4)`. |
| `indices` | `frames()` and `chapter_frames()` only: yield `(array, index)` pairs. `read()` always returns the indices. |

Reads three seconds or more into an MP4, MOV, Matroska or WebM file seek first, when its header says the frames keep
one rate; any other read is counted from the first frame. An MP4's header shows a skipped frame and Matroska's does
not, so a Matroska file with uneven timing is numbered by its header rate after a seek. Frames come out upright, by the
rotation in the header. A stream that cannot seek is read once: a second read raises `ValueError`. A `Video` does one
read at a time: `keyframes()`, another read, `close()`, or a `meta` that has to read the file (Matroska, WebM) in the
middle of a `frames()` loop makes that loop raise `RuntimeError`, so ask for `meta` before the loop.

`Video(src, format="mpegts")` names the container when the data cannot, as for a stream joined midway, and
`container_options={"probesize": "32"}` passes the demuxer its options.
`Video(src, thread_type="NONE")` decodes without threads. Open each `Video` in the process that reads it (in a
DataLoader worker, not before the fork): workers that inherit one share its file position and fail to decode.
`Video(stream, buffer_size=4096)` yields frames sooner from a live stream. `video.container` and `video.stream` are the
PyAV objects, for anything not covered here; call `video.rewind()` before decoding from them yourself. They close with
the `Video`, when it is closed or dropped.

## Chapters

A chaptered file holds crops and trims of a source video. Frame `i` sits at `i / rate`, its source time when the source
keeps one rate, so the seconds between chapters hold no frames, and frame indices are the source's.

```python
from simple_video_utils import ChapterClip, write_chapters

written = write_chapters("source.mp4", "cropped.mkv", [
    ChapterClip(start_frame=30, end_frame=90, box=(64, 32, 320, 288), payload={"signer": 1}),
    ChapterClip(start_frame=200, end_frame=260, box=(-16, 0, 240, 256), resolution=(128, 128)),
])

with Video("cropped.mkv") as video:
    video.chapters                                    # [Chapter(id, start, end, source, frames, payload), ...]
    video.read(30, 90)                                # the first chapter, by source frame index
    for chapter, frames in video.chapter_frames(fps=15, size=(256, 256)):
        ...                                           # one decode, so it works on a stream
```

`box` is `(left, top, right, bottom)` in displayed pixels and may reach past the frame, which pads with black. Sizes
must be multiples of 16. A clip running past the source's last frame is cut there; `write_chapters` returns what it
wrote. Chapters of different sizes need `size=` to be read as one batch. `fps` treats each chapter as its own range in
every read: it starts over on each chapter's first frame and keeps each chapter's last. `chapter.frames` is the range
of source frames a chapter holds. A chapter without frames is left out of `chapter_frames`, and a video without
chapters yields `(None, frames)` once.

Only chapters this library wrote count. Chapters may touch: `[0, 60)` and `[60, 120)` is a valid file. What is
refused is a chapter stored at another time than its source's, as 0.x wrote them before 0.10 (source frames 90-119
stored at 1-2 s): such a file raises `PackedChaptersError` (a `ValueError`) on any read. Write it again.

## Slices

```python
from simple_video_utils import VideoSlice

clip = VideoSlice("cropped.mkv", 2.0, 8.0)                 # a path or bytes; nothing is read yet

clip.size, clip.duration                              # measured without keeping the clip
for chunk in clip:                                    # MP4 bytes, about 64 KiB at a time; iterate again to replay
    ...
data = b"".join(clip)
```

A slice copies packets, so it is lossless and fast, and never holds the clip in memory. The MP4 puts `moov` first, so
it can be read from a pipe. It starts on its first frame and keeps the frames from the keyframe before it, hidden
behind an edit list. With B-frames it also keeps the later frames its last ones are predicted from, and those stay
visible: a few frames at or past `stop`, not always consecutive. Gaps between chapters stay gaps, and the chapters it
spans come along, so reading a chaptered clip gives exactly the frames in range, with the source's frame indices.
Times are matched at the precision of the file's clock. Codecs MP4 cannot carry raise. In a file whose chapters differ
in size, the clip's header describes the chapter it starts in, so a slice within one chapter plays everywhere; one
that runs into a chapter of another size plays only in FFmpeg-based players.

## Writing

```python
from simple_video_utils import VideoWriter, write_video

write_video("out.mp4", frames, fps=30)                # H.264, yuv420p; frames are (H, W, 3) RGB uint8
data = write_video(None, frames, fps=30, options={"crf": "18"})   # the MP4's bytes

with VideoWriter(stream, 30, format="mpegts", options={"tune": "zerolatency"}) as writer:
    for frame in frames:                              # a writable stream gets bytes as frames arrive
        writer.write(frame)
```

Frames are RGB, RGBA `(H, W, 4)` or gray `(H, W)`, all of one shape. Both take the same settings:

| Setting | Meaning |
|---|---|
| `codec`, `pix_fmt` | The encoder and its pixel format: `"h264"` and `"yuv420p"` by default. |
| `options` | The encoder's options, as strings: `{"crf": "18", "preset": "fast", "threads": "1"}`. |
| `format`, `container_options` | The container, when the name does not say, and the muxer's options. |
| `filters` | An FFmpeg filter chain every frame passes before the encoder: `"colorkey=0x00ff00:0.3:0.05"`. |
| `tags` | The container's metadata, where it keeps custom keys (Matroska does, MP4 does not). |

`writer.write(frame, index)` places a frame at `index / fps`, leaving a gap. Mattes round-trip exactly with
`codec="ffv1", pix_fmt="gray"` and `read(format="gray")`; transparency is `codec="libvpx-vp9", pix_fmt="yuva420p"` in
a `.webm`. YUV output is tagged as what it is, sRGB pixels converted by BT.601 in limited range, so players show
the colours that went in. Nothing is written before the first frame.

## From 0.x

| 0.x | 1.0 |
|---|---|
| `from simple_video_utils.metadata import video_metadata` | `from simple_video_utils import video_metadata` |
| `video_metadata_from_bytes(data)` | `Video(data).meta` |
| `open_video(p, thread_type=t)` + `video_metadata_from_container(v)` | `Video(p, thread_type=t)`, `v.meta` |
| `read_frames_exact(p, start_frame=a, end_frame=b)` | `Video(p).frames(a, b + 1)` |
| `read_frames_batched(p, start_time=a, end_time=b)` | `Video(p).read(a, b, seconds=True).frames`: `b` is now excluded |
| `frames, idx = read_frames_batched(p, return_indices=True)` | `frames, idx = Video(p).read()` |
| `read_frames_exact(p, return_indices=True)`, `size=FrameSize((w, h))` | `Video(p).frames(indices=True)`, `size=(w, h)` |
| `read_frames_from_stream(f, fps=15, buffer_size=n)` | `v = Video(f, buffer_size=n)`, then `v.meta`, `v.frames(fps=15)` |
| `read_chapters(src, fps=, size=)` yielding `(dict, start, frames)` | `Video(src).chapter_frames(fps=, size=)` yielding `(Chapter, frames)` |
| `list_chapters(p)` | `Video(p).chapters`, this library's only; every marker: `Video(p).container.chapters()` |
| `keyframe_indices(v)`, `count_frames(v)` | `v.keyframes()`; `v.rewind()`, then `sum(1 for _ in v.container.decode(v.stream))` |
| `slice_video(p, [(a, b)], chapters=True)`, `Video.from_slice(p, a, b)` | `VideoSlice(p, a, b)` |
| `write_chapters(..., trim_to_source=True, thread_type=t)` | `write_chapters(...)`: it always trims; no `thread_type` |
| `Timeline` | gone: a gapped file is on source time already |
| `join_videos`, `slice_video_stream`, `slice_video(size=)` | gone |
