# simple-video-utils

Read, slice and crop videos for sign language processing. One class over [PyAV](https://pyav.org), under 800 lines.

```bash
pip install simple-video-utils  # Python 3.14+
```

```python
from simple_video_utils import Video, video_metadata

video_metadata("sign.mp4")
# VideoMetadata(width=320, height=240, fps=30.0, nb_frames=130, time_base='1/15360', duration=4.33, rotation=0)

with Video("sign.mp4") as video:                      # a path, a URL, bytes, or a binary stream
    video.meta                                        # the same metadata, for any source
    frames, indices = video.read()                    # (N, H, W, 3) uint8 RGB, and each frame's index
    window = video.read(30, 60).frames                # frames 30..59
    second = video.read(1.0, 2.0, seconds=True)       # the frames timed in [1 s, 2 s)
    small = video.read(fps=15, keep_last=True, size=(256, 256))
    for frame, index in video.frames(fps=15):         # the same arguments, decoded as you go
        ...
```

## Metadata

`width` and `height` are as displayed, rotation applied. `nb_frames` is checked rather than trusted: the header's count
is compared with duration x fps and the frames are decoded when they disagree; files with no count (Matroska, WebM)
read their last packet instead. `fps` is measured from the frames when the header has none (browser recordings). A
stream that cannot seek gets the header's values only, and `fps=0.0` when there is no rate.

`video_metadata(path)` caches its last 8 paths. A file that cannot be opened or decoded raises `RuntimeError`.

## Frames

`Video.frames()` yields `(array, index)` pairs lazily. `Video.read()` returns the same frames as one batch,
`Frames(frames, indices)`: `torch.from_numpy(batch.frames)` wraps it without a copy. Both take the same arguments:

| Argument | Meaning |
|---|---|
| `start`, `stop` | Frame indices, `stop` excluded. With `seconds=True`, times: the frames timestamped in `[start, stop)`. |
| `fps` | Drop frames down to about this rate. Never repeats a frame; always keeps the first. |
| `keep_last` | With `fps`, also keep the last frame of the range, which `fps` drops when it falls between steps. |
| `size` | `(width, height)` as displayed. Resized before the RGB conversion. |

Reads three seconds or more into a file seek first. Frames always come out upright. A stream that cannot seek is read
once: a second read raises `ValueError`.

`Video(src, thread_type="NONE")` decodes without threads: use it in processes that fork (DataLoader workers).
`Video(stream, buffer_size=4096)` yields frames sooner from a live stream. `video.container` and `video.stream` are the
PyAV objects, for anything not covered here; call `video.rewind()` before decoding from them yourself.

## Chapters

A chaptered file holds crops and trims of a source video. Each frame keeps its source time, so the seconds between
chapters hold no frames, and frame indices are the source's.

```python
from simple_video_utils import ChapterClip, write_chapters

written = write_chapters("source.mp4", "cropped.mkv", [
    ChapterClip(start_frame=30, end_frame=90, box=(64, 32, 320, 288), payload={"signer": 1}),
    ChapterClip(start_frame=200, end_frame=260, box=(-16, 0, 240, 256), resolution=(128, 128)),
])

with Video("cropped.mkv") as video:
    video.chapters                                    # [Chapter(id, start, end, source, payload), ...]
    video.read(30, 90)                                # the first chapter, by source frame index
    for chapter, frames in video.chapter_frames(fps=15, size=(256, 256)):
        ...                                           # one decode, so it works on a stream
```

`box` is `(left, top, right, bottom)` in displayed pixels and may reach past the frame, which pads with black. Sizes
must be multiples of 16. A clip running past the source's last frame is cut there; `write_chapters` returns what it
wrote. Chapters of different sizes need `size=` to be read as one batch. `fps` treats each chapter as its own range in
every read: it starts over on each chapter's first frame, and `keep_last` keeps each chapter's last. A video without
chapters yields `(None, frames)` once from `chapter_frames`.

Only chapters this library wrote count. Files written before 0.10, which played their chapters back to back, raise
`PackedChaptersError` (a `ValueError`) on any read: write them again.

## Slices

```python
from simple_video_utils import Slice

clip = Slice("cropped.mkv", 12.0, 15.5)              # a path or bytes; nothing is read yet

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
Times are matched at the precision of the file's clock. Codecs MP4 cannot carry raise.

## From 0.x

| 0.x | 1.0 |
|---|---|
| `from simple_video_utils.metadata import video_metadata` | `from simple_video_utils import video_metadata` |
| `video_metadata_from_bytes(data)` | `Video(data).meta` |
| `open_video(p, thread_type=t)` + `video_metadata_from_container(v)` | `Video(p, thread_type=t)`, `v.meta` |
| `read_frames_exact(p, start_frame=a, end_frame=b)` | `Video(p).frames(a, b + 1)` |
| `read_frames_batched(p, start_time=a, end_time=b)` | `Video(p).read(a, b, seconds=True).frames` |
| `frames, idx = read_frames_batched(p, return_indices=True)` | `frames, idx = Video(p).read()` |
| `for frame in read_frames_exact(p)`, `size=FrameSize((w, h))` | `for frame, index in Video(p).frames()`, `size=(w, h)` |
| `read_frames_from_stream(f, fps=15, buffer_size=n)` | `v = Video(f, buffer_size=n)`, then `v.meta`, `v.frames(fps=15)` (pairs) |
| `read_chapters(src, fps=, size=)` yielding `(dict, start, frames)` | `Video(src).chapter_frames(fps=, size=)` yielding `(Chapter, frames)` |
| `list_chapters(p)` | `Video(p).chapters` |
| `keyframe_indices(v)`, `count_frames(v)` | `v.keyframes()`, `sum(1 for _ in v.container.decode(v.stream))` |
| `slice_video(p, [(a, b)], chapters=True)`, `Video.from_slice(p, a, b)` | `Slice(p, a, b)` |
| `write_chapters(..., trim_to_source=True, thread_type=t)` | `write_chapters(...)`: it always trims; no `thread_type` |
| `Timeline` | gone: a gapped file is on source time already |
| `join_videos`, `slice_video_stream`, `slice_video(size=)` | gone |
