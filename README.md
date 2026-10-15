# Simple Video Utils

Lightweight utilities for extracting frames and metadata from videos. Built for sign language processing workflows.

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## Installation

```bash
pip install simple-video-utils
```

## Usage

### Extract Video Metadata

```python
from simple_video_utils.metadata import video_metadata

meta = video_metadata("video.mp4")
print(f"{meta.width}x{meta.height} @ {meta.fps} fps, {meta.duration}s")
# Output: VideoMetadata(width=1920, height=1080, fps=30.0, nb_frames=450, time_base='1/15360', duration=15.0)
```

### Keyframe Indices (GOP Structure)

```python
from simple_video_utils.metadata import keyframe_indices

keys = keyframe_indices("video.mp4")
# Presentation-order frame indices of the keyframes, e.g. [0, 250, 500]
# Demux-only (no decoding) — cheap even for long videos.
```

### Read Frames from File

```python
from simple_video_utils.frames import read_frames_exact

# Read specific frame range (inclusive)
frames = list(read_frames_exact("video.mp4", start_frame=0, end_frame=10))
# Returns 11 frames as numpy arrays (H, W, 3) in RGB format

# Read from frame to end of video
frames = list(read_frames_exact("video.mp4", start_frame=5, end_frame=None))

# Downsample to a target frame rate (drops frames uniformly, never duplicates)
frames = list(read_frames_exact("video.mp4", fps=15))

# Scale every frame to one (width, height), up or down, in display orientation
frames = list(read_frames_exact("video.mp4", size=(256, 256)))
```

`size` scales the planes of a 4:2:0 frame before converting it to RGB. Its colour then stays at 4:2:0 of the new
size, as an encode at that size would keep it, and the read takes a third of the CPU of scaling straight to RGB.
Frames of different sizes, such as a chapter file's, batch in `read_frames_batched` only once `size` scales them to
one size.

### Chapter Files

A chapter file keeps clips of a video as the chapters of one Matroska video track. Each clip is cut square around a
box, sits at its source time with gaps between clips, and has its own frame size. Its chapter carries a JSON
`payload` tag that says where the clip came from and how it was cut.

```python
from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters

# frames [start, end) of the source, each clip cut square around its box (left, top, right, bottom in pixels)
clips = [ChapterClip(30, 90, [412.3, 80.0, 770.9, 700.2], payload={"score": 0.93}), ChapterClip(150, 400, [...])]
payloads = write_chapters("video.mp4", "clips.mkv", clips, crf=23)

for chapter in list_chapters("clips.mkv"):
    print(chapter.start, chapter.end, chapter.payload)
# 1 3 {'start_frame': 30, 'end_frame': 90, 'box': [412.3, 80.0, 770.9, 700.2], 'crop': [400, 80, 784, 704],
#      'pad': [128, 0, 112, 0], 'side': 624, 'scale': 1.0, 'resolution': {'width': 624, 'height': 624}, 'score': 0.93}
```

- `crop` is the box moved out to the source's next 16-pixel macroblock on each side.
- `pad` holds the black bars that square the crop, left, top, right and bottom. The left and top bars are whole
  macroblocks.
- `side` is the square's side, which is whole macroblocks up to `max_side` (768 by default).
- A larger square is scaled down to `max_side`, and `scale` and `resolution` give the result.
- `write_chapters` makes one pass over the source and encodes each chapter with its own x264 encoder.
- `on_frame(clip_index, frame)` sees each source frame inside a clip before it is cut, during that same pass.

Chapter sides are multiples of 16, the H.264 macroblock size. FFmpeg's decoder can keep the previous chapter's size
when two sizes cover the same macroblocks and differ only in their crop, as 470 and 472 do. It then returns the
top-left of the frame and drops its right and bottom edge. Every FFmpeg-based reader reads such a file this way.

The frame readers read a chapter file on the source timeline:
- Frame indices and ranges follow the chapters' source times, not the frames' positions in the file.
- `return_chapters=True` adds each frame's payload: `{}` for a chapter without one, or `None` outside every chapter.
- Only chapters carrying `payload` tags make a chapter file. Any other video with chapters, such as a film, reads by
  position as any video does.

A range read seeks to the keyframe at or before its first frame. It skips decoding the non-reference frames before
that frame, since no other frame refers to them. For a 15-frame clip:
- `thread_type="AUTO"` reads fastest;
- `"SLICE"` takes a quarter less CPU at nearly the same speed;
- `"NONE"` takes the least CPU, for loaders whose worker processes already fill the cores.

```python
from simple_video_utils.frames import read_frames_exact

for frame, index, payload in read_frames_exact("clips.mkv", return_indices=True, return_chapters=True):
    ...  # e.g. 624x624 frames 30 to 89, then the next chapter's
```

### One Open, Many Reads

```python
from simple_video_utils.metadata import open_video, video_metadata_from_container, keyframe_indices
from simple_video_utils.frames import read_frames_exact

# Metadata + windowed frame reads from a single container open —
# e.g. sampling indices from metadata, then decoding just that window.
with open_video("video.mp4") as video:
    meta = video_metadata_from_container(video)
    keys = keyframe_indices(video)
    frames = list(read_frames_exact(video, start_frame=10, end_frame=20))
```

Every helper rewinds the container before reading, so call order doesn't
matter. Requires seekable input; consume one frame read at a time.

`open_video` sets the container's decode `thread_type` up front (default
`"AUTO"`) — PyAV forbids changing it once a stream's codec is open, which
happens on first metadata probe or frame read. Pass `thread_type="NONE"` if
you fork worker processes around decoding (e.g. a DataLoader): an inherited
AUTO-threaded decoder can deadlock post-fork.

### Read Frames from Stream

```python
from simple_video_utils.frames import read_frames_from_stream

# Useful for uploaded files or in-memory video data
with open("video.mp4", "rb") as f:
    meta, frames_gen = read_frames_from_stream(f)
    for frame in frames_gen:
        # Process each frame (numpy array)
        pass
```

### Slice into Clips

```python
from simple_video_utils.slicing import slice_video

# One MP4 (bytes) per (start, end) second range
clips = slice_video("video.mp4", [(0.0, 1.5), (2.0, 3.2)])

# Center-crop to a square and resize to 256x256 (e.g. for model input)
clips = slice_video("video.mp4", [(0.0, 1.5)], size=256)
```

Clips are streamable MP4 (`moov` first), so a consumer can demux them from a
pipe or socket without seeking.

Split an incoming video stream into packet-copied clips as it arrives —
each clip is yielded as soon as its window has been read, without re-encoding.
Each carries its `start` on the source timeline, which counting the clips does
not give you: an empty window yields nothing while the timeline moves past it.

```python
from simple_video_utils.slicing import slice_video_stream

with open("video.mp4", "rb") as stream:
    for start, data in slice_video_stream(stream, duration=0.5):
        process(start, data)
```

Join clips in order. Overlapping slices from the same encoded stream are
de-duplicated and remuxed losslessly; anything else is decoded and encoded
once as H.264 MP4:

```python
from simple_video_utils.joining import join_videos

video = join_videos(clips)
```

### Remote Videos

```python
from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact

# Works with remote URLs
url = "https://example.com/video.mp4"
meta = video_metadata(url)
frames = list(read_frames_exact(url, 0, 5))
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
```
