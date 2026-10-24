# Simple Video Utils

Lightweight utilities for extracting frames and metadata from videos. Built for sign language processing workflows.

![Python](https://img.shields.io/badge/python-3.12+-blue)
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
```

### Resize While Reading

```python
from av.video.reformatter import Interpolation
from simple_video_utils.frames import FrameSize, read_frames_batched

# Stretch to (width, height), resizing native YUV planes before RGB conversion.
frames = read_frames_batched("video.mp4", size=FrameSize((256, 256)))

# Choose PyAV's interpolation, e.g. bicubic for enlarging frames.
frames = read_frames_batched("video.mp4", size=FrameSize((512, 512), method=Interpolation.BICUBIC))
```

`read_frames_exact`, `read_frames_batched` and `read_frames_from_stream` accept `size`. Resolution is in display orientation;
the default method is `Interpolation.AREA`. Without `size`, each frame keeps its original dimensions.
Supply `size` to batch across chapters with different dimensions. Native-plane resizing can produce different
RGB pixels from resizing after RGB conversion, since chroma is resized at the source pixel format's resolution.

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

Pass `return_frame_metadata=True` for native timing and decoded indices:

```python
from simple_video_utils.frames import FrameSize, read_frames_from_stream

with open("video.mp4", "rb") as stream:
    meta, frames = read_frames_from_stream(
        stream, fps=15, size=FrameSize((256, 256)), return_frame_metadata=True,
    )
    for frame, info in frames:
        process(frame, info.index, info.time)
```

`FrameMetadata` contains `index`, `pts`, and `time_base`; its `time` property is exact
seconds as a `Fraction`, or `None` when timing is unavailable. Indices count decoded
frames from the start of the read, before skipping or FPS selection. Metadata describes
the original frame even when pixels are resized or rotated. The default still yields
arrays only. An already-open PyAV input container can also be passed; the streaming
reader preserves its position and leaves it open, without seeking for FPS recovery.

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

### Chapters: Crops and Trims in One File

```python
import json

from simple_video_utils.chapters import ChapterClip, Timeline, list_chapters, write_chapters
from simple_video_utils.frames import read_frames_exact

write_chapters("video.mp4", "chapters.mkv", [
    ChapterClip(30, 60, (100, 50, 500, 354), payload={"signer": "Alice"}),
    ChapterClip(90, 120, (200, 80, 440, 400)),
    ChapterClip(150, 180, (-20, 0, 300, 250), resolution=(256, 192)),
])

for chapter in list_chapters("chapters.mkv"):
    start_seconds = chapter["start"] * chapter["time_base"]
    tags = {key.lower(): value for key, value in chapter["metadata"].items()}
    source_clip = json.loads(tags["payload"])

for frame in read_frames_exact("chapters.mkv"):
    process(frame)  # each chapter can have a different frame shape

timeline = Timeline("chapters.mkv")
timeline.to_file(3.0)     # a source second on the file; inside a gap, the next chapter's start
timeline.to_source(1.2)   # a file second (or an array of them) on the source
```

Pass nonempty clips from a video with a nominal frame rate and an
H.264-compatible pixel format. Trims use source frame indices with an exclusive
end; boxes are integer `(left, top, right, bottom)` display coordinates.
Source frames are rotated upright before cropping and padding.
Box areas outside the source frame become black. `resolution=(width, height)`
resizes the crop to exactly that size; the default keeps the box size.
Clips play consecutively, and chapter payload tags retain their source specs.
Chapter boundaries use Matroska's millisecond clock. A source that ends before
a requested trim finishes raises `ValueError`.

Requires PyAV 19+. Output is a new, video-only Matroska file encoded as H.264
at `crf=23`, using the source frame rate and pixel format. Each chapter's output
width and height must be multiples of 16, otherwise `ValueError` is raised. The
destination must not already exist. Clips in source order share one decode; a clip
starting before the previous one's end decodes from the start again.

Aligned padding keeps the source chroma samples without resampling. Unaligned
crops and full-range sources retain high-depth processing for accurate offsets
and black borders. Scaling preserves the source color matrix, range and color tags.

`list_chapters` yields native PyAV chapter dictionaries and also accepts an open container
without rewinding or closing it. To process each chapter separately:

```python
from simple_video_utils.chapters import read_chapters
from simple_video_utils.frames import FrameSize, stack_frames

for chapter, source_start, frames in read_chapters(
    "chapters.mkv", fps=15, size=FrameSize((256, 256)),
):
    process_chapter(chapter, source_start, stack_frames(frames))
```

`read_chapters` reuses `read_frames_from_stream` with frame metadata, yielding lazy RGB frame iterators
for native half-open chapter ranges `[start, end)`. Consume each iterator before requesting
the next chapter. It supports ordinary chapter metadata in any container PyAV can read;
no signer payload is required. Incoming file objects work when the chapter table precedes
the media. Ranges must have positive duration and must not overlap.

FPS selection restarts per chapter, and `size` uses the same native-plane resizing as the
frame readers. `source_start` is a `Fraction` in seconds: the native chapter start unless
the writer's optional payload supplies `start_frame` (and optionally `rate`). Without
chapters it yields one `(None, 0, frames)` entry covering the whole video. No clips are
encoded or temporary files created.

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
