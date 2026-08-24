# Simple Video Utils

Lightweight utilities for extracting frames and metadata from videos. Built for sign language processing workflows.

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## Goal

Provide simple, efficient tools for video processing in sign language research and applications. 
Uses PyAV for fast frame extraction with support for multiple formats (MP4, WebM) and remote URLs.

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

Split a still-arriving video into packet-copied MP4 clips:

```python
from simple_video_utils.streaming import slice_video_stream

async for clip in slice_video_stream(request.stream(), duration=0.5):
    await process(clip)
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
