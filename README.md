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
print(f"{meta.width}x{meta.height} @ {meta.fps} fps")
# Output: VideoMetadata(width=1920, height=1080, fps=30.0, nb_frames=450, time_base='1/15360')
```

### Read Frames from File

```python
from simple_video_utils.frames import read_frames_exact

# Read specific frame range (inclusive)
frames = list(read_frames_exact("video.mp4", start_frame=0, end_frame=10))
# Returns 11 frames as numpy arrays (H, W, 3) in RGB format

# Read from frame to end of video
frames = list(read_frames_exact("video.mp4", start_frame=5, end_frame=None))
```

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

