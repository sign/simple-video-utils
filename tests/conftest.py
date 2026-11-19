import io
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils import ChapterClip, write_chapters

ASSETS = Path(__file__).parent / "assets"
EXAMPLE = str(ASSETS / "example.mp4")  # 130 frames, 320x240, 30 fps, B-frames, one keyframe
# three chapters of EXAMPLE with gaps between them and different sizes
CLIPS = [ChapterClip(0, 20, (0, 0, 160, 128)), ChapterClip(45, 80, (64, 32, 320, 240), payload={"who": "signer"}),
         ChapterClip(100, 128, (0, 0, 320, 240))]
HELD = [*range(0, 20), *range(45, 80), *range(100, 128)]


class Pipe(io.BytesIO):
    """A stream that cannot seek."""

    def seekable(self):
        return False


def make_video(path, frames=60, fps=30, gop=12, bframes=0, noise=False, times=None, pix_fmt="yuv420p", x264="") -> str:
    """Frame i shows i as two grey levels, which `numbers` reads back; `noise` makes the packets large instead.

    `times` places each frame, in 1/fps ticks, for a video whose rate changes. `x264` adds encoder parameters.
    """
    random = np.random.default_rng(0)
    with av.open(str(path), mode="w") as output:
        fixed = f"keyint={gop}:min-keyint={gop}:scenecut=0{x264}"  # a keyframe exactly every `gop` frames
        stream = output.add_stream("h264", rate=fps, options={"crf": "10", "x264-params": fixed})
        stream.width, stream.height, stream.pix_fmt = 64, 48, pix_fmt
        stream.codec_context.max_b_frames = bframes
        for i in range(frames):
            image = np.repeat([[i % 16 * 16, i // 16 * 16]], 32, axis=1).repeat(48, axis=0)[..., None].repeat(3, axis=2)
            if noise:
                image = random.integers(0, 256, (48, 64, 3))
            frame = av.VideoFrame.from_ndarray(image.astype(np.uint8), format="rgb24")
            frame.pts, frame.time_base = times[i] if times else i, Fraction(1, fps)
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    return str(path)


def numbers(frames) -> list[int]:
    return [round(float(f[:, :32].mean()) / 16) + 16 * round(float(f[:, 32:].mean()) / 16) for f in frames]


def windows(video, indices, fps=30) -> list[list[int]]:
    """The indices a timed read of each frame's own window, [i / fps, (i + 1) / fps), returns."""
    def timed(i):
        return video.frames(i / fps, (i + 1) / fps, seconds=True, size=(16, 16), indices=True)

    return [[index for _, index in timed(i)] for i in indices]


@pytest.fixture(scope="session")
def gapped(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("gapped") / "gapped.mkv"
    write_chapters(EXAMPLE, out, CLIPS, preset="ultrafast")
    return str(out)
