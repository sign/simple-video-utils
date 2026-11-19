"""Writing: frames in, a video that `Video` reads back."""

import io
from fractions import Fraction

import numpy as np
import pytest
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc
from conftest import make_video, numbers

from simple_video_utils import Video, VideoError, VideoWriter, write_video

FRAMES = np.random.default_rng(0).integers(0, 256, (5, 48, 64, 4), dtype=np.uint8)


def shows(number: int) -> np.ndarray:
    """A frame that `numbers` reads back as `number`."""
    return np.repeat([[number % 16 * 16, number // 16 * 16]], 32, axis=1).repeat(48, axis=0).astype(np.uint8)


@pytest.mark.parametrize(("frames", "pix_fmt", "format"), [(FRAMES[..., 0], "gray", "gray"), (FRAMES, "bgra", "rgba")])
def test_lossless_frames_and_tags_read_back_exactly(tmp_path, frames, pix_fmt, format):
    path = tmp_path / "matte.mkv"
    assert write_video(path, frames, 25, codec="ffv1", pix_fmt=pix_fmt, tags={"MATTE_SIZE": "48x64"}) is None
    with Video(path) as video:
        assert (video.read(format=format).frames == frames).all()
        assert (video.codec, video.pix_fmt, video.fps, video.tags["MATTE_SIZE"]) == ("ffv1", pix_fmt, 25, "48x64")
        assert video.sample_aspect_ratio == 1


def test_default_is_h264_mp4_at_the_exact_rate():
    with Video(write_video(None, (shows(i) for i in range(20)), 30000 / 1001)) as video:
        assert (video.codec, video.pix_fmt, video.rate) == ("h264", "yuv420p", Fraction(30000, 1001))
        assert numbers(video.frames()) == list(range(20))
        # sRGB pixels, converted by BT.601: untagged, Apple's decoder shows skin (224, 172, 105) as (226, 181, 117)
        colour = video.stream.codec_context
        assert (colour.colorspace, colour.color_range) == (Colorspace.ITU601, ColorRange.MPEG)
        assert (colour.color_primaries, colour.color_trc) == (ColorPrimaries.BT709, ColorTrc.IEC61966_2_1)


def test_index_places_a_frame_and_timestamps_reports_it(tmp_path):
    with Video(make_video(tmp_path / "gap.mp4", frames=4, times=[0, 1, 5, 6])) as video:
        assert video.timestamps() == pytest.approx([0, 1 / 30, 5 / 30, 6 / 30])


def test_filters_run_before_the_encoder():
    with Video(write_video(None, [shows(1)], 30, filters="hflip,vflip")) as video:
        assert numbers(video.frames()) == [16]  # the two halves changed sides


def test_a_stream_receives_bytes_while_frames_are_written():
    stream, sizes = io.BytesIO(), []
    with VideoWriter(stream, 30, format="mpegts", options={"tune": "zerolatency"}) as writer:
        for i in range(5):
            writer.write(shows(i))
            sizes.append(stream.tell())
    assert sizes[0] < sizes[-1], "bytes arrive with the frames, not at the end"
    with Video(stream.getvalue(), format="mpegts") as video:
        assert numbers(video.frames()) == list(range(5))


def test_what_cannot_be_written_raises():
    with pytest.raises(ValueError, match="different shapes"):
        write_video(None, [shows(0), shows(0)[:32]], 30)
    with pytest.raises(ValueError, match="nope"):
        write_video(None, [shows(0)], 30, codec="nope")
    # leaving the block after the encoder refused to open must not flush it: PyAV crashes the process
    with pytest.raises(VideoError, match="Failed to write video"):
        write_video(None, [shows(0)[:47]], 30)  # 4:2:0 needs even sides
