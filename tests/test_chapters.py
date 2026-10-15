import json
from dataclasses import asdict
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters
from simple_video_utils.frames import read_frames_exact

SOURCE = Path(__file__).parent / "assets" / "example.mp4"


def test_crop_trim_and_read(tmp_path):
    clips = [
        ChapterClip(30, 35, (10, 6, 74, 38), {"signer": "Alice"}),
        ChapterClip(60, 64, (40, 20, 72, 84)),
    ]
    output = tmp_path / "chapters.mkv"
    write_chapters(SOURCE, output, clips, crf=0)
    original = list(read_frames_exact(str(SOURCE)))
    frames = list(read_frames_exact(str(output)))
    chapters = list(list_chapters(output))
    assert len(frames) == 9
    assert len(chapters) == 2
    offset = 0
    for chapter, clip in zip(chapters, clips, strict=True):
        count = clip.end_frame - clip.start_frame
        assert chapter["start"] * chapter["time_base"] == pytest.approx(offset / 30, abs=0.001)
        assert chapter["end"] * chapter["time_base"] == pytest.approx((offset + count) / 30, abs=0.001)
        tags = {key.lower(): value for key, value in chapter["metadata"].items()}
        assert json.loads(tags["payload"]) == json.loads(json.dumps(asdict(clip)))
        left, top, right, bottom = clip.box
        for index in range(count):
            expected = original[clip.start_frame + index][top:bottom, left:right]
            np.testing.assert_allclose(frames[offset + index], expected, atol=1)
        offset += count
    selected = list(read_frames_exact(str(output), start_frame=3, end_frame=6))
    for frame, expected in zip(selected, frames[3:7], strict=True):
        np.testing.assert_array_equal(frame, expected)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        avcc = bytes(stream.codec_context.extradata)
        packet = bytes(next(container.demux(stream)))
    # The header SPS (avcC) must describe the stream, e.g. lossless crf=0 is High 4:4:4.
    sps_size = int.from_bytes(avcc[6:8])
    assert avcc[8 : 8 + sps_size] == packet[4 : 4 + int.from_bytes(packet[:4])]


@pytest.mark.parametrize("box", [(0, 0, 62, 32), (0, 0, 64, 34)])
def test_h264_requires_whole_macroblocks(tmp_path, box):
    with pytest.raises(ValueError, match="multiples of 16"):
        write_chapters(SOURCE, tmp_path / "chapters.mkv", [ChapterClip(30, 35, box)])


@pytest.mark.parametrize("box", [(-5, -3, 27, 29), (45, 29, 77, 61)])
@pytest.mark.parametrize("resolution", [None, (64, 32)])
def test_padding_and_resolution(tmp_path, box, resolution):
    source = tmp_path / "source.mkv"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 192, dtype=np.uint8), format="rgb24")
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "padded.mkv"
    clip = ChapterClip(0, 1, box, resolution=resolution)
    write_chapters(source, output, [clip], crf=0)
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0)).to_ndarray(format="gray")
    width, height = resolution or (32, 32)
    assert actual.shape == (height, width)
    assert actual[height // 2, width // 2] == pytest.approx(192, abs=1)
    if box[0] < 0:
        assert actual[0, 0] <= 1
    else:
        assert actual[-1, -1] <= 1
    if resolution is None:
        y, x = np.indices((height, width))
        valid = (x + box[0] >= 0) & (x + box[0] < 64) & (y + box[1] >= 0) & (y + box[1] < 48)
        np.testing.assert_allclose(actual, np.where(valid, 192, 0), atol=1)
    tags = {key.lower(): value for key, value in next(list_chapters(output))["metadata"].items()}
    assert json.loads(tags["payload"])["resolution"] == (list(resolution) if resolution else None)


def test_resize_controls_h264_alignment(tmp_path):
    output = tmp_path / "resized.mkv"
    write_chapters(SOURCE, output, [ChapterClip(30, 31, (10, 6, 73, 41), resolution=(64, 32))])
    assert next(read_frames_exact(str(output))).shape == (32, 64, 3)
    with pytest.raises(ValueError, match="multiples of 16"):
        write_chapters(SOURCE, tmp_path / "invalid.mkv", [ChapterClip(30, 31, (0, 0, 64, 32), resolution=(62, 32))])
