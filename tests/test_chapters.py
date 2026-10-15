import json
from dataclasses import asdict
from fractions import Fraction

import av
import numpy as np
import pytest

from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters
from simple_video_utils.frames import read_frames_exact


@pytest.fixture(params=[Fraction(25), Fraction(30000, 1001)])
def source(tmp_path, request):
    path = tmp_path / "source.mkv"
    y, x = np.indices((96, 480))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=request.param)
        stream.width, stream.height, stream.pix_fmt = 480, 96, "yuv444p"
        for index in range(12):
            grey = (x // 8 * 2 + y // 8 * 3 + index * 7).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.repeat(grey[:, :, None], 3, axis=2), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path, request.param


@pytest.mark.parametrize("crf", [0, 23])
def test_crops_trims_chapters_and_existing_reader(source, tmp_path, crf):
    path, rate = source
    output = tmp_path / "chapters.mkv"
    # Rectangular crops at arbitrary origins; dimensions lie on whole macroblocks.
    clips = [
        ChapterClip(2, 5, (8, 7, 472, 39), {"signer": "Alice"}),
        ChapterClip(7, 11, (4, 5, 452, 53)),
        ChapterClip(0, 2, (17, 19, 49, 83)),
    ]
    write_chapters(path, output, clips, crf=crf)
    chapters = list(list_chapters(output))
    assert len(chapters) == len(clips)
    offset = 0
    with av.open(str(path)) as container:
        originals = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.video[0].pix_fmt == "yuv444p"
        assert len(container.streams.audio) == 0
        decoded = list(container.decode(video=0))
    existing_reader = list(read_frames_exact(str(output)))
    assert len(decoded) == len(existing_reader) == 9
    for chapter, clip in zip(chapters, clips, strict=True):
        tags = {key.lower(): value for key, value in chapter["metadata"].items()}
        assert json.loads(tags["payload"]) == json.loads(json.dumps(asdict(clip)))
        count = clip.end_frame - clip.start_frame
        assert float(chapter["start"] * chapter["time_base"]) == pytest.approx(float(offset / rate), abs=0.001)
        assert float(chapter["end"] * chapter["time_base"]) == pytest.approx(float((offset + count) / rate), abs=0.001)
        left, top, right, bottom = clip.box
        for index in range(count):
            expected = originals[clip.start_frame + index][top:bottom, left:right]
            actual = decoded[offset + index].to_ndarray(format="rgb24")
            assert actual.shape == (bottom - top, right - left, 3)
            np.testing.assert_allclose(actual, expected, atol=6)
            np.testing.assert_array_equal(existing_reader[offset + index], actual)
            assert decoded[offset + index].time == pytest.approx(float((offset + index) / rate), abs=0.001)
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            container.seek(round(offset / rate / stream.time_base), stream=stream)
            first = next(container.decode(stream))
            assert (first.width, first.height) == (right - left, bottom - top)
        offset += count
    # The existing range reader works across chapter size changes, too.
    selected = list(read_frames_exact(str(output), start_frame=2, end_frame=7))
    for actual, expected in zip(selected, existing_reader[2:8], strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("clips", [
    [],
    [ChapterClip(-1, 3, (0, 0, 20, 30))],
    [ChapterClip(3, 3, (0, 0, 20, 30))],
    [ChapterClip(0, 3, (-1, 0, 20, 30))],
    [ChapterClip(0, 3, (0, 0, 481, 30))],
    [ChapterClip(0, 3, (0, 0, 20, 97))],
    [ChapterClip(0, 3, (20, 0, 20, 30))],
    [ChapterClip(0.5, 3, (0, 0, 20, 30))],
    [ChapterClip(0, 3, (0, 0, 20.5, 30))],
    [ChapterClip(0, 13, (0, 0, 32, 32))],
    [ChapterClip(0, 3, (0, 0, 470, 32))],
    [ChapterClip(0, 3, (0, 0, 472, 32))],
    [ChapterClip(0, 3, (0, 0, 32, 33))],
])
def test_rejects_invalid_clips_without_replacing_destination(source, tmp_path, clips):
    path, _ = source
    output = tmp_path / "chapters.mkv"
    output.write_bytes(b"keep this file")
    with pytest.raises(ValueError, match="at least one|invalid chapter|source ends|multiples of 16"):
        write_chapters(path, output, clips)
    assert output.read_bytes() == b"keep this file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chapters.mkv", "source.mkv"]


def test_does_not_overwrite_source(source):
    path, _ = source
    original = path.read_bytes()
    with pytest.raises(ValueError, match="source and destination"):
        write_chapters(path, path, [ChapterClip(0, 1, (0, 0, 20, 30))])
    assert path.read_bytes() == original


def test_no_chapters(source):
    path, _ = source
    assert list(list_chapters(path)) == []


def test_single_frame_chapters_and_overlapping_trims(source, tmp_path):
    path, rate = source
    output = tmp_path / "short.mkv"
    clips = [
        ChapterClip(0, 1, (0, 0, 32, 32)),
        ChapterClip(0, 7, (0, 0, 64, 48)),
        ChapterClip(1, 2, (0, 0, 48, 64)),
    ]
    write_chapters(path, output, clips)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert [(f.width, f.height) for f in frames] == [(32, 32)] + [(64, 48)] * 7 + [(48, 64)]
    assert [f.time for f in frames] == pytest.approx([float(i / rate) for i in range(9)], abs=0.001)
