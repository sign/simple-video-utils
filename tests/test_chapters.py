import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters
from simple_video_utils.frames import read_frames_batched, read_frames_exact

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


@pytest.mark.parametrize("rate", [Fraction(3), Fraction(25), Fraction(48), Fraction(30000, 1001)])
@pytest.mark.parametrize("lengths", [(8, 1, 8), (2, 3, 1)])
def test_short_chapters_keep_packet_order_and_seek_to_their_keyframes(tmp_path, rate, lengths):
    source = tmp_path / "source.mkv"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(max(lengths)):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 40 + index * 10, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    clips = [ChapterClip(0, length, (0, 0, 32 + (i % 2) * 16, 32)) for i, length in enumerate(lengths)]
    output = tmp_path / "chapters.mkv"
    write_chapters(source, output, clips)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        duration = container.duration / av.time_base
    assert len(frames) == sum(lengths)
    assert [f.time for f in frames] == pytest.approx([float(i / rate) for i in range(sum(lengths))], abs=0.001)
    assert duration == pytest.approx(float(sum(lengths) / rate), abs=0.001)
    assert [(f.width, f.height) for f in frames] == [
        (clip.size[0], clip.size[1]) for clip in clips for _ in range(clip.end_frame)
    ]
    offset = 0
    for chapter, length in zip(list_chapters(output), lengths, strict=True):
        start = chapter["start"] * chapter["time_base"]
        assert float(start) == pytest.approx(float(offset / rate), abs=0.001)
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            container.seek(int(start / stream.time_base), stream=stream)
            first = next(container.decode(stream))
            assert first.pts * first.time_base == start
        offset += length



def test_truncated_trim_raises_instead_of_writing_misleading_chapters(tmp_path):
    with pytest.raises(ValueError, match="shorter"):
        write_chapters(SOURCE, tmp_path / "short.mkv", [ChapterClip(128, 133, (0, 0, 32, 32))])


@pytest.mark.parametrize("box", [(-4, 0, 28, 32), (11, 6, 43, 38)])
@pytest.mark.parametrize("pixel_format", ["yuv420p10le", "yuv422p10le"])
def test_cropping_and_padding_preserve_ten_bit_source_precision(tmp_path, box, pixel_format):
    source = tmp_path / "ten-bit.mkv"
    frame = av.VideoFrame(width=64, height=48, format=pixel_format)
    for index, plane in enumerate(frame.planes):
        plane.update(np.full((plane.height, plane.line_size // 2), 514 if index == 0 else 512, dtype=np.uint16))
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, pixel_format
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, box)], crf=0)
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0))
        assert actual.format.name == pixel_format
        luma = np.frombuffer(actual.planes[0], np.uint16).reshape(actual.height, -1)[:, :actual.width]
        if box[0] < 0:
            np.testing.assert_array_equal(luma[:, :4], 64)  # limited-range black at ten bits
            np.testing.assert_array_equal(luma[:, 4:], 514)
        else:
            np.testing.assert_array_equal(luma, 514)
        for plane in actual.planes[1:]:
            chroma = np.frombuffer(plane, np.uint16).reshape(plane.height, -1)[:, :plane.width]
            np.testing.assert_allclose(chroma, 512, atol=1)


def test_source_keyframes_and_full_range_do_not_change_chapter_encoding(tmp_path):
    source = tmp_path / "keyframes.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=25, options={"crf": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuvj420p"
        stream.codec_context.gop_size = 1
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 192, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "chapters.mkv"
    write_chapters(source, output, [ChapterClip(0, 10, (-4, 0, 60, 48))])
    with av.open(str(output)) as container:
        assert [packet.is_keyframe for packet in container.demux(video=0) if packet.size] == [True] + [False] * 9
    with av.open(str(output)) as container:
        frame = next(container.decode(video=0))
        assert frame.format.name == "yuvj420p"
        rgb = frame.to_ndarray(format="rgb24")
        assert rgb[:, :2].max() <= 2
        np.testing.assert_allclose(rgb[:, 8:], 192, atol=2)


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv422p"])
def test_odd_vertical_crop_resamples_chroma_on_the_crop_grid(tmp_path, pixel_format):
    source = tmp_path / "vertical-colour.mkv"
    frame = av.VideoFrame(width=64, height=48, format=pixel_format)
    for index, plane in enumerate(frame.planes):
        values = np.full((plane.height, plane.line_size), 128, dtype=np.uint8)
        if index == 1:
            values[:] = 40 + np.arange(plane.height)[:, None] * 4
        elif index == 2:
            values[:, :plane.width] = np.where(np.arange(plane.width)[None, :] < 16, 80, 180)
        plane.update(values)
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, pixel_format
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "cropped.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, (10, 7, 42, 39))], crf=0)
    with av.open(str(output)) as container:
        decoded = next(container.decode(video=0))
        plane = decoded.planes[1]
        chroma = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
    # 4:2:0 averages source chroma rows 3/4 (52/56); 4:2:2 keeps source row 7 (68).
    first = 54 if pixel_format == "yuv420p" else 68
    expected = np.repeat((first + np.arange(plane.height) * 4)[:, None], plane.width, axis=1)
    np.testing.assert_allclose(chroma, expected, atol=1)
    if pixel_format == "yuv422p":
        # An odd top needs no conversion in 4:2:2; keep horizontal colour edges lossless.
        plane = decoded.planes[2]
        chroma = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
        expected = np.repeat(np.where(np.arange(plane.width)[None, :] + 5 < 16, 80, 180), plane.height, axis=0)
        np.testing.assert_array_equal(chroma, expected)


def test_size_reads_and_batches_chapters_of_different_sizes(tmp_path):
    output = tmp_path / "chapters.mkv"
    write_chapters(SOURCE, output, [ChapterClip(30, 35, (10, 6, 74, 38)), ChapterClip(60, 64, (40, 20, 72, 84))])
    assert {frame.shape for frame in read_frames_exact(str(output), size=(48, 40))} == {(40, 48, 3)}
    with pytest.raises(ValueError, match="size="):
        read_frames_batched(str(output))
    assert read_frames_batched(str(output), size=(32, 32)).shape == (9, 32, 32, 3)
