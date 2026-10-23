import io
import json
import math
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.video.reformatter import ColorRange

from simple_video_utils.chapters import (
    ChapterClip,
    join_chapters,
    list_chapters,
    read_chapters,
    slice_chapters,
    split_chapters,
    write_chapters,
)
from simple_video_utils.frames import read_frames_exact

SOURCE = Path(__file__).parent / "assets" / "example.mp4"
# stores 640x360 frames that play as 360x640 (display rotation 90)
ROTATED = Path(__file__).parent / "assets" / "rotated90.mp4"


def _write_chroma_source(path, pixel_format, color_range=ColorRange.UNSPECIFIED, colorspace=2):
    frame = av.VideoFrame(width=64, height=48, format=pixel_format)
    frame.time_base = Fraction(1, 25)
    frame.color_range = color_range
    frame.colorspace = colorspace
    dtype = np.uint16 if "10" in pixel_format else np.uint8
    shift = 2 if dtype == np.uint16 else 0
    for plane in frame.planes:
        y, x = np.indices((plane.height, plane.line_size // np.dtype(dtype).itemsize))
        plane.update((np.where((x + y) % 2, 64, 192) << shift).astype(dtype))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, pixel_format
        stream.codec_context.color_range = color_range
        stream.codec_context.colorspace = colorspace
        if colorspace != 2:
            stream.codec_context.color_primaries = stream.codec_context.color_trc = 1
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return frame, dtype, shift



@pytest.mark.parametrize("colorspace", [1, 6, 9])  # BT.709, SMPTE 170M and BT.2020 non-constant luminance.
def test_padding_keeps_source_color_matrix_and_samples(tmp_path, colorspace):
    source = tmp_path / "source.mkv"
    original, _, _ = _write_chroma_source(source, "yuv420p", ColorRange.MPEG, colorspace)
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, (-4, 0, 28, 32))], crf=0)
    with av.open(str(output)) as container:
        frame = next(container.decode(video=0))
        assert frame.colorspace == colorspace
        assert container.streams.video[0].codec_context.color_primaries == 1
        assert container.streams.video[0].codec_context.color_trc == 1
    for index, (plane, source_plane) in enumerate(zip(frame.planes, original.planes, strict=True)):
        pixels = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
        source_pixels = np.frombuffer(source_plane, np.uint8).reshape(source_plane.height, -1)
        offset = 4 if index == 0 else 2
        expected = np.full(pixels.shape, 16 if index == 0 else 128, dtype=np.uint8)
        expected[:, offset:] = source_pixels[:plane.height, :plane.width - offset]
        np.testing.assert_array_equal(pixels, expected)



@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le", "yuv422p10le", "yuv444p10le"])
@pytest.mark.parametrize("box", [(-4, -4, 28, 28), (44, 28, 76, 60)])
def test_aligned_padding_preserves_native_chroma(tmp_path, pixel_format, box):
    source = tmp_path / "source.mkv"
    original, dtype, shift = _write_chroma_source(source, pixel_format)
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, box)], crf=0)
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0))
    for index, (plane, source_plane) in enumerate(zip(actual.planes, original.planes, strict=True)):
        pixels = np.frombuffer(plane, dtype).reshape(plane.height, -1)[:, :plane.width]
        source_pixels = np.frombuffer(source_plane, dtype).reshape(source_plane.height, -1)[:, :source_plane.width]
        y, x = np.indices(pixels.shape)
        x += box[0] // (64 // source_plane.width)
        y += box[1] // (48 // source_plane.height)
        valid = (x >= 0) & (x < source_plane.width) & (y >= 0) & (y < source_plane.height)
        expected = np.full(pixels.shape, (16 if index == 0 else 128) << shift, dtype=dtype)
        expected[valid] = source_pixels[y[valid], x[valid]]
        # 10-bit x264 CRF 0 still quantizes slightly; chroma resampling would lose far more detail.
        np.testing.assert_allclose(pixels, expected, rtol=0, atol=6 if shift else 0)



def test_odd_padded_edges_keep_full_resolution_chroma(tmp_path):
    source = tmp_path / "source.mkv"
    frame, _, _ = _write_chroma_source(source, "yuv420p")
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, (34, 20, 67, 53), resolution=(32, 32))], crf=0)
    graph = av.filter.Graph()
    graph.link_nodes(
        graph.add_buffer(template=frame), graph.add("format", "yuv444p16le"),
        graph.add("pad", "67:53:0:0:black"), graph.add("crop", "33:33:34:20:exact=1"),
        graph.add("scale", "32:32"), graph.add("format", "yuv420p"), graph.add("buffersink"),
    )
    graph.configure()
    graph.push(frame)
    expected = graph.pull().to_ndarray()
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0)).to_ndarray()
    np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
def test_full_range_padding_keeps_black_borders(tmp_path, pixel_format):
    source = tmp_path / "source.mkv"
    _, dtype, shift = _write_chroma_source(source, pixel_format, ColorRange.JPEG)
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, (-4, 0, 28, 32))], crf=0)
    with av.open(str(output)) as container:
        frame = next(container.decode(video=0))
        assert frame.color_range == ColorRange.JPEG
        plane = frame.planes[0]
        luma = np.frombuffer(plane, dtype).reshape(plane.height, -1)[:, :plane.width]
    np.testing.assert_allclose(luma[:, :2], 0, rtol=0, atol=6 if shift else 1)



@pytest.mark.parametrize("resolution", [None, (32, 32)])
def test_full_range_luma_preserves_all_levels(tmp_path, resolution):
    source = tmp_path / "ramp.mkv"
    frame = av.VideoFrame(64, 48, "yuv420p")
    frame.color_range = ColorRange.JPEG
    for index, plane in enumerate(frame.planes):
        values = np.full((plane.height, plane.line_size), 128, dtype=np.uint8)
        if index == 0:
            values[:, :plane.width] = np.arange(48 * 64, dtype=np.uint8).reshape(48, 64)
        plane.update(values)
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.color_range = ColorRange.JPEG
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "chapter.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, (0, 0, 64, 48), resolution=resolution)], crf=0)
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0))
    assert actual.color_range == ColorRange.JPEG
    expected = frame.reformat(*(resolution or (64, 48)), interpolation="BICUBIC", threads=1)
    np.testing.assert_array_equal(actual.to_ndarray(), expected.to_ndarray())


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


@pytest.mark.parametrize("box", [(-4, -2, 28, 30), (36, 18, 68, 50)])
def test_padding_on_the_chroma_grid_keeps_source_pixels(tmp_path, box):
    # noise chroma changes on any 4:4:4 round trip, so equality shows the crop and padding stayed in yuv420p
    source = tmp_path / "noise.mkv"
    frame = av.VideoFrame(width=64, height=48, format="yuv420p")
    rng = np.random.default_rng(0)
    planes = [rng.integers(16, 236, (plane.height, plane.line_size), dtype=np.uint8) for plane in frame.planes]
    for plane, values in zip(frame.planes, planes, strict=True):
        plane.update(values)
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    output = tmp_path / "padded.mkv"
    write_chapters(source, output, [ChapterClip(0, 1, box)], crf=0)
    with av.open(str(output)) as container:
        actual = next(container.decode(video=0))
    for index, (plane, values) in enumerate(zip(actual.planes, planes, strict=True)):
        scale = 1 if index == 0 else 2
        left, top, right, bottom = (edge // scale for edge in box)
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(right, 64 // scale), min(bottom, 48 // scale)
        expected = np.full((plane.height, plane.width), 16 if index == 0 else 128, dtype=np.uint8)
        expected[y0 - top : y1 - top, x0 - left : x1 - left] = values[y0:y1, x0:x1]
        pixels = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
        np.testing.assert_array_equal(pixels, expected)


def test_resize_controls_h264_alignment(tmp_path):
    output = tmp_path / "resized.mkv"
    write_chapters(SOURCE, output, [ChapterClip(30, 31, (10, 6, 73, 41), resolution=(64, 32))])
    assert next(read_frames_exact(str(output))).shape == (32, 64, 3)
    with pytest.raises(ValueError, match="multiples of 16"):
        write_chapters(SOURCE, tmp_path / "invalid.mkv", [ChapterClip(30, 31, (0, 0, 64, 32), resolution=(62, 32))])


def test_odd_size_past_the_frame_pads_to_its_resolution(tmp_path):
    # even left and top, but a 327x245 padded size is off the 4:2:0 chroma grid
    output = tmp_path / "odd.mkv"
    write_chapters(SOURCE, output, [ChapterClip(30, 31, (-4, -2, 323, 243), resolution=(64, 32))])
    assert next(read_frames_exact(str(output))).shape == (32, 64, 3)


@pytest.mark.parametrize("box", [(-32, 0, 0, 32), (320, 0, 352, 32), (0, 240, 32, 272)])
def test_padding_with_no_source_overlap_is_black(tmp_path, box):
    output = tmp_path / "black.mkv"
    write_chapters(SOURCE, output, [ChapterClip(0, 1, box)], crf=0)
    with av.open(str(output)) as container:
        frame = next(container.decode(video=0))
    for plane, black in zip(frame.planes, (16, 128, 128), strict=True):
        pixels = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
        np.testing.assert_allclose(pixels, black, rtol=0, atol=1)


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



def test_clips_out_of_source_order_or_overlapping_keep_their_frames(tmp_path):
    # the second and third clips start before the previous one ended, so each decodes the source again
    clips = [
        ChapterClip(60, 64, (40, 20, 72, 84)),
        ChapterClip(30, 35, (10, 6, 74, 38)),
        ChapterClip(32, 36, (10, 6, 74, 38)),
    ]
    output = tmp_path / "chapters.mkv"
    write_chapters(SOURCE, output, clips, crf=0)
    original = list(read_frames_exact(str(SOURCE)))
    expected = [
        original[index][clip.box[1] : clip.box[3], clip.box[0] : clip.box[2]]
        for clip in clips
        for index in range(clip.start_frame, clip.end_frame)
    ]
    for frame, crop in zip(read_frames_exact(str(output)), expected, strict=True):
        np.testing.assert_allclose(frame, crop, atol=1)


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


def _write_rotated_noise(path, pixel_format, rotation):
    """Four frames of lossless 64x48 noise in `pixel_format`, stored unturned and played turned by `rotation`."""
    rng = np.random.default_rng(1)
    dtype = np.uint16 if pixel_format.endswith("le") else np.uint8
    flat = path.with_name(f"flat-{path.name}")
    with av.open(str(flat), "w") as container:
        stream = container.add_stream("libx264", rate=25, options={"qp": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, pixel_format
        for index in range(4):
            frame = av.VideoFrame(width=64, height=48, format=pixel_format)
            for plane in frame.planes:
                # within the limited range of 8-bit and 10-bit samples alike
                high = 940 if dtype == np.uint16 else 235
                plane.update(rng.integers(20, high, (plane.height, plane.line_size // np.dtype(dtype).itemsize), dtype))
            frame.pts, frame.time_base = index, Fraction(1, 25)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    with av.open(str(flat)) as original, av.open(str(path), "w") as destination:
        stream = destination.add_stream_from_template(original.streams.video[0])
        stream.set_display_rotation(rotation)
        for packet in original.demux(video=0):
            if packet.dts is not None:
                packet.stream = stream
                destination.mux(packet)


def _chapter_planes(source, output, clips):
    write_chapters(source, output, clips, crf=0)
    with av.open(str(output)) as container:
        return [
            [
                np.frombuffer(plane, np.uint16 if frame.format.name.endswith("le") else np.uint8)
                .reshape(plane.height, -1)[:, :plane.width]
                for plane in frame.planes
            ]
            for frame in container.decode(video=0)
        ]


def _assert_chapters_of_upright_copy(tmp_path, source, rotation, clips):
    """`source`'s chapters equal those of the same video stored upright: its frames turned as FFmpeg autorotates
    them, losslessly, with no rotation left."""
    turns = {0: [], 90: [("transpose", "cclock")], 180: [("hflip",), ("vflip",)], 270: [("transpose", "clock")]}
    upright = tmp_path / "upright.mkv"
    with av.open(str(source)) as rotated, av.open(str(upright), "w") as container:
        stored = rotated.streams.video[0]
        stream = container.add_stream("libx264", rate=stored.average_rate, options={"qp": "0"})
        stream.width, stream.height = (stored.height, stored.width) if rotation % 180 else (stored.width, stored.height)
        stream.pix_fmt = stored.pix_fmt
        for name in ("color_range", "colorspace", "color_primaries", "color_trc"):
            setattr(stream.codec_context, name, getattr(stored.codec_context, name))
        graph = av.filter.Graph()
        turned_nodes = [graph.add(*turn) for turn in turns[rotation % 360]]
        graph.link_nodes(graph.add_buffer(template=stored), *turned_nodes, graph.add("buffersink"))
        graph.configure()
        for index, frame in enumerate(rotated.decode(stored)):
            graph.push(frame)
            turned = graph.pull()
            turned.pts, turned.time_base = index, 1 / stored.average_rate
            container.mux(stream.encode(turned))
        container.mux(stream.encode())
    written = [_chapter_planes(video, tmp_path / f"{name}-chapters.mkv", clips)
               for name, video in (("rotated", source), ("upright", upright))]
    assert len(written[0]) == sum(clip.end_frame - clip.start_frame for clip in clips)
    for rotated_planes, upright_planes in zip(*written, strict=True):
        for rotated_plane, upright_plane in zip(rotated_planes, upright_planes, strict=True):
            np.testing.assert_array_equal(rotated_plane, upright_plane)


def test_rotated_source_writes_the_chapters_of_its_upright_copy(tmp_path):
    clips = [ChapterClip(0, 3, (16, 32, 336, 608)), ChapterClip(3, 6, (-16, 576, 368, 672), resolution=(192, 48))]
    _assert_chapters_of_upright_copy(tmp_path, ROTATED, 90, clips)


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("pixel_format", ["yuv420p10le", "yuv444p", "yuv444p10le"])
def test_rotated_pixel_formats_write_the_chapters_of_their_upright_copy(tmp_path, pixel_format, rotation):
    source = tmp_path / "rotated.mp4"
    _write_rotated_noise(source, pixel_format, rotation)
    height = 64 if rotation % 180 else 48
    # one box inside the displayed frame, one padded past its bottom-left corner
    clips = [ChapterClip(0, 2, (0, 0, 32, 32)), ChapterClip(2, 4, (-16, height - 16, 16, height + 16))]
    _assert_chapters_of_upright_copy(tmp_path, source, rotation, clips)


@pytest.mark.parametrize("rotation", [90, 270])
@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
@pytest.mark.parametrize("box", [(1, 0, 33, 32), (0, 1, 32, 33), (7, 5, 39, 37), (-3, -5, 29, 27)])
def test_rotated_odd_display_edges_write_the_chapters_of_the_upright_copy(tmp_path, rotation, pixel_format, box):
    # an odd left or top in display coordinates puts the crop off the turned frame's chroma grid
    source = tmp_path / "rotated.mp4"
    _write_rotated_noise(source, pixel_format, rotation)
    _assert_chapters_of_upright_copy(tmp_path, source, rotation, [ChapterClip(0, 2, box)])


@pytest.mark.parametrize("rotation", [90, 270])
@pytest.mark.parametrize("clips", [
    # out of order and overlapping: decoding restarts at frame 0 and must keep turning the frames
    [ChapterClip(2, 4, (0, 0, 32, 48)), ChapterClip(0, 3, (16, 16, 48, 48), resolution=(16, 16)),
     ChapterClip(1, 3, (-16, 0, 16, 32))],
    # a gap after frame 0, which the rotation is read from: that frame must be cut once, not twice
    [ChapterClip(0, 1, (0, 0, 32, 32)), ChapterClip(3, 4, (0, 0, 32, 32))],
])
def test_rotated_clips_in_any_order_write_the_chapters_of_the_upright_copy(tmp_path, rotation, clips):
    source = tmp_path / "rotated.mp4"
    _write_rotated_noise(source, "yuv420p", rotation)
    _assert_chapters_of_upright_copy(tmp_path, source, rotation, clips)


@pytest.mark.parametrize("rotation", [-90, -180, -270, 450])
def test_negative_and_wrapped_rotations_write_the_chapters_of_the_upright_copy(tmp_path, rotation):
    source = tmp_path / "rotated.mp4"
    _write_rotated_noise(source, "yuv420p", rotation)
    _assert_chapters_of_upright_copy(tmp_path, source, rotation, [ChapterClip(0, 2, (0, 0, 32, 32))])


@pytest.mark.parametrize("rotation", [90, 270])
def test_rotated_422_source_is_turned_upright(tmp_path, rotation):
    # transpose takes no 4:2:2, so FFmpeg converts before the turn and the format pin converts back; turned 4:2:2
    # chroma has no exact upright copy, so luma alone is held to the stored frame turned
    source = tmp_path / "rotated.mp4"
    _write_rotated_noise(source, "yuv422p", rotation)
    with av.open(str(source)) as container:
        plane = next(container.decode(video=0)).planes[0]
        stored = np.frombuffer(plane, np.uint8).reshape(plane.height, -1)[:, :plane.width]
    luma, chroma, _ = _chapter_planes(source, tmp_path / "chapters.mkv", [ChapterClip(0, 1, (0, 0, 48, 64))])[0]
    assert chroma.shape == (64, 24)
    np.testing.assert_array_equal(luma, np.rot90(stored, 1 if rotation == 90 else 3))


def test_rotated_source_crops_the_displayed_frame_upright(tmp_path):
    clips = [ChapterClip(0, 3, (16, 32, 336, 608)), ChapterClip(3, 6, (-16, 576, 368, 672))]
    output = tmp_path / "chapters.mkv"
    write_chapters(ROTATED, output, clips, crf=0)
    shown = list(read_frames_exact(str(ROTATED)))
    frames = list(read_frames_exact(str(output)))
    assert [frame.shape for frame in frames] == [(576, 320, 3)] * 3 + [(96, 384, 3)] * 3
    for index in range(3):
        np.testing.assert_array_equal(frames[index], shown[index][32:608, 16:336])
        # the second clip pads 16 pixels left and right and 32 below the 360x640 frame; padding converts its chroma, so
        # its pixels move by a few levels, where a crop turned the wrong way is ~37 off on average
        inside = frames[3 + index][:64, 16:376].astype(int) - shown[3 + index][576:640]
        assert np.abs(inside).mean() < 1
        assert frames[3 + index][64:].max() <= 3


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("resolution", [None, (64, 64)])
def test_chapters_autorotate_before_cropping_and_padding(tmp_path, rotation, resolution):
    source = tmp_path / "rotated.mp4"
    with av.open(str(SOURCE)) as original, av.open(str(source), "w") as destination:
        stream = destination.add_stream_from_template(original.streams.video[0])
        stream.set_display_rotation(rotation)
        for packet in original.demux(video=0):
            if packet.dts is not None:
                packet.stream = stream
                destination.mux(packet)
    with av.open(str(source)) as container:
        stored = next(container.decode(video=0))
        assert stored.rotation % 360 == rotation
        shown = np.rot90(stored.to_ndarray(format="rgb24"), rotation // 90)
    height, width = shown.shape[:2]
    clips = [
        ChapterClip(0, 1, (0, 0, 128, 128)),
        ChapterClip(0, 1, (-16, height - 32, width + 16, height + 32), resolution=resolution),
    ]
    output = tmp_path / "chapters.mkv"
    write_chapters(source, output, clips, crf=0)
    frames = list(read_frames_exact(str(output)))
    np.testing.assert_array_equal(frames[0], shown[:128, :128])
    if resolution is None:
        expected = np.pad(shown[-32:], ((0, 32), (16, 16), (0, 0)))
        np.testing.assert_allclose(frames[1], expected, atol=3)
    else:
        assert frames[1].shape == (64, 64, 3)
        assert frames[1][40:].max() <= 3
        assert frames[1][:24, 8:56].mean() > 10
    with av.open(str(output)) as container:
        assert all(frame.rotation == 0 for frame in container.decode(video=0))


def test_phone_video_crop_uses_display_dimensions(tmp_path):
    output = tmp_path / "phone-crop.mkv"
    write_chapters(ROTATED, output, [ChapterClip(0, 3, (0, 0, 128, 512))], crf=0)
    shown = list(read_frames_exact(str(ROTATED), 0, 2))
    for actual, expected in zip(read_frames_exact(str(output)), shown, strict=True):
        np.testing.assert_array_equal(actual, expected[:512, :128])


def _signer_chapters(tmp_path):
    """Source frames [6, 36) at 48x48 and [54, 84) at 32x32: a 0.6 s gap between them."""
    output = tmp_path / "signers.mkv"
    write_chapters(SOURCE, output, [ChapterClip(6, 36, (0, 0, 48, 48)), ChapterClip(54, 84, (16, 16, 48, 48))])
    return output


def _frame_count(data: bytes) -> int:
    with av.open(io.BytesIO(data)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_read_chapters_places_each_on_the_source_timeline(tmp_path):
    output = _signer_chapters(tmp_path)
    chapters = read_chapters(output)
    assert [(c.start, c.end, c.source, c.source_end) for c in chapters] == [
        (0, 1, Fraction(1, 5), Fraction(6, 5)), (1, 2, Fraction(9, 5), Fraction(14, 5))]
    assert read_chapters(io.BytesIO(output.read_bytes())) == chapters
    assert chapters[1].overlap(1.0, 2.0) == (Fraction(9, 5), 2)
    assert chapters[0].overlap(1.3, 1.7) is None
    assert read_chapters(SOURCE) == []


def test_list_chapters_preserves_native_fields_and_adds_source_times(tmp_path):
    output = _signer_chapters(tmp_path)
    with av.open(str(output)) as container:
        native = container.chapters()
        actual = list(list_chapters(container))
        assert next(container.decode(video=0)).width == 48
    for chapter, original, start in zip(actual, native, [Fraction(1, 5), Fraction(9, 5)], strict=True):
        assert {key: chapter[key] for key in original} == original
        assert (chapter["source_start"], chapter["source_end"]) == (start, start + 1)
    data = io.BytesIO(output.read_bytes())
    data.seek(0, io.SEEK_END)
    assert list(list_chapters(data)) == actual
    assert list(list_chapters(data)) == actual


@pytest.mark.parametrize(("metadata", "expected"), [
    ({"title": "Introduction"}, None),
    ({"payload": json.dumps({"start_frame": 1250, "rate": "1000"})}, Fraction(5, 4)),
])
def test_list_chapters_handles_native_and_explicit_rate_metadata(tmp_path, metadata, expected):
    output = tmp_path / "metadata.mkv"
    with av.open(str(SOURCE)) as source, av.open(str(output), "w") as destination:
        destination.set_chapters([{"id": 0, "start": 0, "end": 1000, "time_base": Fraction(1, 1000),
                                   "metadata": metadata}])
        stream = destination.add_stream_from_template(source.streams.video[0])
        for packet in source.demux(video=0):
            if packet.dts is not None:
                packet.stream = stream
                destination.mux(packet)
    chapter, = list_chapters(output)
    if expected is None:
        assert "source_start" not in chapter
        assert "source_end" not in chapter
    else:
        assert (chapter["source_start"], chapter["source_end"]) == (expected, expected + 1)


def test_split_chapters_yields_each_chapter_from_zero(tmp_path):
    pieces = split_chapters(_signer_chapters(tmp_path))
    chapter, data = next(pieces)  # one chapter at a time
    assert chapter.source == Fraction(1, 5)
    rest = list(pieces)
    assert [c.source for c, _ in rest] == [Fraction(9, 5)]
    for piece, size in zip([data, rest[0][1]], [(48, 48), (32, 32)], strict=True):
        with av.open(io.BytesIO(piece)) as container:
            frames = list(container.decode(video=0))
        assert (len(frames), (frames[0].width, frames[0].height), frames[0].pts) == (30, size, 0)


@pytest.mark.parametrize("cut", [lambda src: next(split_chapters(src)),
                                 lambda src: next(slice_chapters(src, [(0, 1)]))])
def test_a_file_without_chapters_cannot_be_cut_into_them(cut):
    with pytest.raises(ValueError, match="no chapters"):
        cut(SOURCE)


def test_slice_chapters_cuts_source_spans_and_joins_them(tmp_path):
    output = _signer_chapters(tmp_path)
    within, across, gap, whole = slice_chapters(output, [(0.3, 0.9), (1.0, 2.2), (1.3, 1.7), (0, math.inf)], size=32)
    assert [s for s, _ in within] == [Fraction(3, 10)]
    assert [s for s, _ in across] == [1, Fraction(9, 5)]
    assert gap == []
    assert [s for s, _ in whole] == [Fraction(1, 5), Fraction(9, 5)]
    joined = join_chapters(across)
    assert [c.source for c in read_chapters(io.BytesIO(joined))] == [1, Fraction(9, 5)]
    assert [_frame_count(d) for _, d in split_chapters(io.BytesIO(joined))] == [_frame_count(d) for _, d in across]


def test_slice_chapters_reencodes_a_joined_file_at_its_own_size(tmp_path):
    """A 256-like joined file looks already sized; its pieces must still start on a keyframe at the right place."""
    joined = tmp_path / "joined.mkv"
    joined.write_bytes(join_chapters(next(slice_chapters(_signer_chapters(tmp_path), [(1.0, 2.2)], size=32))))
    pieces = next(slice_chapters(joined, [(0, math.inf)], size=32))
    assert [s for s, _ in pieces] == [1, Fraction(9, 5)]
    again = io.BytesIO(join_chapters(pieces))
    assert [c.source for c in read_chapters(again)] == [1, Fraction(9, 5)]
    assert [_frame_count(d) for _, d in split_chapters(again)] == [6, 12]


def test_join_chapters_rescales_packet_durations(tmp_path):
    pieces = next(slice_chapters(_signer_chapters(tmp_path), [(1.0, 2.2)], size=32))
    joined = join_chapters(pieces)
    with av.open(io.BytesIO(joined)) as container:
        assert container.duration / av.time_base == pytest.approx(0.6, abs=0.001)
        for packet in container.demux(video=0):
            if packet.size:
                assert float(packet.duration * packet.time_base) == pytest.approx(1 / 30, abs=0.001)


def test_slice_chapters_labels_the_first_retained_source_frame(tmp_path):
    pieces = next(slice_chapters(_signer_chapters(tmp_path), [(0.301, 0.9)], size=32))
    (start, clip), = pieces
    assert start == Fraction(1, 3)
    assert _frame_count(clip) == 17
    chapter, = read_chapters(io.BytesIO(join_chapters(pieces)))
    assert chapter.source == Fraction(333, 1000)


def test_slice_chapters_labels_frames_after_a_rounded_chapter_start(tmp_path):
    output = tmp_path / "rounded-source.mkv"
    write_chapters(SOURCE, output, [ChapterClip(10, 12, (0, 0, 32, 32)), ChapterClip(40, 47, (0, 0, 32, 32))])
    (start, _), = next(slice_chapters(output, [(1.36, 1.5)], size=32))
    assert start == Fraction(41, 30)


def test_slice_chapters_skips_an_overlap_holding_no_frame_start(tmp_path):
    output = _signer_chapters(tmp_path)               # chapter 1 ends at source 1.2 (frame starts every 1/30 s)
    [piece], = slice_chapters(output, [(1.19, 1.9)], size=32)   # 0.01 s of chapter 1: no frame starts there
    assert piece[0] == Fraction(9, 5)
    assert _frame_count(piece[1]) == 3                 # [1.8, 1.9): frames at 1.8, 1.833, 1.866


def test_slice_chapters_reads_ms_rounded_bounds_as_their_frames(tmp_path):
    """30 fps chapters start at 0.0667 s (frame 2), stored as 67 ms: a whole chapter keeps exactly its frames."""
    output = tmp_path / "rounded.mkv"
    write_chapters(SOURCE, output, [ChapterClip(10, 12, (0, 0, 32, 32)), ChapterClip(40, 47, (0, 0, 32, 32))])
    assert [c.start for c in read_chapters(output)] == [0, Fraction(67, 1000)]
    first, second = next(slice_chapters(output, [(0, math.inf)], size=32))
    assert (_frame_count(first[1]), _frame_count(second[1])) == (2, 7)
