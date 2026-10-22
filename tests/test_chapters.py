import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.video.reformatter import ColorRange

from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters
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


def test_rotated_source_writes_the_chapters_of_its_upright_copy(tmp_path):
    # the same video stored upright: its frames turned as FFmpeg autorotates them, losslessly, with no rotation left
    upright = tmp_path / "upright.mkv"
    with av.open(str(ROTATED)) as source, av.open(str(upright), "w") as container:
        stored = source.streams.video[0]
        stream = container.add_stream("libx264", rate=stored.average_rate, options={"qp": "0"})
        stream.width, stream.height, stream.pix_fmt = stored.height, stored.width, stored.pix_fmt
        for name in ("color_range", "colorspace", "color_primaries", "color_trc"):
            setattr(stream.codec_context, name, getattr(stored.codec_context, name))
        graph = av.filter.Graph()
        graph.link_nodes(graph.add_buffer(template=stored), graph.add("transpose", "cclock"), graph.add("buffersink"))
        graph.configure()
        for index, frame in enumerate(source.decode(stored)):
            graph.push(frame)
            turned = graph.pull()
            turned.pts, turned.time_base = index, 1 / stored.average_rate
            container.mux(stream.encode(turned))
        container.mux(stream.encode())
    clips = [ChapterClip(0, 3, (16, 32, 336, 608)), ChapterClip(3, 6, (-16, 576, 368, 672), resolution=(192, 48))]
    written = []
    for name, video in (("rotated", ROTATED), ("upright", upright)):
        write_chapters(video, tmp_path / f"{name}-chapters.mkv", clips, crf=0)
        with av.open(str(tmp_path / f"{name}-chapters.mkv")) as container:
            written.append([frame.to_ndarray() for frame in container.decode(video=0)])
    assert len(written[0]) == 6
    for rotated, stored_upright in zip(*written, strict=True):
        np.testing.assert_array_equal(rotated, stored_upright)


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
