from pathlib import Path

import av
import numpy as np
import pytest
from av.video.reformatter import Interpolation

from simple_video_utils.chapters import ChapterClip, write_chapters
from simple_video_utils.frames import FrameSize, read_frames_batched, read_frames_exact, stack_frames

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(params=["yuv420p", "yuv444p", "yuv420p10le"])
def grayscale_video(tmp_path, request):
    path = tmp_path / "gray.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, request.param
        checker = (np.indices((48, 64)).sum(axis=0) % 2 * 80 + 32).astype(np.uint8)
        for index in range(3):
            array = np.repeat((checker + index * 40)[..., None], 3, axis=2)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return str(path)


def test_area_resize_averages_pixels_and_preserves_frames(grayscale_video):
    original = read_frames_batched(grayscale_video)
    resized = read_frames_batched(grayscale_video, size=FrameSize((32, 24)))
    expected = original.reshape(3, 24, 2, 32, 2, 3).mean(axis=(2, 4))
    assert resized.shape == (3, 24, 32, 3)
    assert resized.dtype == np.uint8
    np.testing.assert_allclose(resized, expected, atol=2)


def test_interpolation_is_forwarded_to_pyav(grayscale_video):
    original = read_frames_batched(grayscale_video)
    resized = read_frames_batched(grayscale_video, size=FrameSize((128, 96), method=Interpolation.POINT))
    np.testing.assert_array_equal(resized, original.repeat(2, axis=1).repeat(2, axis=2))
    smooth = read_frames_batched(grayscale_video, size=FrameSize((128, 96), method=Interpolation.BICUBIC))
    assert not np.array_equal(resized, smooth)


def test_padding_preserves_aspect_ratio_and_centers_on_black(grayscale_video):
    padded = read_frames_batched(grayscale_video, size=FrameSize((80, 81), mode="pad"))
    content = read_frames_batched(grayscale_video, size=FrameSize((80, 60)))
    assert padded.shape == (3, 81, 80, 3)
    assert not padded[:, :10].any()
    assert not padded[:, 70:].any()
    np.testing.assert_array_equal(padded[:, 10:70], content)


def test_size_uses_display_orientation():
    source = str(ASSETS / "rotated90.mp4")
    resized = next(read_frames_exact(source, end_frame=0, size=FrameSize((180, 320))))
    original = next(read_frames_exact(source, end_frame=0))
    expected = original.reshape(320, 2, 180, 2, 3).mean(axis=(1, 3))
    assert resized.shape == (320, 180, 3)
    assert np.abs(resized.astype(float) - expected).mean() < 3
    padded = next(read_frames_exact(source, end_frame=0, size=FrameSize((200, 300), mode="pad")))
    assert padded.shape == (300, 200, 3)
    assert not padded[:, :15].any()
    assert not padded[:, 184:].any()
    assert padded[:, 16:184].any()


def test_size_preserves_range_fps_and_indices():
    source = str(ASSETS / "example.mp4")
    arguments = dict(start_frame=15, end_frame=45, fps=10, return_indices=True)
    _, expected_indices = read_frames_batched(source, **arguments)
    size = FrameSize((48, 32))
    batch, indices = read_frames_batched(source, **arguments, size=size)
    streamed = list(read_frames_exact(source, **arguments, size=size))
    np.testing.assert_array_equal(indices, expected_indices)
    assert batch.shape == (len(indices), 32, 48, 3)
    for (frame, index), expected, expected_index in zip(streamed, batch, indices, strict=True):
        np.testing.assert_array_equal(frame, expected)
        assert index == expected_index
        assert frame.flags.c_contiguous


@pytest.mark.parametrize("mode", ["rescale", "pad"])
def test_variable_size_chapters_resize_in_one_read(tmp_path, mode):
    output = tmp_path / "chapters.mkv"
    write_chapters(ASSETS / "example.mp4", output, [
        ChapterClip(30, 35, (10, 6, 74, 38)),
        ChapterClip(60, 64, (40, 20, 72, 84)),
    ])
    size = FrameSize((48, 40), mode=mode)
    frames = read_frames_batched(str(output), size=size)
    assert frames.shape == (9, 40, 48, 3)
    expected = list(read_frames_exact(str(output), size=size))
    np.testing.assert_array_equal(frames, np.stack(expected))
    with pytest.raises(ValueError, match="different shapes"):
        read_frames_batched(str(output))


def test_shape_change_at_chunk_boundary_is_rejected():
    frames = [np.zeros((2, 2, 3), np.uint8)] * 2 + [np.zeros((1, 1, 3), np.uint8)]
    with pytest.raises(ValueError, match="different shapes"):
        stack_frames(frames, size_hint=2)


@pytest.mark.parametrize("size", [None, FrameSize((63, 48))])
def test_odd_width_read_writes_the_last_rgb_column(tmp_path, size):
    source = tmp_path / "odd.mkv"
    frame = av.VideoFrame(81, 48, "yuv420p")
    for plane, value in zip(frame.planes, (235, 128, 128), strict=True):
        plane.update(bytes([value]) * plane.buffer_size)
    with av.open(str(source), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 81, 48, "yuv420p"
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    for array in read_frames_exact(str(source), size=size):
        np.testing.assert_allclose(array, 255, atol=1)


@pytest.mark.parametrize("resolution", [(0, 10), (-1, 10), (10,), (10, 20, 30)])
def test_invalid_resolution_is_rejected(resolution):
    with pytest.raises(ValueError, match="resolution"):
        FrameSize(resolution)


def test_invalid_mode_and_noninteger_resolution_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        FrameSize((10, 10), mode="crop")
    with pytest.raises(TypeError):
        FrameSize((10.5, 10))


def test_invalid_interpolation_is_rejected():
    with pytest.raises(TypeError, match="Interpolation"):
        FrameSize((10, 10), method="bogus")


@pytest.mark.parametrize("reader", [read_frames_exact, read_frames_batched])
def test_invalid_size_is_rejected_before_opening(reader):
    with pytest.raises(TypeError, match="FrameSize"):
        reader("missing.mp4", size=(32, 24))
