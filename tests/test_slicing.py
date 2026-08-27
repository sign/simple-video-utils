import io
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.frames import read_frames_exact
from simple_video_utils.slicing import slice_video, slice_video_stream


def _encode_video(target, width=320, height=240, frames=30, fps=30, codec="h264", format=None):
    with av.open(target, mode="w", format=format) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            arr = np.full((height, width, 3), i * 8 % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        container.mux(stream.encode())
    return target


def _write_video(path, **kwargs):
    return str(_encode_video(str(path), **kwargs))


def _streaming_video(frames=36, fps=24, codec="h264", format="mpegts") -> bytes:
    buffer = _encode_video(io.BytesIO(), width=64, height=48, frames=frames, fps=fps,
                           codec=codec, format=format)
    return buffer.getvalue()


@pytest.fixture
def video(tmp_path):
    return _write_video(tmp_path / "src.mp4")


def _dims(clip: bytes) -> tuple[int, int]:
    with av.open(io.BytesIO(clip), mode="r") as container:
        frame = next(container.decode(video=0))
        return frame.width, frame.height


def test_slice_returns_one_clip_per_range(video):
    clips = list(slice_video(video, [(0.0, 0.3), (0.5, 0.8)]))
    assert len(clips) == 2
    assert all(clip for clip in clips)


def test_slice_keeps_source_size_by_default(video):
    # Default path stream-copies, so the source resolution is preserved.
    [clip] = slice_video(video, [(0.0, 0.3)])
    assert _dims(clip) == (320, 240)


def test_slice_center_crops_and_resizes(video):
    [clip] = slice_video(video, [(0.0, 0.3)], size=256)
    assert _dims(clip) == (256, 256)


def test_matching_size_is_copied(tmp_path):
    # Source is already 128x128, so size=128 needs no re-encode.
    square = _write_video(tmp_path / "square.mp4", width=128, height=128, frames=15)
    [clip] = slice_video(square, [(0.0, 0.3)], size=128)
    assert _dims(clip) == (128, 128)


def test_copy_playback_starts_at_requested_start():
    # The keyframe lead-in is muxed at negative pts, which the mp4 muxer turns
    # into an edit list — decoders skip straight to `start` instead of showing
    # everything from the keyframe.
    src = str(Path(__file__).parent / "assets" / "example.mp4")
    [clip] = slice_video(src, [(2.7, 3.03)])
    with av.open(io.BytesIO(clip)) as container:
        frames = list(container.decode(video=0))
        first_time = float(frames[0].pts * frames[0].time_base)
        first_pixels = frames[0].to_ndarray(format="rgb24")
    assert first_time == 0.0
    assert len(frames) <= 13  # 10 frames cover [2.7, 3.03] at 30 fps, plus B-frame extras
    [expected] = read_frames_exact(src, start_frame=81, end_frame=81)  # the frame at 2.7s
    np.testing.assert_array_equal(first_pixels, expected)


def test_copy_keeps_trailing_frames():
    # Packets arrive in decode order, so a B-frame with pts <= end can follow a
    # packet with pts > end; cutting on pts used to drop such tail frames.
    # The synthetic fixture encodes without that reordering — use a real asset.
    src = str(Path(__file__).parent / "assets" / "example-short.mp4")
    [clip] = slice_video(src, [(0.0, 0.68)])
    with av.open(io.BytesIO(clip)) as container:
        clip_count = sum(1 for _ in container.decode(video=0))
    assert clip_count >= 21  # frames 0..20 cover [0, 0.68] at 30 fps


@pytest.mark.parametrize("size", [None, 256])
def test_slicing_is_reproducible(video, size):
    # Both paths must be byte-stable: no wall-clock timestamp, encoder state, or
    # container metadata may leak into the output across repeated runs.
    outputs = {
        tuple(slice_video(video, [(0.0, 0.3), (0.5, 0.8)], size=size))
        for _ in range(3)
    }
    assert len(outputs) == 1


@pytest.mark.parametrize("bad", [(5.0, 5.5), (-0.1, 0.3), (0.5, 0.2), (0.5, 0.5)])
@pytest.mark.parametrize("size", [None, 256])
def test_out_of_range_slice_raises(video, bad, size):
    with pytest.raises(ValueError, match="out of range"):
        list(slice_video(video, [bad], size=size))


def _frames(video: bytes) -> list[np.ndarray]:
    with av.open(io.BytesIO(video)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_stream_slices_are_split_into_duration_windows():
    data = _streaming_video()
    source_frames = _frames(data)
    clips = list(slice_video_stream(io.BytesIO(data), duration=0.5))
    decoded = [_frames(clip) for clip in clips]
    assert len(decoded) == 3
    assert [frames[0].shape[:2] for frames in decoded] == [(48, 64)] * 3
    for index, frames in enumerate(decoded):
        assert len(frames) >= 12
        np.testing.assert_array_equal(frames[0], source_frames[index * 12])


def test_stream_slices_webm_reencodes_exact_windows():
    # WebM has no edit list to hide the keyframe lead-in a copied clip would
    # have to carry, so VP8/VP9 sources are decoded and re-encoded into exact
    # MP4 windows: no lead-in, no trailing frames, window content only.
    data = _streaming_video(codec="vp8", format="webm")
    source_frames = _frames(data)
    clips = list(slice_video_stream(io.BytesIO(data), duration=0.5))
    assert len(clips) == 3
    for index, clip in enumerate(clips):
        frames = _frames(clip)
        assert len(frames) == 12
        np.testing.assert_allclose(  # atol: H.264 is lossy
            frames[0].astype(int), source_frames[index * 12].astype(int), atol=3)


def test_webm_file_is_reencoded_at_source_resolution(tmp_path):
    # The copy path is off for codecs MP4 can't carry, even without ``size``:
    # slice_video re-encodes them at the source resolution.
    src = _write_video(tmp_path / "src.webm", codec="vp8")
    [clip] = slice_video(src, [(0.0, 0.3)])
    assert _dims(clip) == (320, 240)


class _CountingReader(io.RawIOBase):
    """Non-seekable reader that tracks how far the source has been read."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        chunk = self.data[self.pos : self.pos + len(buffer)]
        buffer[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def test_stream_clips_arrive_before_eof():
    # The point of streaming: each clip is yielded once its window's packets
    # have arrived, not after the source is exhausted. PyAV reads the input in
    # buffer_size chunks, so read-ahead — and with it clip latency — is capped
    # by buffer_size, not by the stream length.
    reader = _CountingReader(_streaming_video(frames=240))  # 10s at 24 fps
    positions = [reader.pos for _ in slice_video_stream(reader, duration=0.5, buffer_size=4096)]
    assert len(positions) == 20
    assert positions[0] <= 3 * 4096
    assert positions == sorted(positions)
    assert len(set(positions)) > 2


@pytest.mark.parametrize("duration", [0, -1, float("inf")])
def test_stream_slice_duration_must_be_positive_and_finite(duration):
    with pytest.raises(ValueError, match="duration"):
        list(slice_video_stream(io.BytesIO(b"video"), duration=duration))
