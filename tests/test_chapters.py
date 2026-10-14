import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils import frames as frames_module
from simple_video_utils.frames import read_frames_batched, read_frames_exact
from simple_video_utils.metadata import count_frames, open_video

ASSETS = Path(__file__).parent / "assets"
FPS = 10
# source frames [start, end) and the square side of each chapter, on whole 16-px macroblocks as an export writes it:
# the file starts 3 s in, with gaps between the chapters
CHAPTERS = [(30, 40, 64), (45, 55, 32), (60, 70, 48)]
INDICES = [*range(30, 40), *range(45, 55), *range(60, 70)]


@pytest.fixture(scope="module")
def chapter_video(tmp_path_factory) -> str:
    """One Matroska track holding each chapter at its source time and its own size, as a chapter export writes it:
    every chapter has its own encoder, whose headers travel in the stream, and a JSON payload tag."""
    path = tmp_path_factory.mktemp("chapters") / "chapters.mkv"
    rate = Fraction(FPS)
    with av.open(str(path), "w", format="matroska") as container:
        container.set_chapters([
            {
                "id": k + 1,
                "start": start * 1000 // FPS,
                "end": end * 1000 // FPS,
                "time_base": Fraction(1, 1000),
                "metadata": {"title": f"{start}-{end}", "payload": json.dumps({"start_frame": start, "side": side})},
            }
            for k, (start, end, side) in enumerate(CHAPTERS)
        ])
        track = container.add_stream("libx264", rate=rate)
        track.width = track.height = CHAPTERS[0][2]
        track.pix_fmt = "yuv420p"
        for start, end, side in CHAPTERS:
            encoder = av.CodecContext.create("libx264", "w")
            encoder.width = encoder.height = side
            encoder.pix_fmt = "yuv420p"
            encoder.time_base, encoder.framerate = 1 / rate, rate
            for index in [*range(start, end), None]:
                frame = None
                if index is not None:
                    image = np.full((side, side, 3), 10 * (index % 10) + 100, np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
                    frame.pts = index
                for packet in encoder.encode(frame):
                    packet.stream = track
                    container.mux(packet)
    return str(path)


def test_chapter_frames_keep_their_own_size_at_their_source_index(chapter_video):
    frames = list(read_frames_exact(chapter_video, return_indices=True))
    assert [index for _, index in frames] == INDICES
    assert [frame.shape for frame, _ in frames] == [(64, 64, 3)] * 10 + [(32, 32, 3)] * 10 + [(48, 48, 3)] * 10


def test_size_scales_every_frame_up_or_down(chapter_video):
    frames = list(read_frames_exact(chapter_video, size=(48, 40)))
    assert {frame.shape for frame in frames} == {(40, 48, 3)}
    # each frame is one grey level, which scaling keeps
    levels = [10 * (index % 10) + 100 for index in INDICES]
    assert [float(frame.mean()) for frame in frames] == pytest.approx(levels, abs=2)


def test_each_frame_carries_its_chapter_payload(chapter_video):
    frames = read_frames_exact(chapter_video, return_indices=True, return_chapters=True)
    payloads = [payload for _, _, payload in frames]
    expected = [{"start_frame": start, "side": side} for start, end, side in CHAPTERS for _ in range(end - start)]
    assert payloads == expected


def test_a_frame_range_follows_the_source_timeline(chapter_video):
    # a read seeks to its first frame's keyframe, on the file's own source times
    frames = read_frames_exact(chapter_video, start_frame=38, end_frame=46, return_indices=True)
    assert [index for _, index in frames] == [38, 39, 45, 46]
    frames = list(read_frames_exact(chapter_video, start_frame=50, return_indices=True))
    assert [index for _, index in frames] == [*range(50, 55), *range(60, 70)]
    # the non-reference frames before it, which the read skips, change none of its frames
    whole = list(read_frames_exact(chapter_video))
    assert all(np.array_equal(frame, whole[INDICES.index(index)]) for frame, index in frames)


def test_a_range_read_leaves_a_reused_container_decoding_every_frame(chapter_video):
    with open_video(chapter_video) as video:
        # past the last frame, the whole last chapter is the run up to the range, its non-reference frames skipped
        assert list(read_frames_exact(video, start_frame=75)) == []
        assert count_frames(video) == len(INDICES)


def test_batching_reads_the_frame_count_from_the_header(chapter_video, monkeypatch):
    # Matroska stores no frame count: its duration stands in, and nothing reads the whole file to count frames
    monkeypatch.setattr(frames_module, "video_metadata", lambda src: pytest.fail("counted the file's frames"))
    frames_module._count_and_rate_cached.cache_clear()
    assert read_frames_batched(chapter_video, start_frame=45, end_frame=54).shape == (10, 32, 32, 3)


def test_batching_frames_of_different_sizes_needs_a_size(chapter_video):
    with pytest.raises(ValueError, match="size="):
        read_frames_batched(chapter_video)
    frames, indices, payloads = read_frames_batched(chapter_video, size=(32, 32), return_indices=True,
                                                    return_chapters=True)
    assert frames.shape == (30, 32, 32, 3)
    assert indices.tolist() == INDICES
    assert [payload["start_frame"] for payload in payloads] == [30] * 10 + [45] * 10 + [60] * 10
    assert read_frames_batched(chapter_video, start_frame=45, end_frame=54).shape == (10, 32, 32, 3)


def test_a_video_without_chapters_has_no_payloads():
    payloads = [payload for _, payload in read_frames_exact(str(ASSETS / "example.mp4"), end_frame=4,
                                                            return_chapters=True)]
    assert payloads == [None] * 5


def test_size_applies_in_display_orientation():
    """rotated90.mp4 is stored 640x360 and shown 360x640: a size is the shown frame's."""
    path = str(ASSETS / "rotated90.mp4")
    full = next(read_frames_exact(path, end_frame=0))
    half = next(read_frames_exact(path, end_frame=0, size=(180, 320)))
    assert full.shape == (640, 360, 3)
    assert half.shape == (320, 180, 3)
    pooled = full.reshape(320, 2, 180, 2, 3).mean(axis=(1, 3))
    assert np.abs(half - pooled).mean() < 3


def test_size_is_validated_at_call_time():
    with pytest.raises(ValueError, match="size"):
        read_frames_exact("missing.mp4", size=(0, 10))
