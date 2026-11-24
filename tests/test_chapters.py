import errno
import io
from fractions import Fraction

import av
import numpy as np
import pytest
from av.video.reformatter import ColorRange
from conftest import ASSETS, EXAMPLE, Pipe, make_video, windows

from simple_video_utils import ChapterClip, Video, VideoSlice, chapters, write_chapters


def planes(path, frames: range) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The (Y, U, V) planes of those decoded frames, as stored."""
    out = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in frames:
                height, width = frame.height, frame.width
                packed = frame.to_ndarray()
                chroma = packed[height:].reshape(2, height // 2, width // 2)
                out.append((packed[:height], chroma[0], chroma[1]))
    return out


def write(tmp_path, src, clips, **options):
    out = tmp_path / "out.mkv"
    return out, write_chapters(src, out, clips, preset="ultrafast", **options)


def test_a_crop_keeps_the_source_pixels(tmp_path):
    clips = [ChapterClip(3, 9, (64, 32, 224, 160)), ChapterClip(40, 44, (0, 0, 320, 240))]
    out, written = write(tmp_path, EXAMPLE, clips, crf=0)
    assert written == clips
    pairs = zip(planes(out, range(6)), planes(EXAMPLE, range(3, 9)), strict=True)
    for (y, u, v), (source_y, source_u, source_v) in pairs:
        assert np.array_equal(y, source_y[32:160, 64:224])
        assert np.array_equal(u, source_u[16:80, 32:112])
        assert np.array_equal(v, source_v[16:80, 32:112])
    with av.open(str(out)) as container:  # each frame sits at its source time
        times = [packet.pts * packet.time_base for packet in container.demux(video=0) if packet.size]
        assert [round(time * 30) for time in times] == [*range(3, 9), *range(40, 44)]


def near(a, b) -> bool:
    """Equal but for the dither of the 16-bit conversion an off-grid crop goes through."""
    return bool((np.abs(a.astype(int) - b) <= 1).all())


def test_a_box_past_the_frame_pads_with_black_even_off_the_chroma_grid(tmp_path):
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(0, 2, (-13, -7, 163, 137))], crf=0)
    [(y, _, _)], [(source_y, _, _)] = planes(out, range(1)), planes(EXAMPLE, range(1))
    assert y.shape == (144, 176)
    assert near(y[:7], 16)
    assert near(y[:, :13], 16)
    assert near(y[7:, 13:], source_y[:137, :163])


def test_a_full_range_source_stays_full_range(tmp_path):
    source = make_video(tmp_path / "full.mp4", frames=1, noise=True, pix_fmt="yuvj420p")
    out, _ = write(tmp_path, source, [ChapterClip(0, 1, (-8, -8, 56, 40))], crf=0)
    with av.open(str(out)) as container:
        assert next(container.decode(video=0)).color_range == ColorRange.JPEG
    [(y, _, _)], [(source_y, _, _)] = planes(out, range(1)), planes(source, range(1))
    assert near(y[:8], 0), "black is 0 in full range, 16 in limited"
    assert near(y[:, :8], 0)
    assert near(y[8:, 8:], source_y[:40, :56])


def test_a_reserved_colour_tag_is_read_after_the_first_frame(tmp_path):
    # the header says colorspace 3, which the scale filter refuses; decoding a frame corrects it to 2
    out, _ = write(tmp_path, ASSETS / "bad_colorspace.mp4", [ChapterClip(0, 2, (0, 0, 64, 64))])
    with Video(out) as video:
        assert len(video.read().frames) == 2


def test_a_rotated_source_is_cut_as_it_is_displayed(tmp_path):
    rotated = ASSETS / "fsboard_rotated90.mp4"
    out, _ = write(tmp_path, rotated, [ChapterClip(2, 6, (40, 160, 200, 480))], crf=0)
    with Video(rotated) as source, Video(out) as video:
        assert video.meta.rotation == 0
        frames = video.read().frames
        assert frames.shape == (4, 320, 160, 3)
        assert np.abs(frames.astype(int) - source.read(2, 6).frames[:, 160:480, 40:200]).mean() < 1


def test_chapters_that_touch_read_back(tmp_path):
    # at 30 fps the boundary is 33.3 ms, which Matroska stores as 33
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(0, 1, (0, 0, 160, 128)), ChapterClip(1, 10, (0, 0, 160, 128))])
    with Video(out) as video:
        assert [sum(1 for _ in frames) for _, frames in video.chapter_frames()] == [1, 9]
    clip = VideoSlice(out, 0, 1).read()
    with Video(clip) as video:  # and the MP4 a slice writes, whose chapter ends are rebuilt from frame counts
        assert video.read().indices.tolist() == list(range(10))


def test_a_timed_read_finds_every_frame_of_a_chapter(tmp_path):
    # frame 47 is at 1566.67 ms and its chapter starts at 1567; frame 49 is at 1633.33 and is stored at 1633
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(47, 80, (0, 0, 160, 128))])
    with Video(out) as video:
        assert windows(video, range(47, 80)) == [[index] for index in range(47, 80)]


def test_a_slice_carries_many_short_chapters(tmp_path):
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(n, n + 1, (0, 0, 16, 16)) for n in range(0, 128, 2)])
    clip = VideoSlice(out, 0, 5)  # every chapter mark needs room in the MP4 header, reserved before it is written
    with Video(clip.read()) as video:
        assert video.read().indices.tolist() == list(range(0, 128, 2))


def test_a_chapter_file_states_its_sources_rate(tmp_path):
    # Matroska keeps a rate as whole nanoseconds per frame, which libav hands back as 60/1 here
    rate = Fraction(1600000, 26667)
    source = make_video(tmp_path / "source.mp4", frames=12, fps=rate)
    out, _ = write(tmp_path, source, [ChapterClip(2, 8, (0, 0, 64, 48))])
    with Video(out) as video:
        assert video.stream.average_rate == 60
        assert video.rate == rate
        assert ChapterClip(**video.chapters[0].payload).end_frame == 8, "the payload is still the clip, and only that"


def test_a_chaptered_slice_is_cut_at_the_rate_its_chapters_state(tmp_path, gapped):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(VideoSlice(gapped, 0.5, 3.5).read())  # 45 frames over 3 seconds: its header says 120/7 fps
    out, _ = write(tmp_path, clip, [ChapterClip(0, 5, (0, 0, 160, 128))])
    with Video(out) as video, av.open(str(out)) as container:
        assert video.meta.fps == 30
        assert [packet.pts for packet in container.demux(video=0) if packet.size] == [0, 33, 67, 100, 133]


def test_resolution_resizes_a_chapter(tmp_path):
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(0, 3, (0, 0, 320, 240), resolution=(160, 112))])
    with Video(out) as video:
        assert video.read().frames.shape == (3, 112, 160, 3)
        assert video.chapters[0].payload["resolution"] == [160, 112]


def test_clips_are_cut_at_the_last_frame_the_source_decodes(tmp_path):
    clips = [ChapterClip(100, 140, (0, 0, 160, 128)), ChapterClip(150, 160, (0, 0, 16, 16))]
    out, written = write(tmp_path, EXAMPLE, clips)
    assert written == [ChapterClip(100, 130, (0, 0, 160, 128))]
    with Video(out) as video:
        assert [c.frames for c in video.chapters] == [range(100, 130)]
        assert video.read().indices.tolist() == list(range(100, 130))


@pytest.mark.parametrize(("fps", "clips"), [
    (30, [ChapterClip(100, 104, (0, 0, 64, 48))]),  # libav sets no start time on a file of under 8 frames
    (60, [ChapterClip(6 * i + 1, 6 * i + 5, (0, 0, 64, 48)) for i in range(14)]),  # 13 gaps, each rounded to a ms
])
def test_frame_count_is_what_the_chapters_hold(tmp_path, fps, clips):
    out, _ = write(tmp_path, make_video(tmp_path / "source.mp4", frames=110, fps=fps), clips)
    for source in (str(out), Pipe(out.read_bytes())):  # a pipe has the header alone, whose clock runs through the gaps
        with Video(source) as video:
            assert video.meta.nb_frames == sum(clip.end_frame - clip.start_frame for clip in clips)


def test_a_source_without_a_header_rate(tmp_path):
    out, _ = write(tmp_path, ASSETS / "no_average_rate_5s.webm", [ChapterClip(4, 30, (0, 0, 96, 96))])
    with Video(out) as video:
        assert video.meta.fps == pytest.approx(19.8)
        assert len(video.read().frames) == 26


@pytest.mark.parametrize(("clips", "error"), [
    ([], "at least one clip"),
    ([ChapterClip(5, 9, (0, 0, 16, 16)), ChapterClip(8, 12, (0, 0, 16, 16))], "must not overlap"),
    ([ChapterClip(0, 4, (0, 0, 20, 16))], "multiples of 16"),
    ([ChapterClip(4, 4, (0, 0, 16, 16))], "start_frame < end_frame"),
    ([ChapterClip(500, 510, (0, 0, 16, 16))], "every clip starts past"),
])
def test_bad_clips_raise_and_leave_no_file(tmp_path, clips, error):
    with pytest.raises(ValueError, match=error):
        write(tmp_path, EXAMPLE, clips)
    assert not list(tmp_path.iterdir())


def test_an_existing_output_is_left_alone(tmp_path):
    out = tmp_path / "out.mkv"
    out.write_bytes(b"mine")
    with pytest.raises(FileExistsError):
        write_chapters(EXAMPLE, out, [ChapterClip(0, 4, (0, 0, 16, 16))])
    assert out.read_bytes() == b"mine"


def test_a_write_the_device_refuses_leaves_no_file(tmp_path, monkeypatch):
    class Full(io.FileIO):
        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

    # closing the file flushes it, so the close in the cleanup is refused as well
    monkeypatch.setattr(chapters, "open", lambda path, mode: io.BufferedWriter(Full(path, "x")), raising=False)
    with pytest.raises(OSError, match="No space left"):
        write(tmp_path, EXAMPLE, [ChapterClip(0, 4, (0, 0, 16, 16))])
    assert not list(tmp_path.iterdir())
