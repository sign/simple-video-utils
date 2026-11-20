import io
import struct
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import ASSETS, EXAMPLE, Pipe, make_video, numbers, windows

from simple_video_utils import Video, VideoSlice


@pytest.fixture
def source(tmp_path):
    return make_video(tmp_path / "source.mp4", frames=120)


def read(src):
    with Video(src) as video:
        return video.read().frames


def test_slice_copies_the_frames_between_start_and_stop(source):
    whole = read(source)
    clip = VideoSlice(source, 1.0, 2.5).read()
    assert np.array_equal(read(clip), whole[30:75])
    with av.open(io.BytesIO(clip)) as container:
        packets = [packet for packet in container.demux(video=0) if packet.size]
        # frame 30 is no keyframe: the clip starts at keyframe 24, and an edit list hides what precedes frame 30
        assert len(packets) == 51
        assert packets[0].is_keyframe
        assert [p.pts for p in packets].index(0) == 6
        assert container.streams.video[0].codec_context.name == "h264"
    with Video(clip) as video:
        assert video.keyframes() == [6, 18, 30, 42], "counted in the frames a read returns, not the hidden ones"


@pytest.mark.parametrize("suffix", ["mp4", "mkv"])
def test_consecutive_slices_share_no_frame(tmp_path, suffix):
    source = make_video(tmp_path / f"source.{suffix}", frames=120)  # Matroska stores 1/30 s as 33 ms, 2/30 as 67
    cuts = [0, 1 / 30, 2 / 30, 1, 61 / 30, 9]
    assert [numbers(read(VideoSlice(source, a, b).read())) for a, b in pairwise(cuts)] == [
        [0], [1], list(range(2, 30)), list(range(30, 61)), list(range(61, 120))]


def test_a_slice_keeps_its_sources_clock_and_reads_like_it(tmp_path):
    source = make_video(tmp_path / "source.mkv", frames=120)  # 0, 33, 67 ms: a slice from 67 holds 0, 33, 66
    clip = VideoSlice(source, 2 / 30, 3.0).read()
    with Video(clip) as video:
        assert video.stream.time_base == Fraction(1, 1000)
        assert numbers(video.read(fps=15).frames) == [*range(2, 90, 2), 89]
        assert len(video.read(fps=30).frames) == 88, "the source's own rate drops nothing"
        assert windows(video, range(88)) == [[index] for index in range(88)]
    assert numbers(read(VideoSlice(clip, 2 / 30, 10 / 30).read())) == list(range(4, 12))


def test_a_clock_that_ticks_once_a_frame_gets_half_a_frame_of_slack(tmp_path):
    # AVI at 29.97 fps: frame 28 is stamped 0.968 s and frame 29 1.001 s, and a whole tick of slack would be a frame
    source = make_video(tmp_path / "ntsc.avi", frames=90, fps=Fraction(30000, 1001))
    with Video(source) as video:
        assert video.read(1.0, 2.0, seconds=True).indices.tolist() == list(range(29, 59))
    assert numbers(read(VideoSlice(source, 1.0, 2.0).read())) == list(range(29, 59))


def test_a_chaptered_slice_reads_like_its_file(gapped):
    with Video(VideoSlice(gapped, 0.5, 3.5).read()) as video:  # starts on frame 15, in the middle of a chapter
        held = [*range(15, 20), *range(45, 80), *range(100, 105)]
        assert windows(video, held) == [[index] for index in held]
        assert video.read(fps=15, size=(16, 16)).indices.tolist() == [15, 17, 19, *range(45, 80, 2), 100, 102, 104]


def test_slice_keeps_an_hvc1_sample_entry():
    clip = VideoSlice(ASSETS / "buggy_mov_header.mov", 0.0, 0.3).read()  # an iPhone recording, HEVC
    assert clip[clip.index(b"stsd") + 16:][:4] == b"hvc1", "the muxer's own choice, hev1, does not play on Apple's"


def test_slice_needs_a_keyframe_to_start_from(tmp_path):
    source = make_video(tmp_path / "source.ts", frames=90, gop=30)  # MPEG-TS seeks to a packet, not a keyframe
    with pytest.raises(ValueError, match="no keyframe"):
        _ = VideoSlice(source, 1.5, 2.5).size


def test_slice_starting_on_the_frames_shown_before_a_keyframe(tmp_path):
    # an open GOP: frames 27-29 are decoded after keyframe 30, and predicted from frame 26 too
    source = make_video(tmp_path / "open.mp4", gop=30, bframes=3, x264=":open-gop=1")
    assert numbers(read(VideoSlice(source, 28 / 30, 40 / 30).read()))[:12] == list(range(28, 40))


@pytest.mark.skipif("libsvtav1" not in av.codecs_available, reason="this PyAV build cannot encode AV1")
def test_slice_copies_a_codec_whose_decoder_is_no_encoder(tmp_path):
    source = str(tmp_path / "av1.mp4")
    with av.open(source, mode="w") as output:
        stream = output.add_stream("libsvtav1", rate=30, options={"g": "30"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(60):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 4 * i, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    # PyAV's wheels decode AV1 with libdav1d, and no encoder has that name: copying packets must not need one
    assert np.array_equal(read(VideoSlice(source, 1.0, 1.5).read()), read(source)[30:45])


def test_slice_streams_as_it_is_read(tmp_path):
    source = make_video(tmp_path / "noise.mp4", frames=120, noise=True)
    whole = read(source)
    clip = VideoSlice(source, 0.4, 3.9)
    assert clip.duration == pytest.approx(3.5)
    assert clip.size > 4 * 64 * 1024, "measured without keeping any media"
    chunks = list(clip)
    data = b"".join(chunks)
    assert len(chunks) > 4
    assert max(map(len, chunks)) < 2 * 64 * 1024
    assert len(data) == clip.size
    assert data == clip.read(), "replays byte for byte"
    assert data.index(b"moov") < data.index(b"mdat")
    with Video(Pipe(data)) as video:  # moov first: a reader needs no seeking
        assert np.array_equal(video.read().frames, whole[12:117])


def test_slice_of_bytes_and_of_b_frames(tmp_path):
    data = Path(make_video(tmp_path / "b.mkv", frames=90, bframes=2)).read_bytes()
    whole = read(data)
    clip = read(VideoSlice(data, 1.2, 2.2).read())
    # Matroska drops decode timestamps after a seek; frame 65 is predicted from the frame at `stop`, which remains
    assert np.array_equal(clip, whole[36:67])


def test_b_frames_leave_later_frames_visible_in_a_plain_slice(tmp_path):
    source = make_video(tmp_path / "b.mp4", frames=120, gop=30, bframes=3)
    shown = numbers(read(VideoSlice(source, 0, 4 / 30).read()))
    # frames 0-3 are predicted from frames 4 and 6, which are copied with them and cannot be hidden
    assert shown[:4] == [0, 1, 2, 3]
    assert shown[4:] == [4, 6]


def test_slice_times_are_matched_on_the_files_clock(gapped):
    # frame 100 of 30 fps is at 3.3333 s, stored as 3333 ms: a slice from 100 / 30 must still start on it
    with Video(VideoSlice(gapped, 100 / 30, 110 / 30).read()) as video:
        assert video.read().indices.tolist() == list(range(100, 110))


def test_slice_keeps_gaps_and_carries_its_chapters(gapped):
    with Video(gapped) as video:
        frames, indices = video.read(size=(64, 64))
        held = dict(zip(indices.tolist(), frames, strict=True))
    clip = VideoSlice(gapped, 0.5, 3.5)
    data = clip.read()
    assert clip.duration == pytest.approx(3.0, abs=0.01)
    with Video(data) as video:
        assert [(float(c.start), float(c.end), float(c.source)) for c in video.chapters] == [
            pytest.approx(times, abs=0.001) for times in [(0, 5 / 30, 0.5), (1, 65 / 30, 1.5), (85 / 30, 3, 100 / 30)]]
        assert video.meta.fps == 30
        assert video.meta.nb_frames == 45
        frames, indices = video.read(size=(64, 64))
        assert indices.tolist() == [*range(15, 20), *range(45, 80), *range(100, 105)], "source frame numbers"
        assert all(np.array_equal(frame, held[index]) for frame, index in zip(frames, indices.tolist(), strict=True))
        assert video.read(60, 70, size=(64, 64)).indices.tolist() == list(range(60, 70))
    again = VideoSlice(data, 0.1, 1.2).read()
    with Video(again) as video:  # a slice of a slice is still placed on the original source
        assert video.read(size=(64, 64)).indices.tolist() == [18, 19, *range(45, 51)]


def test_frame_count_leaves_out_the_frames_a_slice_hides(source, gapped):
    # from frame 13 one frame is hidden, and duration x fps is within one of the header's 48. From frame 65 of the
    # gapped file 20 are hidden (the chapter's keyframe is 45), as many as the gap it spans holds none
    for clip, count in ((VideoSlice(source, 13 / 30, 2.0), 47), (VideoSlice(gapped, 65 / 30, 110 / 30), 25)):
        for wrap in (bytes, Pipe):
            with Video(wrap(clip.read())) as video:
                assert video.meta.nb_frames == count == sum(1 for _ in video.frames())


def test_slice_header_describes_the_chapter_it_starts_in(gapped):
    # the file's one header is its first chapter's, 160x128; Apple's players refuse a clip whose frames contradict it
    def declared(clip: bytes) -> tuple[int, int]:
        entry = clip.index(b"avc1", clip.index(b"stsd"))
        return struct.unpack(">HH", clip[entry + 28:entry + 32])

    assert declared(VideoSlice(gapped, 0.1, 0.5).read()) == (160, 128)
    third = VideoSlice(gapped, 100 / 30, 110 / 30).read()
    assert declared(third) == (320, 240)
    assert read(third).shape == (10, 240, 320, 3)
    assert third == VideoSlice(gapped, 100 / 30, 110 / 30).read(), "and it still replays byte for byte"


def test_slice_starting_in_a_gap_starts_on_its_first_frame(gapped):
    data = VideoSlice(gapped, 3.0, 4.0).read()  # the third chapter starts at 3.333
    with av.open(io.BytesIO(data)) as container:
        packets = [packet for packet in container.demux(video=0) if packet.size]
        assert packets[0].pts == 0, "nothing from the chapter before the gap"
        assert len(packets) == 20
    with Video(data) as video:
        assert video.read().indices.tolist() == list(range(100, 120))


@pytest.mark.parametrize(("start", "stop", "error"), [
    (9.0, 10.0, "no frames"), (2.0, 1.0, "out of range"), (-1.0, 1.0, "out of range"),
    (0.0, float("inf"), "out of range"),
])
def test_bad_slices_raise(source, start, stop, error):
    with pytest.raises(ValueError, match=error):
        next(iter(VideoSlice(source, start, stop)))


def test_slice_in_a_gap_has_no_frames(gapped):
    with pytest.raises(ValueError, match="no frames"):
        _ = VideoSlice(gapped, 0.7, 1.4).size


def test_slice_needs_a_source_it_can_reopen():
    with open(EXAMPLE, "rb") as file, pytest.raises(TypeError, match="reopens"):
        VideoSlice(file, 0, 1)


def test_slice_notices_a_source_that_changed(tmp_path, source):
    clip = VideoSlice(source, 1.0, 3.0)
    assert clip.size
    make_video(source, frames=40)
    with pytest.raises(OSError, match="changed"):
        clip.read()
