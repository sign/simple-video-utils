import io
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import ASSETS, EXAMPLE, HELD, Pipe, make_video, numbers

from simple_video_utils import PackedChaptersError, Video, VideoMetadata, stack_frames, video_metadata


@pytest.fixture(scope="module")
def full():
    with Video(EXAMPLE) as video:
        return video.read().frames


def test_metadata():
    meta = video_metadata(EXAMPLE)
    assert meta == VideoMetadata(320, 240, 30.0, 130, "1/15360", pytest.approx(130 / 30), 0)
    assert list(meta._asdict()) == ["width", "height", "fps", "nb_frames", "time_base", "duration", "rotation"]
    assert video_metadata(EXAMPLE) is meta, "cached per path"
    assert video_metadata(Path(EXAMPLE)) == meta


@pytest.mark.parametrize("wrap", [bytes, io.BytesIO, Pipe])
def test_metadata_from_bytes_and_streams(wrap):
    with Video(wrap((ASSETS / "no_nb_frames.webm").read_bytes())) as video:
        # a stream that cannot seek gets the header's estimate; anything else reads the last packet
        assert video.meta.nb_frames == (108 if wrap is Pipe else 107)
        assert (video.meta.width, video.meta.height, video.meta.fps) == (640, 480, 25.0)


@pytest.mark.parametrize("src", [str(ASSETS / "missing.mp4"), b"not a video", str(ASSETS / "empty.mp4")])
def test_unreadable_video_raises_runtime_error(src):
    with pytest.raises(RuntimeError, match="Failed to open video"):
        Video(src)


def test_frame_count_survives_a_header_that_overcounts():
    with av.open(str(ASSETS / "buggy_mov_header.mov")) as container:
        assert container.streams.video[0].frames == 11
    with Video(ASSETS / "buggy_mov_header.mov") as video:
        assert video.meta.nb_frames == len(video.read(size=(64, 36)).frames) == 3


@pytest.mark.parametrize(("name", "count"), [("no_nb_frames.mkv", 37), ("no_nb_frames.webm", 107)])
def test_frame_count_without_a_header_count(name, count):
    with Video(ASSETS / name) as video:
        assert video.stream.frames == 0
        assert video.meta.nb_frames == count == sum(1 for _ in video.frames())


def test_rate_is_measured_when_the_header_has_none():
    data = (ASSETS / "mediarecorder-no-average-rate.webm").read_bytes()
    with Video(data) as video:
        assert not video.stream.average_rate
        assert video.meta.fps == pytest.approx(19.36, abs=0.01)
        assert video.meta.nb_frames == len(video.read().frames) == 45
    with Video(Pipe(data)) as video:  # measuring needs a rewind
        assert video.meta.fps == 0.0
        assert sum(1 for _ in video.frames()) == 45


def test_rotation_comes_from_the_header_and_frames_come_out_upright():
    rotated = str(ASSETS / "fsboard_rotated90.mp4")
    with av.open(rotated) as container:
        stored = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    with Video(rotated) as video:
        assert (video.meta.width, video.meta.height, video.meta.rotation) == (360, 640, 90)
        assert np.array_equal(video.read().frames, np.rot90(stored, axes=(1, 2)))
        assert video.read(0, 2, size=(90, 160)).frames.shape == (2, 160, 90, 3)


def test_reserved_colour_tags_do_not_break_the_conversion():
    with av.open(str(ASSETS / "bad_colorspace.mp4")) as container:
        assert container.streams.video[0].codec_context.colorspace == 3  # a value the H.264 spec reserves
    with Video(ASSETS / "bad_colorspace.mp4") as video:
        assert video.read(size=(64, 36)).frames.shape == (182, 36, 64, 3)


def test_read_is_the_stacked_frames(full):
    with Video(EXAMPLE) as video:
        assert full.shape == (130, 240, 320, 3)
        assert full.dtype == np.uint8
        assert np.array_equal(np.stack(list(video.frames())), full)
        assert np.array_equal(video.read(10, 20).frames, full[10:20])
        assert np.array_equal(video.read(np.int64(100)).frames, full[100:]), "past three seconds a read seeks"
        frames, indices = video.read(125, 500)
        assert indices.tolist() == [125, 126, 127, 128, 129]
        assert np.array_equal(frames, full[125:])


def test_read_by_seconds(full):
    with Video(EXAMPLE) as video:
        assert np.array_equal(video.read(1.0, 2.0, seconds=True).frames, full[30:60])
        assert np.array_equal(video.read(-0.5, 0.7, seconds=True).frames, full[:21]), "0.7 s is frame 21"
        assert np.array_equal(next(video.frames(video.meta.duration / 2, seconds=True)), full[65])


@pytest.mark.parametrize(("arguments", "error"), [
    ({"start": -1}, ValueError), ({"start": 5, "stop": 4}, ValueError), ({"start": 1.5}, TypeError),
    ({"start": 2.0, "stop": 1.0, "seconds": True}, ValueError), ({"fps": 0}, ValueError),
    ({"size": (0, 16)}, ValueError), ({"size": (16,)}, ValueError), ({"start": 5, "stop": 5}, ValueError),
])
def test_bad_arguments_raise_at_the_call(arguments, error):
    with Video(EXAMPLE) as video, pytest.raises(error):
        video.frames(**arguments) if arguments.get("stop") != 5 else video.read(**arguments)


def test_a_timed_read_goes_by_each_frames_own_time():
    # a browser recording: frames at 0, 57, 164, 213 ms. An average rate would put frame 3 inside [0.1, 0.2)
    with Video(ASSETS / "mediarecorder-no-average-rate.webm") as video:
        assert [index for _, index in video.frames(0.1, 0.2, seconds=True, indices=True)] == [2]
        assert video.read(0.164, 0.26, seconds=True).indices.tolist() == [2, 3]
    with av.open(str(ASSETS / "no_average_rate_5s.webm")) as container:
        times = [frame.time for frame in container.decode(video=0)]
    with Video(ASSETS / "no_average_rate_5s.webm") as video:  # far enough in to seek; 20 fps would say 80-89
        timed = video.read(4.0, 4.5, seconds=True).indices.tolist()
        assert timed == [index for index, time in enumerate(times) if 4.0 <= time < 4.5] == list(range(79, 89))


def test_a_stream_that_cannot_seek_is_read_once(tmp_path):
    data = Path(make_video(tmp_path / "v.mkv", frames=120)).read_bytes()
    with Video(Pipe(data)) as video:
        assert numbers(video.frames(100, 103)) == [100, 101, 102], "decoded from the start, with no seek"
        with pytest.raises(ValueError, match="read only once"):
            next(video.frames())


def test_seeked_read_is_exact_when_the_header_rate_drifts():
    # the header says 59.89 fps, the frames run at 60: placing frames by the header rate lands one off (#26)
    with Video(ASSETS / "vfr_dup_index.mp4") as video:
        whole = video.read().frames
        assert np.array_equal(video.read(400, 430).frames, whole[400:430])


def test_seeked_read_without_a_header_rate():
    with Video(ASSETS / "no_average_rate_5s.webm") as video:
        whole = video.read().frames
        assert len(whole) == video.meta.nb_frames == 100
        assert np.array_equal(video.read(80, 90).frames, whole[80:90])


def test_fps_drops_frames_and_never_repeats_them(full):
    with Video(EXAMPLE) as video:
        frames, indices = video.read(fps=10)
        assert indices.tolist() == list(range(0, 130, 3))
        assert np.array_equal(frames, full[::3])
        assert len(video.read(fps=120).frames) == 130
        assert video.read(40, 60, fps=15).indices.tolist() == list(range(40, 60, 2))


def test_keep_last_adds_the_last_frame_when_fps_would_drop_it(full):
    with Video(EXAMPLE) as video:
        assert video.read(0, 30, fps=10).indices.tolist() == list(range(0, 30, 3))
        batch = video.read(0, 30, fps=10, keep_last=True)
        assert batch.indices.tolist() == [*range(0, 30, 3), 29]
        assert np.array_equal(batch.frames, full[batch.indices])
        assert video.read(0, 28, fps=10, keep_last=True).indices.tolist() == list(range(0, 28, 3)), "27 was kept anyway"
        assert [index for _, index in video.frames(120, fps=10, keep_last=True, indices=True)] == [120, 123, 126, 129]
        with pytest.raises(ValueError, match="keep_last"):
            video.read(keep_last=True)


def test_keep_last_keeps_the_last_frame_of_each_chapter(gapped):
    with Video(gapped) as video:
        kept = video.read(fps=15, keep_last=True, size=(32, 32)).indices.tolist()
        assert kept == [*range(0, 20, 2), 19, *range(45, 80, 2), *range(100, 128, 2), 127]
        counts = [sum(1 for _ in frames) for _, frames in video.chapter_frames(fps=15, size=(32, 32), keep_last=True)]
        assert counts == [11, 18, 15]


def test_size_resizes_before_the_rgb_conversion():
    with av.open(EXAMPLE) as container:
        frame = next(container.decode(video=0))
        expected = frame.reformat(width=64, height=48, interpolation="AREA").to_ndarray(format="rgb24")
    with Video(EXAMPLE) as video:
        assert np.array_equal(video.read(0, 1, size=(64, 48)).frames[0], expected)


def test_one_video_serves_many_reads(full):
    with Video(EXAMPLE, thread_type="NONE") as video:
        assert video.meta.nb_frames == 130
        assert np.array_equal(video.read(5, 15).frames, full[5:15])
        assert video.keyframes() == [0]
        assert np.array_equal(video.read(0, 5).frames, full[:5])
        video.rewind()
        assert sum(1 for _ in video.container.decode(video.stream)) == 130


def test_frames_are_decoded_as_they_are_asked_for(tmp_path):
    path = make_video(tmp_path / "v.mkv", frames=120, noise=True)
    with Video(path) as video:
        whole = video.read().frames

    class Counting(Pipe):
        count = 0

        def readinto(self, buffer):
            self.count += len(buffer)
            return super().readinto(buffer)

    stream = Counting(Path(path).read_bytes())
    with Video(stream, buffer_size=4096) as video:
        frames = video.frames()
        assert np.array_equal([next(frames), next(frames)], whole[:2])
        assert stream.count < len(stream.data.getbuffer()) / 2


def test_keyframes_are_in_presentation_order(tmp_path):
    path = make_video(tmp_path / "v.mp4", frames=24, gop=8, bframes=2)
    with av.open(path) as container:
        truth = [index for index, frame in enumerate(container.decode(video=0)) if frame.key_frame]
    with Video(path) as video:
        assert video.keyframes() == truth == [0, 8, 16]


def test_stack_frames():
    frames = [np.full((2, 3, 3), i, np.uint8) for i in range(100)]
    for hint in (0, 1, 64, 99, 100, 101, 1000):
        batch = stack_frames(iter(frames), hint)
        assert batch.shape == (100, 2, 3, 3)
        assert batch.flags.owndata
        assert (batch[:, 0, 0, 0] == np.arange(100)).all()
    with pytest.raises(ValueError, match="no frames"):
        stack_frames(iter(()))
    with pytest.raises(ValueError, match="different shapes"):
        stack_frames([frames[0], np.zeros((3, 3, 3), np.uint8)])


def test_chapters_place_frames_on_the_source(gapped, full):
    with Video(gapped) as video:
        spans = [(c.payload["start_frame"], c.payload["end_frame"]) for c in video.chapters]
        assert spans == [(0, 20), (45, 80), (100, 128)]
        assert [float(c.source) for c in video.chapters] == pytest.approx([0, 1.5, 100 / 30])
        assert [float(c.start) for c in video.chapters] == pytest.approx([0, 1.5, 3.333])
        assert video.chapters[1].payload["payload"] == {"who": "signer"}
        assert video.meta.nb_frames == len(HELD), "the frames it holds, not the source's"
        frames, indices = video.read(size=(160, 128))
        assert indices.tolist() == HELD
        # the first chapter is the source's top-left corner
        assert np.abs(frames[:20].astype(int) - full[:20, :128, :160]).mean() < 4
        assert video.read(15, 50, size=(16, 16)).indices.tolist() == [*range(15, 20), *range(45, 50)]
        assert video.read(105, 110).indices.tolist() == list(range(105, 110)), "a seek lands in a chapter"
        assert video.read(0.6, 1.6, seconds=True, size=(16, 16)).indices.tolist() == [18, 19, 45, 46, 47]
        with pytest.raises(ValueError, match="different shapes"):
            video.read()


@pytest.mark.parametrize("wrap", [str, lambda path: Pipe(Path(path).read_bytes())])
def test_chapter_frames_reads_each_chapter_from_one_decode(gapped, wrap):
    with Video(wrap(gapped)) as video:
        chapters = [(chapter.id, [frame.shape for frame in frames]) for chapter, frames in video.chapter_frames()]
    assert chapters == [(1, [(128, 160, 3)] * 20), (2, [(208, 256, 3)] * 35), (3, [(240, 320, 3)] * 28)]


def test_fps_starts_over_in_each_chapter(gapped):
    with Video(gapped) as video:
        counts = [sum(1 for _ in frames) for _, frames in video.chapter_frames(fps=15, size=(32, 32))]
        assert counts == [10, 18, 14]
        kept = video.read(fps=15, size=(32, 32)).indices.tolist()
        assert kept == [*range(0, 20, 2), *range(45, 80, 2), *range(100, 128, 2)]


def test_packed_chapters_are_refused(tmp_path):
    # before 0.10 chapters played back to back: here source frames 30-59 and 90-119 sit at 0-1 s and 1-2 s
    path = tmp_path / "packed.mkv"
    with av.open(EXAMPLE) as source, av.open(str(path), mode="w") as output:
        output.set_chapters([{"id": i, "start": i * 1000, "end": (i + 1) * 1000, "time_base": Fraction(1, 1000),
                              "metadata": {"payload": json.dumps({"start_frame": start, "end_frame": start + 30})}}
                             for i, start in enumerate((30, 90))])
        track = output.add_stream_from_template(source.streams.video[0])
        for packet in source.demux(video=0):
            if packet.size:
                packet.stream = track
                output.mux(packet)
    with Video(path) as video, pytest.raises(PackedChaptersError, match="packed"):
        video.read()
    with pytest.raises(PackedChaptersError):
        video_metadata(str(path))
    assert issubclass(PackedChaptersError, ValueError)


def test_a_video_without_our_chapters_has_none(tmp_path):
    path = tmp_path / "marked.mkv"
    with av.open(EXAMPLE) as source, av.open(str(path), mode="w") as output:
        output.set_chapters([{"id": i, "start": i * 2000, "end": i * 2000 + 2000, "time_base": Fraction(1, 1000),
                              "metadata": {"title": title}} for i, title in enumerate(("Intro", "1", "[2]"))])
        track = output.add_stream_from_template(source.streams.video[0])
        for packet in source.demux(video=0):
            if packet.size:
                packet.stream = track
                output.mux(packet)
    with Video(path) as video:
        assert len(video.container.chapters()) == 3
        assert video.chapters == []
        assert [(chapter, sum(1 for _ in frames)) for chapter, frames in video.chapter_frames()] == [(None, 130)]
