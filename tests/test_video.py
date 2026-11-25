import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import warnings
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.bitstream import BitStreamFilterContext
from conftest import ASSETS, EXAMPLE, HELD, Pipe, make_video, numbers, windows

from simple_video_utils import (
    InvalidVideoError,
    PackedChaptersError,
    Video,
    VideoError,
    VideoMetadata,
    stack_frames,
    unpack_chapters,
    video_metadata,
    write_video,
)
from simple_video_utils.chapters import _replace_chapters


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
    assert video_metadata(next(entry for entry in os.scandir(ASSETS) if entry.name == "example.mp4")) == meta


@pytest.mark.parametrize("wrap", [bytes, io.BytesIO, Pipe])
def test_metadata_from_bytes_and_streams(wrap):
    with Video(wrap((ASSETS / "no_nb_frames.webm").read_bytes())) as video:
        # a stream that cannot seek gets the header's estimate; anything else reads the last packet
        assert video.meta.nb_frames == (108 if wrap is Pipe else 107)
        assert (video.meta.width, video.meta.height, video.meta.fps) == (640, 480, 25.0)


@pytest.mark.parametrize("src", [str(ASSETS / "missing.mp4"), b"not a video", str(ASSETS / "empty.mp4")])
def test_unreadable_video_raises_runtime_error(src):
    with pytest.raises(RuntimeError, match="Failed to open video") as error:
        Video(src)
    # a file that is missing is no fault of its data, so only the other two are invalid
    assert isinstance(error.value, VideoError)
    assert isinstance(error.value, InvalidVideoError) == ("missing" not in str(src))
    assert av.logging.get_level() is None, "every open switches PyAV's log callback off: it can deadlock a close"


def test_gray_frames_and_a_named_container(tmp_path):
    with Video(EXAMPLE) as video:
        gray, colour = video.read(0, 3, format="gray").frames, video.read(0, 3).frames
        assert gray.shape == colour.shape[:3]
        assert np.abs(gray - colour.mean(axis=3)).max() < 16
    with Video(EXAMPLE, container_options={"probesize": "4096"}) as video:
        assert video.container.options == {"probesize": "4096"}
    with pytest.raises(InvalidVideoError):  # not what the file is, so it ends before a stream is found
        Video(EXAMPLE, format="mpegts")


def test_frame_count_survives_a_header_that_overcounts():
    with av.open(str(ASSETS / "buggy_mov_header.mov")) as container:
        assert container.streams.video[0].frames == 11
    with Video(ASSETS / "buggy_mov_header.mov") as video:
        assert video.meta.nb_frames == len(video.read(size=(64, 36)).frames) == 3
    with Video(Pipe((ASSETS / "buggy_mov_header.mov").read_bytes())) as video:
        assert video.meta.nb_frames == 3, "its edit list hides 8 samples: the index says so, nothing is decoded"


def test_frame_count_is_decoded_when_the_header_and_the_clock_disagree(tmp_path):
    data = bytearray(Path(make_video(tmp_path / "v.mp4", frames=60)).read_bytes())
    duration = data.index(b"mdhd") + 20  # the track's length: make it say half of what its samples add up to
    data[duration:duration + 4] = struct.pack(">I", struct.unpack_from(">I", data, duration)[0] // 2)
    with Video(bytes(data)) as video:
        assert (video.stream.frames, video.duration) == (60, 1.0)
        assert video.meta.nb_frames == len(video.read().frames) == 60


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


@pytest.mark.parametrize("degrees", [90, 180])
def test_an_orientation_only_in_the_h264_stream_turns_no_frame(tmp_path, degrees):
    # no display matrix in the header, and libav hands the SEI's rotation to the frames that carry it: here the first
    path = tmp_path / "sei.mp4"
    with av.open(EXAMPLE) as source, av.open(str(path), mode="w") as output:
        stream = source.streams.video[0]
        track = output.add_stream_from_template(stream)
        tag = BitStreamFilterContext(f"h264_metadata=display_orientation=insert:rotate={degrees}", stream, track)
        for packet in source.demux(stream):
            for tagged in tag.filter(packet if packet.size else None):
                tagged.stream = track
                output.mux(tagged)
    with av.open(str(path)) as container:
        assert [frame.rotation % 360 for frame in container.decode(video=0)][:3] == [degrees, 0, 0]
    with Video(path) as video, Video(EXAMPLE) as plain:
        assert (video.meta.width, video.meta.height, video.meta.rotation) == (320, 240, 0)
        assert np.array_equal(video.read().frames, plain.read().frames)


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
        assert np.array_equal(video.read(3.5, 4.0, seconds=True).frames, full[105:120]), "this far in, after a seek"
        frame, index = next(video.frames(video.meta.duration / 2, seconds=True, indices=True))
        assert index == 65
        assert np.array_equal(frame, full[65])


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
    with Video(ASSETS / "no_average_rate_5s.webm") as video:  # counted, as it has no rate; 20 fps would say 80-89
        timed = video.read(4.0, 4.5, seconds=True).indices.tolist()
        assert timed == [index for index, time in enumerate(times) if 4.0 <= time < 4.5] == list(range(79, 89))


def test_a_timed_read_allows_for_the_rounding_of_the_files_clock(tmp_path):
    # Matroska keeps milliseconds, and this file's first frame is stored at 67 ms: every time is read against it
    with Video(make_video(tmp_path / "late.mkv", frames=60, times=range(2, 62))) as video:
        assert video.stream.start_time == 67
        assert windows(video, range(60)) == [[index] for index in range(60)]
        _, index = next(video.frames(0.034, seconds=True, indices=True))
        assert index == 2, "a millisecond after a stored time is after it"


def test_a_stream_that_cannot_seek_is_read_once(tmp_path):
    data = Path(make_video(tmp_path / "v.mkv", frames=120)).read_bytes()
    with Video(Pipe(data)) as video:
        frames = list(video.frames(100, 103, indices=True))  # decoded from the start, with no seek
        assert [index for _, index in frames] == numbers(frame for frame, _ in frames) == [100, 101, 102]
        with pytest.raises(ValueError, match="read only once"):
            next(video.frames())


def test_reads_stay_exact_after_the_frame_rate_changes(tmp_path):
    # 160 frames at 60 fps, then 40 at 10 fps: no single rate says which frame sits at a given time
    times = [10 * i for i in range(160)] + [1600 + 60 * i for i in range(40)]
    with Video(make_video(tmp_path / "vfr.mp4", frames=200, fps=600, times=times)) as video:
        whole = video.read()
        for late in (video.read(5.0, 5.5, seconds=True), video.read(184, 189)):
            assert late.indices.tolist() == [184, 185, 186, 187, 188]
            assert np.array_equal(late.frames, whole.frames[184:189])


def test_a_timed_read_needs_no_frame_rate():
    with Video(Pipe((ASSETS / "mediarecorder-no-average-rate.webm").read_bytes())) as video:
        assert video.meta.fps == 0.0
        assert [index for _, index in video.frames(0.1, 0.2, seconds=True, indices=True)] == [2]
    with Video(Pipe((ASSETS / "mediarecorder-no-average-rate.webm").read_bytes())) as video:
        assert [index for _, index in video.frames(10, 14, indices=True)] == [10, 11, 12, 13]  # nor by index


def test_seeked_read_is_exact_when_the_header_rate_drifts():
    # the header says 59.89 fps, the frames run at 60: placing frames by the header rate lands one off (#26)
    with Video(ASSETS / "vfr_dup_index.mp4") as video:
        whole = video.read().frames
        assert np.array_equal(video.read(400, 430).frames, whole[400:430])
        # frames 499 and 500 are two frame times apart: the rates agree within 1%, the places after them do not
        assert np.array_equal(video.read(540, 549).frames, whole[540:549])


@pytest.mark.parametrize("extension", ["ts", "avi"])
def test_a_container_that_does_not_seek_to_a_frame_is_counted(tmp_path, extension):
    # MPEG-TS resumes at the keyframe after a seek's target, and H.264 in AVI starts a frame late
    with Video(make_video(tmp_path / f"v.{extension}", frames=240, gop=60)) as video:
        batch = video.read(97, 100)
        assert batch.indices.tolist() == numbers(batch.frames) == [97, 98, 99]


def test_a_read_far_into_a_file_without_a_header_rate_is_counted():
    with Video(ASSETS / "no_average_rate_5s.webm") as video:
        whole = video.read().frames
        assert len(whole) == video.meta.nb_frames == 100
        assert np.array_equal(video.read(80, 90).frames, whole[80:90])


def test_a_read_up_to_a_stop_measures_no_rate(monkeypatch):
    monkeypatch.setattr(Video, "_decoded", property(lambda video: pytest.fail("decoded the file to measure a rate")))
    with Video(ASSETS / "mediarecorder-no-average-rate.webm") as video:
        assert video.read(10, 14).indices.tolist() == [10, 11, 12, 13]
        assert video.read(0.1, 0.2, seconds=True).indices.tolist() == [2]


def test_stop_may_be_far_past_the_end():
    data = (ASSETS / "mediarecorder-no-average-rate.webm").read_bytes()  # nothing in it says how many frames it holds
    for source in (data, Pipe(data)):
        with Video(source) as video:
            assert len(video.read(0, sys.maxsize).frames) == 45, "`stop` must not size the batch"
    with Video(EXAMPLE) as video:
        assert len(video.read(1.0, math.inf, seconds=True).frames) == 100


def test_fps_drops_frames_never_repeats_them_and_keeps_the_last(full):
    with Video(EXAMPLE) as video:
        frames, indices = video.read(fps=10)
        assert indices.tolist() == list(range(0, 130, 3))
        assert np.array_equal(frames, full[::3])
        assert len(video.read(fps=120).frames) == 130
        batch = video.read(0, 30, fps=10)
        assert batch.indices.tolist() == [*range(0, 30, 3), 29], "the last frame, which the step alone would drop"
        assert np.array_equal(batch.frames, full[batch.indices])
        assert video.read(0, 30, fps=10, keep_last=False).indices.tolist() == list(range(0, 30, 3))
        assert video.read(0, 28, fps=10).indices.tolist() == list(range(0, 28, 3)), "27 is on the step"
        assert [index for _, index in video.frames(120, fps=10, indices=True)] == [120, 123, 126, 129]


def test_size_resizes_before_the_rgb_conversion():
    with av.open(EXAMPLE) as container:
        frame = next(container.decode(video=0))
        expected = frame.reformat(width=64, height=48, interpolation="AREA").to_ndarray(format="rgb24")
    with Video(EXAMPLE) as video:
        assert np.array_equal(video.read(0, 1, size=(64, 48)).frames[0], expected)


def test_one_video_serves_many_reads(full):
    with Video(EXAMPLE, thread_type="NONE") as video:
        assert video.stream.codec_context.thread_count == 1, "some decoders (AV1's) thread by the count alone"
        assert video.meta.nb_frames == 130
        assert np.array_equal(video.read(5, 15).frames, full[5:15])
        assert video.keyframes() == [0]
        assert np.array_equal(video.read(0, 5).frames, full[:5])
        video.rewind()
        assert sum(1 for _ in video.container.decode(video.stream)) == 130


@pytest.mark.parametrize("interrupt", [
    lambda video: video.meta,  # Matroska has no frame count in its header, so this reads the end of the file
    lambda video: video.read(100, 102), Video.keyframes, lambda video: next(video.frames(60)), Video.close,
])
def test_a_read_that_lost_its_place_raises(tmp_path, interrupt):
    with Video(make_video(tmp_path / "v.mkv", frames=120)) as video:
        frames, later = video.frames(indices=True), video.frames(0, 3)
        assert next(frames)[1] == 0
        interrupt(video)
        with pytest.raises(RuntimeError, match="before this read finished"):
            next(frames)
        if interrupt is not Video.close:
            assert numbers(later) == [0, 1, 2], "a read started afterwards is whole"


def test_frames_are_decoded_as_they_are_asked_for(tmp_path):
    path = make_video(tmp_path / "v.mkv", frames=120, noise=True)
    with Video(path) as video:
        whole = video.read().frames

    stream = Pipe(Path(path).read_bytes())
    with Video(stream, buffer_size=4096) as video:
        frames = video.frames()
        assert np.array_equal([next(frames), next(frames)], whole[:2])
        assert stream.tell() < len(stream.getbuffer()) / 2


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
    for count in (1, 64):  # 64 fills the first chunk: the next one must not take the new shape
        with pytest.raises(ValueError, match="different shapes"):
            stack_frames([*frames[:count], np.zeros((3, 3, 3), np.uint8)])


def test_chapters_place_frames_on_the_source(gapped, full):
    with Video(gapped) as video:
        assert [c.frames for c in video.chapters] == [range(0, 20), range(45, 80), range(100, 128)]
        assert [float(c.source) for c in video.chapters] == pytest.approx([0, 1.5, 100 / 30])
        assert [float(c.start) for c in video.chapters] == pytest.approx([0, 1.5, 3.333])
        assert video.chapters[1].payload["payload"] == {"who": "signer"}
        assert video.meta.nb_frames == len(HELD), "the frames it holds, not the source's"
        frames, indices = video.read(size=(160, 128))
        assert indices.tolist() == HELD
        # the first chapter is the source's top-left corner
        assert np.abs(frames[:20].astype(int) - full[:20, :128, :160]).mean() < 4
        early, own, late = (np.abs(frames[1:20].astype(int) - full[i:i + 19, :128, :160]).mean() for i in (0, 1, 2))
        assert own < min(early, late), "frame for frame: the source one frame earlier or later matches worse"
        assert video.read(15, 50, size=(16, 16)).indices.tolist() == [*range(15, 20), *range(45, 50)]
        assert video.read(105, 110).indices.tolist() == list(range(105, 110)), "a seek lands in a chapter"
        assert video.read(0.6, 1.6, seconds=True, size=(16, 16)).indices.tolist() == [18, 19, 45, 46, 47]
        with pytest.raises(ValueError, match="different shapes"):
            video.read()
        assert video.keyframes() == [0, 45, 100], "the same indices as reads, not positions in the file"


def test_metadata_does_not_depend_on_what_was_read(gapped):
    with Video(gapped) as video:
        video.read(50, 60)  # the decoder now reports the second chapter's size, 256x208
        assert (video.meta.width, video.meta.height) == (160, 128)


@pytest.mark.parametrize("wrap", [str, lambda path: Pipe(Path(path).read_bytes())])
def test_chapter_frames_reads_each_chapter_from_one_decode(gapped, wrap):
    with Video(wrap(gapped)) as video:
        chapters = [(chapter.id, [(frame.shape, index) for frame, index in frames])
                    for chapter, frames in video.chapter_frames(indices=True)]
    assert chapters == [(1, [((128, 160, 3), index) for index in range(20)]),
                        (2, [((208, 256, 3), index) for index in range(45, 80)]),
                        (3, [((240, 320, 3), index) for index in range(100, 128)])]


def test_fps_starts_over_in_each_chapter_and_keeps_its_last_frame(gapped):
    with Video(gapped) as video:
        counts = [sum(1 for _ in frames) for _, frames in video.chapter_frames(fps=15, size=(32, 32))]
        assert counts == [11, 18, 15]
        kept = video.read(fps=15, size=(32, 32)).indices.tolist()
        assert kept == [*range(0, 20, 2), 19, *range(45, 80, 2), *range(100, 128, 2), 127]
        stepped = video.read(fps=15, size=(32, 32), keep_last=False).indices.tolist()
        assert stepped == [*range(0, 20, 2), *range(45, 80, 2), *range(100, 128, 2)]


@pytest.fixture
def packed(tmp_path) -> Path:
    # as 0.x wrote before 0.10, chapters back to back: source frames 30-59 and 90-119 sit at 0-1 s and 1-2 s
    path = Path(shutil.copy(EXAMPLE, tmp_path / "packed.mkv"))
    _replace_chapters(path, [
        {"id": i + 1, "start": i * 1000, "end": (i + 1) * 1000, "time_base": Fraction(1, 1000),
         "metadata": {"payload": json.dumps({"start_frame": start, "end_frame": start + 30, "who": "signer"})}}
        for i, start in enumerate((30, 90))])
    return path


def test_a_packed_file_is_converted_in_place_when_opened_by_its_path(packed, full):
    with pytest.warns(UserWarning, match="converting the file in place"), Video(packed) as video:
        frames, indices = video.read()
    assert indices.tolist() == [*range(30, 60), *range(90, 120)]
    assert np.array_equal(frames, full[:60]), "the same pictures, each now at its source's time"
    converted = packed.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # it is converted once: the second open has nothing to warn about
        with Video(packed) as video:
            assert [(c.id, float(c.start), float(c.end), c.payload["who"]) for c in video.chapters] == [
                (1, 1, 2, "signer"), (2, 3, 4, "signer")]
            assert video.meta.nb_frames == 60
        unpack_chapters(packed)
    assert packed.read_bytes() == converted, "a file that keeps source time is left alone"


def test_a_packed_video_that_cannot_be_rewritten_is_read_from_a_temporary_copy(packed, full, tmp_path, monkeypatch):
    spare = tmp_path / "spare"
    spare.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spare))
    stored = packed.read_bytes()
    packed.parent.chmod(0o555)  # as on a read-only mount
    try:
        for _ in range(2):  # nothing is kept, so every open converts again
            with pytest.warns(UserWarning, match="cannot be rewritten"):
                video = Video(packed)
            assert any(spare.iterdir())
            frames, indices = video.read()
            video.close()
            assert indices.tolist() == [*range(30, 60), *range(90, 120)]
            assert np.array_equal(frames, full[:60])
            assert not any(spare.iterdir()), "the copy goes when the video is closed"
        assert packed.read_bytes() == stored
        spare.chmod(0o555)
        with pytest.raises(PackedChaptersError, match="could not be converted: "):
            video_metadata(packed)
    finally:
        packed.parent.chmod(0o755)
        spare.chmod(0o755)


def test_a_packed_video_without_a_file_is_refused(packed):
    assert issubclass(PackedChaptersError, ValueError)
    with pytest.raises(PackedChaptersError, match="unpack_chapters"):  # bytes have no file to convert
        Video(packed.read_bytes())


# ordinary markers, and titles that only look like ours: a title is anyone's text
TITLES = ("Intro", "1", "[2]", '{"start_frame": Infinity}', '{"start_frame": 0, "rate": "-30"}', "[" * 100_000)


def test_a_video_without_our_chapters_has_none(tmp_path):
    path = Path(shutil.copy(EXAMPLE, tmp_path / "marked.mkv"))
    _replace_chapters(path, [{"id": i, "start": i * 2000, "end": i * 2000 + 2000, "time_base": Fraction(1, 1000),
                              "metadata": {"title": title}} for i, title in enumerate(TITLES)])
    with Video(path) as video:
        assert len(video.container.chapters()) == len(TITLES)
        assert video.chapters == []
        assert [(chapter, sum(1 for _ in frames)) for chapter, frames in video.chapter_frames()] == [(None, 130)]


def test_a_stream_is_opened_for_reading_whatever_mode_it_reports():
    # an upload spooled in memory (Starlette's UploadFile.file) says "w+b", which PyAV alone takes as "open to write"
    with tempfile.SpooledTemporaryFile() as upload:
        upload.write(Path(EXAMPLE).read_bytes())
        upload.seek(0)
        with Video(upload) as video:
            assert video.meta.nb_frames == 130


def test_a_dropped_video_closes_its_container(monkeypatch):
    container = Video(EXAMPLE).container  # nothing closes it, as in the one-liner `Video(path).read()`
    assert not container.input_was_opened, "PyAV alone keeps the file and decoder until the cyclic collector runs"
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    with pytest.raises(RuntimeError, match="Failed to open video"):
        Video(b"not a video")
    assert not unraisable, "a Video that never opened has nothing to close"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes on this platform")
def test_a_named_pipe_cannot_seek_even_when_it_is_given_by_path(tmp_path):
    pipe = tmp_path / "video.mkv"
    os.mkfifo(pipe)
    threading.Thread(target=pipe.write_bytes, args=((ASSETS / "no_nb_frames.mkv").read_bytes(),), daemon=True).start()
    with Video(pipe) as video:
        assert video.meta.nb_frames == 38, "the header's estimate: counting would use the pipe up"
        assert sum(1 for _ in video.frames()) == 37
        with pytest.raises(ValueError, match="read only once"):
            next(video.frames())


def test_reads_by_time_refuse_a_stream_without_timestamps(tmp_path):
    path = make_video(tmp_path / "raw.h264", frames=30)  # an elementary stream: frames, but no times
    with Video(path) as video:
        assert numbers(video.read(10, 13).frames) == [10, 11, 12]
    with Video(path) as video, pytest.raises(ValueError, match="frame times"):
        video.read(0.2, 0.4, seconds=True)
    with Video(path) as video, pytest.raises(ValueError, match="frame times"):
        next(video.frames(fps=10))


def test_what_pyav_cannot_decode_is_an_invalid_video(tmp_path):
    unknown = write_video(None, [np.zeros((48, 64, 3), np.uint8)] * 3, 30, codec="mpeg4", format="avi")
    with pytest.raises(InvalidVideoError, match="no decoder"):
        Video(unknown.replace(b"FMP4", b"ZZZZ"))  # a codec no decoder knows
    whole = (ASSETS / "no_nb_frames.webm").read_bytes()
    with Video(whole[:len(whole) // 50]) as video, pytest.raises(VideoError):  # the header, and nothing to seek to
        _ = video.meta
    with Video(whole[:len(whole) // 50]) as video:  # no chapters and no frames is still one group, an empty one
        assert [(chapter, list(frames)) for chapter, frames in video.chapter_frames()] == [(None, [])]
    with Video(whole[:len(whole) // 2]) as video:  # the header still promises 107 frames
        assert video.meta.nb_frames == len(video.read().frames) == 33
