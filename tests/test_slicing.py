import io
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import EXAMPLE, Pipe, make_video, numbers

from simple_video_utils import Slice, Video


@pytest.fixture
def source(tmp_path):
    return make_video(tmp_path / "source.mp4", frames=120)


def read(clip: bytes, **arguments):
    with Video(clip) as video:
        return video.read(**arguments).frames


def test_slice_copies_the_frames_between_start_and_stop(source):
    with Video(source) as video:
        whole = video.read().frames
    clip = b"".join(Slice(source, 1.0, 2.5))
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


def test_consecutive_slices_share_no_frame(source):
    assert [numbers(read(b"".join(Slice(source, a, b)))) for a, b in ((0, 1), (1, 2), (2, 9))] == [
        list(range(30)), list(range(30, 60)), list(range(60, 120))]


def test_slice_streams_as_it_is_read(tmp_path):
    source = make_video(tmp_path / "noise.mp4", frames=120, noise=True)
    with Video(source) as video:
        whole = video.read().frames
    clip = Slice(source, 0.4, 3.9)
    assert clip.duration == pytest.approx(3.5)
    assert clip.size > 4 * 64 * 1024, "measured without keeping any media"
    chunks = list(clip)
    data = b"".join(chunks)
    assert len(chunks) > 4
    assert max(map(len, chunks)) < 2 * 64 * 1024
    assert len(data) == clip.size
    assert data == b"".join(clip), "replays byte for byte"
    assert data.index(b"moov") < data.index(b"mdat")
    with Video(Pipe(data)) as video:  # moov first: a reader needs no seeking
        assert np.array_equal(video.read().frames, whole[12:117])


def test_slice_of_bytes_and_of_b_frames(tmp_path):
    data = Path(make_video(tmp_path / "b.mkv", frames=90, bframes=2)).read_bytes()
    with Video(data) as video:
        whole = video.read().frames
    clip = read(b"".join(Slice(data, 1.2, 2.2)))
    # Matroska drops decode timestamps after a seek; with B-frames, the frame at `stop` can remain
    assert np.array_equal(clip[:30], whole[36:66])
    assert len(clip) <= 31


def test_b_frames_leave_later_frames_visible_in_a_plain_slice(tmp_path):
    source = make_video(tmp_path / "b.mp4", frames=120, gop=30, bframes=3)
    shown = numbers(read(b"".join(Slice(source, 0, 4 / 30))))
    # frames 0-3 are predicted from frames 4 and 6, which are copied with them and cannot be hidden
    assert shown[:4] == [0, 1, 2, 3]
    assert shown[4:] == [4, 6]


def test_slice_times_are_matched_on_the_files_clock(gapped):
    # frame 100 of 30 fps is at 3.3333 s, stored as 3333 ms: a slice from 100 / 30 must still start on it
    with Video(b"".join(Slice(gapped, 100 / 30, 110 / 30))) as video:
        assert video.read().indices.tolist() == list(range(100, 110))


def test_slice_keeps_gaps_and_carries_its_chapters(gapped):
    with Video(gapped) as video:
        frames, indices = video.read(size=(64, 64))
        held = dict(zip(indices.tolist(), frames, strict=True))
    clip = Slice(gapped, 0.5, 3.5)
    data = b"".join(clip)
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
    again = b"".join(Slice(data, 0.1, 1.2))
    with Video(again) as video:  # a slice of a slice is still placed on the original source
        assert video.read(size=(64, 64)).indices.tolist() == [18, 19, *range(45, 51)]


def test_slice_starting_in_a_gap_starts_on_its_first_frame(gapped):
    data = b"".join(Slice(gapped, 3.0, 4.0))  # the third chapter starts at 3.333
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
        next(iter(Slice(source, start, stop)))


def test_slice_in_a_gap_has_no_frames(gapped):
    with pytest.raises(ValueError, match="no frames"):
        _ = Slice(gapped, 0.7, 1.4).size


def test_slice_needs_a_source_it_can_reopen():
    with open(EXAMPLE, "rb") as file, pytest.raises(TypeError, match="reopens"):
        Slice(file, 0, 1)


def test_slice_notices_a_source_that_changed(tmp_path, source):
    clip = Slice(source, 1.0, 3.0)
    assert clip.size
    make_video(source, frames=40)
    with pytest.raises(OSError, match="changed"):
        b"".join(clip)
