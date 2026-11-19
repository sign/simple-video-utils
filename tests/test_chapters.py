import av
import numpy as np
import pytest
from av.video.reformatter import ColorRange
from conftest import ASSETS, EXAMPLE

from simple_video_utils import ChapterClip, Video, write_chapters


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
    source = tmp_path / "full.mp4"
    with av.open(str(source), mode="w") as output:
        stream = output.add_stream("h264", rate=30, options={"crf": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuvj420p"
        image = np.linspace(0, 255, 64 * 48 * 3).reshape(48, 64, 3).astype(np.uint8)
        output.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuvj420p")))
        output.mux(stream.encode())
    out, _ = write(tmp_path, source, [ChapterClip(0, 1, (-8, -8, 56, 40))], crf=0)
    with av.open(str(out)) as container:
        assert next(container.decode(video=0)).color_range == ColorRange.JPEG
    [(y, _, _)], [(source_y, _, _)] = planes(out, range(1)), planes(source, range(1))
    assert near(y[:8], 0), "black is 0 in full range, 16 in limited"
    assert near(y[:, :8], 0)
    assert near(y[8:, 8:], source_y[:40, :56])


def test_a_rotated_source_is_cut_as_it_is_displayed(tmp_path):
    rotated = ASSETS / "rotated90.mp4"
    out, _ = write(tmp_path, rotated, [ChapterClip(2, 6, (40, 160, 200, 480))], crf=0)
    with Video(rotated) as source, Video(out) as video:
        assert video.meta.rotation == 0
        frames = video.read()
        assert frames.shape == (4, 320, 160, 3)
        assert np.abs(frames.astype(int) - source.read(2, 6)[:, 160:480, 40:200]).mean() < 1


def test_resolution_resizes_a_chapter(tmp_path):
    out, _ = write(tmp_path, EXAMPLE, [ChapterClip(0, 3, (0, 0, 320, 240), resolution=(160, 112))])
    with Video(out) as video:
        assert video.read().shape == (3, 112, 160, 3)
        assert video.chapters[0].payload["resolution"] == [160, 112]


def test_clips_are_cut_at_the_last_frame_the_source_decodes(tmp_path):
    clips = [ChapterClip(100, 140, (0, 0, 160, 128)), ChapterClip(150, 160, (0, 0, 16, 16))]
    out, written = write(tmp_path, EXAMPLE, clips)
    assert written == [ChapterClip(100, 130, (0, 0, 160, 128))]
    with Video(out) as video:
        assert [(c.payload["start_frame"], c.payload["end_frame"]) for c in video.chapters] == [(100, 130)]
        assert video.read(indices=True)[1].tolist() == list(range(100, 130))


def test_a_source_without_a_header_rate(tmp_path):
    out, _ = write(tmp_path, ASSETS / "no_average_rate_5s.webm", [ChapterClip(4, 30, (0, 0, 96, 96))])
    with Video(out) as video:
        assert video.meta.fps == pytest.approx(19.8)
        assert len(video.read()) == 26


def test_decoding_threads_do_not_change_the_output(tmp_path):
    clips = [ChapterClip(0, 12, (0, 0, 160, 128))]
    write_chapters(EXAMPLE, tmp_path / "auto.mkv", clips, preset="ultrafast")
    write_chapters(EXAMPLE, tmp_path / "none.mkv", clips, preset="ultrafast", thread_type="NONE")
    packets = []
    for name in ("auto.mkv", "none.mkv"):
        with av.open(str(tmp_path / name)) as container:
            packets.append([bytes(packet) for packet in container.demux(video=0)])
    assert packets[0] == packets[1]


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
