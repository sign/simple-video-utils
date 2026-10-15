from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.chapters import MAX_SIDE, ChapterClip, list_chapters, square, write_chapters
from simple_video_utils.frames import read_frames_batched, read_frames_exact
from simple_video_utils.metadata import count_frames, open_video

ASSETS = Path(__file__).parent / "assets"
FPS = 10
# source frames [start, end) and the square box of each chapter, on whole 16-px macroblocks so no bars square it: the
# chapters start 3 s into the source, with gaps between them
CLIPS = [
    ChapterClip(30, 40, [0.0, 0.0, 64.0, 64.0]),
    ChapterClip(45, 55, [16.0, 16.0, 48.0, 48.0]),
    ChapterClip(60, 70, [16.0, 16.0, 64.0, 64.0], payload={"note": "last"}),
]
SIDES = [64, 32, 48]
INDICES = [*range(30, 40), *range(45, 55), *range(60, 70)]


def level(index: int) -> int:
    """Each source frame is one grey level, which a crop keeps."""
    return 10 * (index % 10) + 100


def write_source(path: Path, frames: int, fps: int = FPS, width: int = 160, height: int = 120) -> str:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), level(index), np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return str(path)


@pytest.fixture(scope="module")
def chapter_video(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("chapters")
    out = directory / "chapters.mkv"
    write_chapters(write_source(directory / "source.mp4", frames=80), out, CLIPS)
    return str(out)


def test_list_chapters_gives_each_chapter_its_time_and_payload(chapter_video):
    chapters = list_chapters(chapter_video)
    assert [(chapter.start, chapter.end) for chapter in chapters] == [(3, 4), (Fraction(9, 2), Fraction(11, 2)), (6, 7)]
    assert chapters[1].payload == {
        "start_frame": 45,
        "end_frame": 55,
        "box": [16.0, 16.0, 48.0, 48.0],
        "crop": [16, 16, 48, 48],
        "pad": [0, 0, 0, 0],
        "side": 32,
        "scale": 1.0,
        "resolution": {"width": 32, "height": 32},
    }
    assert chapters[2].payload["note"] == "last"


def test_chapter_frames_keep_their_own_size_at_their_source_index(chapter_video):
    frames = list(read_frames_exact(chapter_video, return_indices=True))
    assert [index for _, index in frames] == INDICES
    assert [frame.shape for frame, _ in frames] == [(side, side, 3) for side in SIDES for _ in range(10)]
    assert [float(frame.mean()) for frame, _ in frames] == pytest.approx([level(index) for index in INDICES], abs=2)


def test_each_frame_carries_its_chapter_payload(chapter_video):
    frames = read_frames_exact(chapter_video, return_indices=True, return_chapters=True)
    starts = [clip.start_frame for clip in CLIPS for _ in range(10)]
    assert [payload["start_frame"] for _, _, payload in frames] == starts


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


def test_a_batch_carries_each_frame_s_payload(chapter_video):
    frames, indices, payloads = read_frames_batched(
        chapter_video, start_frame=45, end_frame=54, return_indices=True, return_chapters=True
    )
    assert frames.shape == (10, 32, 32, 3)
    assert indices.tolist() == list(range(45, 55))
    assert [payload["side"] for payload in payloads] == [32] * 10


def test_chapters_without_payloads_read_by_position_as_any_video(chapter_video, tmp_path):
    # a film's chapters say nothing of where its frames come from: its frames count from 0, though it starts 3 s in
    path = tmp_path / "film.mkv"
    with av.open(chapter_video) as source, av.open(str(path), "w", format="matroska") as film:
        film.set_chapters([{**chapter, "metadata": {"title": "scene"}} for chapter in source.chapters()])
        track = film.add_stream_from_template(source.streams.video[0])
        for packet in source.demux(video=0):
            if packet.size:
                packet.stream = track
                film.mux(packet)
    assert [chapter.payload for chapter in list_chapters(path)] == [None] * 3
    frames = list(read_frames_exact(str(path), return_indices=True, return_chapters=True))
    assert [index for _, index, _ in frames] == list(range(len(INDICES)))
    assert [payload for _, _, payload in frames] == [{}] * len(INDICES)
    assert [index for _, index in read_frames_exact(str(path), start_frame=12, end_frame=14, return_indices=True)] == [
        12, 13, 14,
    ]


def test_a_video_without_chapters_has_no_payloads():
    payloads = [payload for _, payload in read_frames_exact(str(ASSETS / "example.mp4"), end_frame=4,
                                                            return_chapters=True)]
    assert payloads == [None] * 5
    assert list_chapters(ASSETS / "example.mp4") == []


def test_square_cuts_the_crop_out_to_whole_macroblocks_and_bars_it_by_whole_macroblocks():
    # each edge out to its next macroblock: 80 x 48, squared by 16 rows above and below
    assert square(box=[10.5, 20.2, 70.9, 50.1], frame_size=(160, 120)) == ([0, 16, 80, 64], [0, 16, 0, 16], 80, 80)
    # a box running off an odd-sized frame stops at its last even column, here a macroblock's; the 48 columns to
    # spare split into whole macroblocks, 32 before the crop and 16 after it
    assert square(box=[100.0, 0.0, 200.0, 99.0], frame_size=(161, 120)) == ([96, 0, 160, 112], [32, 0, 16, 0], 112, 112)
    # stopped by the frame's edge 8 rows into a macroblock, the crop is barred up to the whole one
    assert square(box=[150.0, 110.0, 159.0, 119.0], frame_size=(161, 121)) == (
        [144, 96, 160, 120],
        [16, 0, 0, 8],
        32,
        32,
    )
    # a box inside one macroblock keeps it
    assert square(box=[3.0, 3.0, 9.0, 9.0], frame_size=(160, 120)).crop == [0, 0, 16, 16]
    # a square larger than max_side is scaled down to it, not rounded
    assert square(box=[0.0, 0.0, 1920.0, 1080.0], frame_size=(1920, 1080))[2:] == (1920, MAX_SIDE)
    # a box running off the frame's left and top edges is cut at them
    assert square(box=[-5.0, -20.0, 30.0, 40.0], frame_size=(160, 120)).crop == [0, 0, 32, 48]
    with pytest.raises(ValueError, match="empty crop"):
        square(box=[170.0, 0.0, 180.0, 10.0], frame_size=(160, 120))
    with pytest.raises(ValueError, match="empty crop"):
        square(box=[-40.0, -40.0, -20.0, -20.0], frame_size=(160, 120))


def test_write_chapters_writes_each_clip_as_a_square_chapter_at_its_source_time(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=8, fps=3)
    clips = [ChapterClip(0, 3, [10.0, 40.0, 31.0, 71.0]), ChapterClip(5, 8, [24.0, 40.0, 45.0, 71.0])]
    out = tmp_path / "source.mkv"
    payloads = write_chapters(source, out, clips)

    assert [(p["crop"], p["pad"], p["side"], p["resolution"]) for p in payloads] == [
        ([0, 32, 32, 80], [16, 0, 0, 0], 48, {"width": 48, "height": 48}),
        ([16, 32, 48, 80], [16, 0, 0, 0], 48, {"width": 48, "height": 48}),
    ]
    assert [chapter.payload for chapter in list_chapters(out)] == payloads
    with av.open(str(out)) as container:
        assert len(container.streams) == 1
        assert [container.metadata[key] for key in ("SOURCE_FPS", "SOURCE_FRAMES", "SOURCE_WIDTH")] == ["3", "8", "160"]
        # the file lasts as long as the source, to the millisecond Matroska keeps, though its last chapter ends early
        assert container.duration == pytest.approx(8 / 3 * 1e6, abs=1e3)
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == ["0-3", "5-8"]
        frames = [(round(frame.time * 3), frame.width, frame.height) for frame in container.decode(video=0)]
        # a seek to a chapter's start lands on its own first frame, not the previous chapter's keyframe
        stream = container.streams.video[0]
        container.seek(int(list_chapters(out)[1].start / stream.time_base), stream=stream)
        assert round(next(container.decode(stream)).time * 3) == 5
    assert frames == [(index, 48, 48) for index in (0, 1, 2, 5, 6, 7)]


def test_chapters_starting_on_the_frame_after_the_last_play_in_order_at_their_own_sizes(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=9, fps=3)
    # each encoder's first packet decodes before its first frame shows, so it must still follow the last chapter's
    boxes = [[10.0, 40.0, 31.0, 71.0], [0.0, 10.0, 61.0, 111.0], [20.0, 20.0, 61.0, 61.0]]
    out = tmp_path / "source.mkv"
    write_chapters(source, out, [ChapterClip(3 * k, 3 * k + 3, box) for k, box in enumerate(boxes)])
    with av.open(str(out)) as container:
        frames = [(round(frame.time * 3), frame.width) for frame in container.decode(video=0)]
    assert frames == [(0, 48), (1, 48), (2, 48), (3, 112), (4, 112), (5, 112), (6, 48), (7, 48), (8, 48)]


def test_every_chapter_decodes_at_its_payload_size_after_another_of_the_same_macroblocks(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=12, fps=3)
    # cut on even pixels, these boxes would square to 34 and 36, which share three macroblocks: FFmpeg would show the
    # second at the first's size. Cut out to whole macroblocks, both square to 48
    clips = [ChapterClip(0, 6, [10.0, 40.0, 31.0, 74.0]), ChapterClip(6, 12, [20.0, 36.0, 41.0, 72.0])]
    out = tmp_path / "source.mkv"
    payloads = write_chapters(source, out, clips)
    with av.open(str(out)) as container:
        sizes = [(round(frame.time * 3), frame.width) for frame in container.decode(video=0)]
    assert [p["side"] for p in payloads] == [48, 48]
    assert sizes == [(index, 48) for index in range(12)]


def test_a_square_larger_than_max_side_is_scaled_down_to_it(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=3, fps=3)
    out = tmp_path / "source.mkv"
    [payload] = write_chapters(source, out, [ChapterClip(0, 3, [16.0, 48.0, 47.0, 79.0])], max_side=16)
    with av.open(str(out)) as container:
        assert {(frame.width, frame.height) for frame in container.decode(video=0)} == {(16, 16)}
    assert (payload["side"], payload["scale"], payload["resolution"]) == (32, 0.5, {"width": 16, "height": 16})


def test_on_frame_sees_each_clip_frame_whole_before_the_cut(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=8, fps=3)
    seen = []
    clips = [ChapterClip(1, 3, [0.0, 0.0, 32.0, 32.0]), ChapterClip(5, 7, [0.0, 0.0, 32.0, 32.0])]
    write_chapters(source, tmp_path / "source.mkv", clips,
                   on_frame=lambda k, frame: seen.append((k, round(frame.time * 3), frame.width)))
    assert seen == [(0, 1, 160), (0, 2, 160), (1, 5, 160), (1, 6, 160)]


def test_write_chapters_rejects_clips_it_cannot_write(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=6, fps=3)
    box = [0.0, 0.0, 32.0, 32.0]
    with pytest.raises(ValueError, match="no clips"):
        write_chapters(source, tmp_path / "none.mkv", [])
    with pytest.raises(ValueError, match="overlap"):
        write_chapters(source, tmp_path / "overlap.mkv", [ChapterClip(0, 4, box), ChapterClip(3, 6, box)])
    with pytest.raises(ValueError, match="inside the clip"):
        write_chapters(source, tmp_path / "long.mkv", [ChapterClip(3, 9, box)])
    with pytest.raises(ValueError, match="rotated"):
        write_chapters(str(ASSETS / "rotated90.mp4"), tmp_path / "rotated.mkv", [ChapterClip(0, 3, box)])
