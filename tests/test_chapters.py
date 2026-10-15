from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.chapters import ChapterClip, list_chapters, write_chapters
from simple_video_utils.frames import read_frames_batched, read_frames_exact

ASSETS = Path(__file__).parent / "assets"
FPS = 10
# source frames [start, end) and the square each chapter is cropped to, on whole 16-px macroblocks, with gaps between
# the clips in the source
CLIPS = [
    ChapterClip(30, 40, "crop=64:64:0:0"),
    ChapterClip(45, 55, "crop=32:32:16:16"),
    ChapterClip(60, 70, "crop=48:48:16:16", payload={"note": "last"}),
]
SIDES = [64, 32, 48]
SOURCE_INDICES = [*range(30, 40), *range(45, 55), *range(60, 70)]


def level(index: int) -> int:
    """Each source frame is one grey level, which a crop keeps."""
    return 10 * (index % 10) + 100


def write_source(
    path: Path,
    frames: int,
    fps: int = FPS,
    width: int = 160,
    height: int = 120,
    pix_fmt: str = "yuv420p",
    gop: int = 12,
) -> str:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt, stream.codec_context.gop_size = width, height, pix_fmt, gop
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
    chapters = list(list_chapters(chapter_video))
    # back to back from the file's start, ten frames each at the source's 10 fps
    assert [(chapter.start, chapter.end) for chapter in chapters] == [(0, 1), (1, 2), (2, 3)]
    assert [chapter.payload for chapter in chapters] == [
        {"start_frame": 30, "end_frame": 40},
        {"start_frame": 45, "end_frame": 55},
        {"start_frame": 60, "end_frame": 70, "note": "last"},
    ]
    assert list(list_chapters(ASSETS / "example.mp4")) == []


def test_chapter_frames_read_back_to_back_at_their_own_sizes(chapter_video):
    frames = list(read_frames_exact(chapter_video))
    assert [frame.shape for frame in frames] == [(side, side, 3) for side in SIDES for _ in range(10)]
    assert [float(frame.mean()) for frame in frames] == pytest.approx([level(i) for i in SOURCE_INDICES], abs=2)
    # a range is file positions: the last three of the first chapter, then the second's first two
    sides = [frame.shape[0] for frame in read_frames_exact(chapter_video, start_frame=7, end_frame=11)]
    assert sides == [64, 64, 64, 32, 32]


def test_size_scales_every_chapter_s_frames_to_one_size(chapter_video):
    frames = list(read_frames_exact(chapter_video, size=(48, 40)))
    assert {frame.shape for frame in frames} == {(40, 48, 3)}
    assert [float(frame.mean()) for frame in frames] == pytest.approx([level(i) for i in SOURCE_INDICES], abs=2)


def test_batching_frames_of_different_sizes_needs_a_size(chapter_video):
    with pytest.raises(ValueError, match="size="):
        read_frames_batched(chapter_video)
    assert read_frames_batched(chapter_video, size=(32, 32)).shape == (30, 32, 32, 3)


def test_write_chapters_writes_each_clip_through_its_filters(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=8, fps=3)
    clips = [ChapterClip(0, 3, "crop=32:48:0:32,pad=48:48:16:0"), ChapterClip(5, 8, "crop=32:48:16:32,scale=16:24")]
    out = tmp_path / "source.mkv"
    write_chapters(source, out, clips)

    assert [chapter.payload for chapter in list_chapters(out)] == [
        {"start_frame": 0, "end_frame": 3},
        {"start_frame": 5, "end_frame": 8},
    ]
    with av.open(str(out)) as container:
        assert len(container.streams) == 1
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == ["0-3", "5-8"]
        frames = [(round(frame.time * 3), frame.width, frame.height) for frame in container.decode(video=0)]
        # a seek to a chapter's start lands on its own first frame, not the previous chapter's keyframe
        stream = container.streams.video[0]
        container.seek(int(list(list_chapters(out))[1].start / stream.time_base), stream=stream)
        assert round(next(container.decode(stream)).time * 3) == 3
    assert frames == [(0, 48, 48), (1, 48, 48), (2, 48, 48), (3, 16, 24), (4, 16, 24), (5, 16, 24)]


def test_a_chapter_without_filters_keeps_the_source_s_size_and_pixel_format(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=3, fps=3, pix_fmt="yuvj420p")
    out = tmp_path / "source.mkv"
    write_chapters(source, out, [ChapterClip(0, 3)])
    with av.open(str(out)) as container:
        assert {(frame.width, frame.height, frame.format.name) for frame in container.decode(video=0)} == {
            (160, 120, "yuvj420p")
        }


def test_x264_picks_the_chapter_s_frame_types_not_the_source_s(tmp_path):
    # every source frame is a keyframe; passed on, its type would make every chapter frame one too
    source = write_source(tmp_path / "source.mp4", frames=10, fps=10, gop=1)
    out = tmp_path / "source.mkv"
    write_chapters(source, out, [ChapterClip(0, 10)])
    with av.open(str(out)) as container:
        assert [packet.is_keyframe for packet in container.demux(video=0) if packet.size] == [True] + [False] * 9


def test_on_frame_sees_each_clip_frame_whole_before_its_filters(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=8, fps=3)
    seen = []
    clips = [ChapterClip(1, 3, "crop=32:32:0:0"), ChapterClip(5, 7, "crop=32:32:0:0")]
    write_chapters(
        source,
        tmp_path / "source.mkv",
        clips,
        on_frame=lambda k, frame: seen.append((k, round(frame.time * 3), frame.width)),
    )
    assert seen == [(0, 1, 160), (0, 2, 160), (1, 5, 160), (1, 6, 160)]


def test_write_chapters_rejects_clips_it_cannot_write(tmp_path):
    source = write_source(tmp_path / "source.mp4", frames=6, fps=3)
    with pytest.raises(ValueError, match="no clips"):
        write_chapters(source, tmp_path / "none.mkv", [])
    with pytest.raises(ValueError, match="overlap"):
        write_chapters(source, tmp_path / "overlap.mkv", [ChapterClip(0, 4), ChapterClip(3, 6)])
    with pytest.raises(ValueError, match="inside the clip"):
        write_chapters(source, tmp_path / "long.mkv", [ChapterClip(3, 9)])
    # 34 and 36 share three macroblocks: FFmpeg would show the second chapter at the first's size
    with pytest.raises(ValueError, match="macroblocks"):
        write_chapters(
            source, tmp_path / "blocks.mkv", [ChapterClip(0, 3, "crop=34:34:0:0"), ChapterClip(3, 6, "crop=36:36:0:0")]
        )
