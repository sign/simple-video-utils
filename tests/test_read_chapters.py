import io
import json
import os
from fractions import Fraction
from queue import Queue
from threading import Thread

import av
import numpy as np
import pytest

from simple_video_utils.chapters import list_chapters, read_chapters
from simple_video_utils.frames import FrameSize, read_frames_exact, stack_frames


def video(format="matroska", bounds=((0, 1000), (1000, 2000)), metadata=None):
    output = io.BytesIO()
    with av.open(output, "w", format=format) as container:
        container.set_chapters(
            [
                {
                    "id": i,
                    "start": start,
                    "end": end,
                    "time_base": Fraction(1, 1000),
                    "metadata": (metadata or [{"title": "First"}, {"title": "Second"}])[i],
                }
                for i, (start, end) in enumerate(bounds)
            ]
        )
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 60
        for i in range(90):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 2, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return output.getvalue()


def mixed_video():
    output = io.BytesIO()
    with av.open(output, "w", format="matroska") as container:
        container.set_chapters(
            [
                {
                    "id": i,
                    "start": i * 1000,
                    "end": (i + 1) * 1000,
                    "time_base": Fraction(1, 1000),
                    "metadata": {"payload": json.dumps({"start_frame": start})},
                }
                for i, start in enumerate((30, 90))
            ]
        )
        track = container.add_mux_stream("h264", rate=30, width=64, height=32)
        for i, (width, height) in enumerate(((64, 32), (32, 64))):
            encoder = av.CodecContext.create("libx264", "w")
            encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
            encoder.time_base, encoder.framerate = Fraction(1, 30), Fraction(30)
            encoder.max_b_frames = 0
            for j in range(30):
                frame = av.VideoFrame.from_ndarray(
                    np.full((height, width, 3), 32 + i * 160, dtype=np.uint8), format="rgb24"
                )
                frame.pts, frame.time_base = i * 30 + j, Fraction(1, 30)
                for packet in encoder.encode(frame):
                    packet.stream = track
                    container.mux(packet)
            for packet in encoder.encode():
                packet.stream = track
                container.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize("format", ["matroska", "mp4"])
def test_native_chapters_need_no_source_payload(format):
    result = [
        (chapter, start, stack_frames(frames))
        for chapter, start, frames in read_chapters(io.BytesIO(video(format)), fps=15, size=FrameSize((256, 256)))
    ]
    assert [start for _, start, _ in result] == [0, 1]
    assert [frames.shape for _, _, frames in result] == [(15, 256, 256, 3)] * 2
    assert result[0][2][-1].mean() < result[1][2][0].mean()


def test_non_keyframe_bounds_are_half_open():
    raw = video(bounds=((350, 900), (900, 2400)))
    with av.open(io.BytesIO(raw)) as container:
        expected = [
            (frame.pts * frame.time_base, frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)
        ]
    for chapter, _, frames in read_chapters(io.BytesIO(raw)):
        low, high = (chapter[key] * chapter["time_base"] for key in ("start", "end"))
        arrays = [array for time, array in expected if low <= time < high]
        np.testing.assert_array_equal(stack_frames(frames), np.stack(arrays))


def test_optional_source_payload_and_unrelated_metadata():
    tags = [{"payload": json.dumps({"start_frame": 60, "rate": "30"})}, {"payload": "ordinary non-JSON metadata"}]
    result = [
        (start, len(list(frames))) for _, start, frames in read_chapters(io.BytesIO(video(metadata=tags)), fps=15)
    ]
    assert result == [(2, 15), (1, 15)]


def test_variable_dimensions_resize_and_preserve_source_starts():
    result = [
        (start, stack_frames(frames))
        for _, start, frames in read_chapters(io.BytesIO(mixed_video()), fps=15, size=FrameSize((256, 256)))
    ]
    assert [start for start, _ in result] == [1, 3]
    assert [frames.shape for _, frames in result] == [(15, 256, 256, 3)] * 2
    assert result[0][1][0].mean() == pytest.approx(32, abs=3)
    assert result[1][1][0].mean() == pytest.approx(192, abs=3)


@pytest.mark.parametrize("bounds", [((0, 0),), ((0, 1000), (500, 1500))])
def test_invalid_chapter_ranges_are_rejected(bounds):
    with pytest.raises(ValueError, match="chapters must"):
        [list(gen) for _, _, gen in read_chapters(io.BytesIO(video(bounds=bounds)))]


def test_plain_video_keeps_all_frames_and_matches_the_existing_reader(tmp_path):
    path = tmp_path / "plain.mkv"
    path.write_bytes(video(bounds=()))
    expected = stack_frames(read_frames_exact(str(path)))
    result = [(chapter, start, stack_frames(frames)) for chapter, start, frames in read_chapters(path)]
    assert len(result) == 1
    chapter, start, frames = result[0]
    assert chapter is None
    assert start == 0
    np.testing.assert_array_equal(frames, expected)


def test_list_chapters_reuses_a_container_without_rewinding_or_closing():
    with av.open(io.BytesIO(video())) as container:
        decoded = container.decode(video=0)
        first = next(decoded)
        assert list(list_chapters(container)) == container.chapters()
        assert next(decoded).pts > first.pts


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"size": (256, 256)}])
def test_read_options_are_validated_before_opening(kwargs):
    with pytest.raises((ValueError, TypeError)):
        next(read_chapters("missing.mkv", **kwargs))


def test_incoming_pipe_yields_frames_before_eof():
    raw = video(bounds=((0, 1000), (1000, 3000)))
    ids = [chapter["id"] for chapter in list_chapters(io.BytesIO(raw))]
    reader, writer = os.pipe()
    events = Queue()

    def consume():
        try:
            with os.fdopen(reader, "rb", 0) as source:
                for chapter, _, frames in read_chapters(source, fps=15):
                    count = 0
                    for _ in frames:
                        count += 1
                        if count == 1:
                            events.put(("frame", chapter["id"]))
                    events.put(("count", count))
        except Exception as error:  # noqa: BLE001 - propagate a reader-thread failure to the test
            events.put(error)

    thread = Thread(target=consume, daemon=True)
    thread.start()
    try:
        with os.fdopen(writer, "wb", 0) as target:
            split = len(raw) - max(512, len(raw) // 5)
            target.write(raw[:split])
            assert events.get(timeout=5) == ("frame", ids[0])
            target.write(raw[split:])
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert [events.get(timeout=1) for _ in range(3)] == [("count", 15), ("frame", ids[1]), ("count", 30)]
