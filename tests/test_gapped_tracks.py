import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from simple_video_utils.joining import VideoCrop, VideoTrack, merge_video_tracks, write_video_crops, write_video_tracks


def _tracks():
    return [
        VideoTrack(
            start_frame=start,
            end_frame=end,
            width=width,
            height=height,
            frames=[np.full((height, width, 3), value, dtype=np.uint8)] * (end - start),
            payload={"signer_id": start},
        )
        for start, end, width, height, value in [(2, 4, 64, 48, 200), (6, 8, 31, 23, 150)]
    ]


@pytest.mark.parametrize("gap_value", [0, 90])
def test_merge_video_tracks_preserves_timeline_and_payloads(tmp_path, gap_value):
    source, output = tmp_path / "source.mkv", tmp_path / "merged.mp4"
    write_video_tracks(_tracks(), source, fps=5, source_frames=10)

    with av.open(str(source)) as container:
        assert [(stream.width, stream.height) for stream in container.streams.video] == [(64, 48), (32, 24)]
        starts = {}
        for packet in container.demux(*container.streams.video):
            for frame in packet.decode():
                starts.setdefault(packet.stream.index, frame.time)
        assert list(starts.values()) == pytest.approx([0.4, 1.2])

    kwargs = {"gap_frame": np.full((48, 80, 3), gap_value, dtype=np.uint8)} if gap_value else {}
    merge_video_tracks(source=source, output=output, **kwargs)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert container.streams.video[0].width == (80 if gap_value else 64)
        frames = list(container.decode(video=0))
        assert len(frames) == 10
        assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx([i / 5 for i in range(10)])
        colors = [frame.to_ndarray(format="rgb24") for frame in frames]
        assert np.mean(colors[0]) == pytest.approx(gap_value, abs=3)
        assert np.mean(colors[5]) == pytest.approx(gap_value, abs=3)
        assert np.mean(colors[9]) == pytest.approx(gap_value, abs=3)
        first_left = (container.streams.video[0].width - 64) // 2
        assert np.mean(colors[2][:, first_left : first_left + 64]) == pytest.approx(200, abs=3)
        left = (container.streams.video[0].width - 32) // 2
        assert np.mean(colors[6][12:35, left : left + 31]) == pytest.approx(150, abs=5)
        assert np.mean(colors[6][0, 0]) < 5
        assert [entry["payload"]["signer_id"] for entry in json.loads(container.metadata["comment"])] == [2, 6]


def test_merge_video_tracks_rejects_a_file_without_the_format_tag(tmp_path):
    source = tmp_path / "plain.mkv"
    with av.open(str(source), mode="w", format="matroska") as container:
        container.metadata.update(source_fps="5", source_frames="2")
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        stream.metadata["title"] = "0-2"
        for _ in range(2):
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")))
        container.mux(stream.encode())

    with pytest.raises(ValueError, match="gapped_tracks"):
        merge_video_tracks(source=source, output=tmp_path / "merged.mp4")


def test_merge_video_tracks_accepts_tracks_out_of_timeline_order(tmp_path):
    source, output = tmp_path / "source.mkv", tmp_path / "merged.mp4"
    rate = Fraction(30000, 1001)
    tracks = [
        VideoTrack(start, start + 20, 16, 16, [np.full((16, 16, 3), value, np.uint8)] * 20, {"start": start})
        for start, value in [(100, 200), (2, 100)]
    ]
    write_video_tracks(tracks, source, fps=rate, source_frames=130)
    merge_video_tracks(source, output)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 130
        assert [frame.time for frame in frames] == pytest.approx([float(i / rate) for i in range(130)])
        for index, value in [(0, 0), (2, 100), (21, 100), (22, 0), (100, 200), (119, 200), (120, 0)]:
            assert frames[index].to_ndarray(format="rgb24").mean() == pytest.approx(value, abs=3)
        assert [entry["payload"]["start"] for entry in json.loads(container.metadata["comment"])] == [2, 100]


@pytest.mark.parametrize(
    ("ranges", "fps", "total", "width"),
    [
        ([(2, 4)], 0, 10, 16),
        ([(2, 4)], -5, 10, 16),
        ([(2, 4)], 5, -1, 16),
        ([(-1, 2)], 5, 10, 16),
        ([(2, 2)], 5, 10, 16),
        ([(4, 2)], 5, 10, 16),
        ([(2, 11)], 5, 10, 16),
        ([(3, 5), (2, 4)], 5, 10, 16),
        ([(2, 4)], 5, 10, 0),
    ],
)
def test_write_video_tracks_validates_before_consuming_frames(tmp_path, ranges, fps, total, width):
    def frames():
        pytest.fail("invalid track metadata must be rejected before reading frames")
        yield

    output = tmp_path / "existing.mkv"
    output.write_bytes(b"original")
    tracks = [VideoTrack(start, end, width, 16, frames()) for start, end in ranges]
    with pytest.raises(ValueError, match="source_fps|source_frames|dimensions"):
        write_video_tracks(tracks, output, fps=fps, source_frames=total)
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("count", [1, 3])
def test_write_video_tracks_preserves_output_on_frame_count_mismatch(tmp_path, count):
    output = tmp_path / "existing.mkv"
    output.write_bytes(b"original")
    track = VideoTrack(0, 2, 16, 16, [np.zeros((16, 16, 3), np.uint8)] * count)
    with pytest.raises(ValueError, match="declares 2 frames"):
        write_video_tracks([track], output, fps=5, source_frames=2)
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("extension", ["mp4", "mkv"])
def test_write_video_crops_keeps_native_frames_and_exact_crop_pixels(tmp_path, monkeypatch, extension):
    source, output = tmp_path / f"source.{extension}", tmp_path / "crops.mkv"
    rate = Fraction(30000, 1001)
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=rate, options={"crf": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(40):
            gray = (np.indices((48, 64)).sum(axis=0) + index + 60).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.repeat(gray[:, :, None], 3, axis=2), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1) / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    with av.open(str(source)) as container:
        originals = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    def reject_numpy_padding(*args, **kwargs):
        pytest.fail("source crops must not pass through NumPy RGB frames")

    monkeypatch.setattr(np, "pad", reject_numpy_padding)
    crops = [VideoCrop(20, 35, 10, 8, 32, 24, {"id": 1}), VideoCrop(2, 8, 3, 5, 31, 23, {"id": 0})]
    write_video_crops(source, crops, output)
    with av.open(str(output)) as container:
        assert container.metadata["SOURCE_FPS"] == str(rate)
        assert container.metadata["SOURCE_FRAMES"] == "40"
        for stream, crop in zip(container.streams.video, reversed(crops), strict=True):
            assert (stream.width, stream.height) == (32, 24)
            container.seek(stream.start_time, stream=stream)
            frames = list(container.decode(stream))
            assert len(frames) == crop.end_frame - crop.start_frame
            assert frames[0].time == pytest.approx(float(crop.start_frame / rate), abs=0.001)
            for index, frame in enumerate(frames, crop.start_frame):
                rgb = frame.to_ndarray(format="rgb24")
                expected = originals[index][crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]
                assert np.abs(rgb[:crop.height, :crop.width].astype(int) - expected).mean() < 4
                assert rgb[crop.height - 1, crop.width - 1].mean() == pytest.approx(expected[-1, -1].mean(), abs=5)
            assert json.loads(stream.metadata["PAYLOAD"]) == crop.payload


@pytest.mark.parametrize("crop", [VideoCrop(0, 2, -1, 0, 16, 16), VideoCrop(0, 2, 0, 0, 9999, 16)])
def test_write_video_crops_rejects_out_of_bounds_crops(tmp_path, crop):
    source = Path(__file__).parent / "assets" / "example.mp4"
    output = tmp_path / "crops.mkv"
    with pytest.raises(ValueError, match="crop must"):
        write_video_crops(source, [crop], output)
    assert not output.exists()
