import json
from fractions import Fraction

import av
import numpy as np
import pytest

from simple_video_utils import tracks
from simple_video_utils.tracks import VideoCrop, merge_video_tracks, write_video_crops


@pytest.fixture(params=["mp4", "mkv"])
def source(tmp_path, request):
    path = tmp_path / f"source.{request.param}"
    rate = Fraction(30000, 1001)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate, options={"crf": "0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(40):
            gray = (np.indices((48, 64)).sum(axis=0) + index + 60).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.repeat(gray[:, :, None], 3, axis=2), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1) / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path, rate


@pytest.mark.parametrize("gap_color", ["black", "white"])
def test_crops_round_trip_on_source_timeline(source, tmp_path, gap_color):
    path, rate = source
    cropped, merged = tmp_path / "crops.mkv", tmp_path / "merged.mp4"
    with av.open(str(path)) as container:
        originals = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    crops = [VideoCrop(20, 35, 0, 0, 48, 32, {"id": 1}), VideoCrop(2, 8, 3, 5, 31, 23, {"id": 0})]
    write_video_crops(path, crops, cropped)
    with av.open(str(cropped)) as container:
        assert container.metadata["SOURCE_FPS"] == str(rate)
        assert container.metadata["SOURCE_FRAMES"] == "40"
        for stream, crop in zip(container.streams.video, reversed(crops), strict=True):
            assert (stream.width, stream.height) == (crop.width + crop.width % 2, crop.height + crop.height % 2)
            assert stream.metadata["title"] == f"{crop.start_frame}-{crop.end_frame}"
            assert json.loads(stream.metadata["PAYLOAD"]) == crop.payload
            container.seek(stream.start_time, stream=stream)
            frames = list(container.decode(stream))
            assert len(frames) == crop.end_frame - crop.start_frame
            assert frames[0].time == pytest.approx(float(crop.start_frame / rate), abs=0.001)
            for index, frame in enumerate(frames, crop.start_frame):
                rgb = frame.to_ndarray(format="rgb24")
                expected = originals[index][crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]
                assert np.abs(rgb[:crop.height, :crop.width].astype(int) - expected).mean() < 4
                assert rgb[crop.height - 1, crop.width - 1].mean() == pytest.approx(expected[-1, -1].mean(), abs=5)
            if crop.width % 2:
                assert rgb[:, -1].mean() < 10

    merge_video_tracks(cropped, merged, gap_color=gap_color)
    with av.open(str(merged)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 40
        assert [frame.time for frame in frames] == pytest.approx([float(i / rate) for i in range(40)])
        for index in [0, 1, 8, 19, 35, 39]:
            assert frames[index].to_ndarray(format="rgb24").mean() == pytest.approx(
                255 if gap_color == "white" else 0, abs=3
            )
        for index in [2, 7]:
            rgb = frames[index].to_ndarray(format="rgb24")
            expected = originals[index][5:28, 3:34]
            assert np.abs(rgb[4:27, 8:39].astype(int) - expected).mean() < 5
            assert rgb[0, 0].mean() < 5
        for index in [20, 34]:
            rgb = frames[index].to_ndarray(format="rgb24")
            assert np.abs(rgb.astype(int) - originals[index][:32, :48]).mean() < 5
        assert [entry["payload"]["id"] for entry in json.loads(container.metadata["comment"])] == [0, 1]


@pytest.mark.parametrize("crops", [
    [],
    [VideoCrop(-1, 2, 0, 0, 16, 16)],
    [VideoCrop(2, 2, 0, 0, 16, 16)],
    [VideoCrop(0, 41, 0, 0, 16, 16)],
    [VideoCrop(0, 2, 0, 0, 16, 16), VideoCrop(1, 3, 0, 0, 16, 16)],
    [VideoCrop(0, 2, -1, 0, 16, 16)],
    [VideoCrop(0, 2, 0, 0, 0, 16)],
    [VideoCrop(0, 2, 0, 0, 65, 16)],
])
def test_invalid_crops_preserve_output(source, tmp_path, crops):
    output = tmp_path / "existing.mkv"
    output.write_bytes(b"original")
    with pytest.raises(ValueError, match="crop|track ranges"):
        write_video_crops(source[0], crops, output)
    assert output.read_bytes() == b"original"


def test_truncated_source_preserves_output(source, tmp_path, monkeypatch):
    get_metadata = tracks.video_metadata_from_container
    monkeypatch.setattr(tracks, "video_metadata_from_container", lambda c: get_metadata(c)._replace(nb_frames=42))
    output = tmp_path / "existing.mkv"
    output.write_bytes(b"original")
    with pytest.raises(ValueError, match="decoded 1 frames"):
        write_video_crops(source[0], [VideoCrop(39, 42, 0, 0, 16, 16)], output)
    assert output.read_bytes() == b"original"


def test_merge_rejects_untagged_video(source, tmp_path):
    output = tmp_path / "merged.mp4"
    with pytest.raises(ValueError, match="gapped_tracks"):
        merge_video_tracks(source[0], output)
    assert not output.exists()
