import json

import av
import numpy as np
import pytest

from simple_video_utils.joining import VideoTrack, merge_video_tracks, write_video_tracks


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
