import json

import av
import numpy as np
import pytest

from simple_video_utils.joining import merge_video_tracks


@pytest.mark.parametrize("gap_value", [0, 90])
def test_merge_video_tracks_preserves_timeline_and_payloads(tmp_path, gap_value):
    source, output = tmp_path / "source.mkvg", tmp_path / "merged.mp4"
    with av.open(str(source), mode="w", format="matroska") as container:
        container.metadata.update(source_fps="5", source_frames="10")
        clips = []
        for start, end, width, height, value in [(2, 4, 64, 48, 200), (6, 8, 32, 24, 150)]:
            stream = container.add_stream("libx264", rate=5)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.metadata.update(title=f"{start}-{end}", payload=f'{{"signer_id": {start}}}')
            clips.append((stream, end - start, width, height, value))
        for stream, count, width, height, value in clips:
            for _ in range(count):
                frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value, dtype=np.uint8), format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode())

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
        assert np.mean(colors[6][12:36, left : left + 32]) == pytest.approx(150, abs=5)
        assert np.mean(colors[6][0, 0]) < 5
        assert [entry["payload"]["signer_id"] for entry in json.loads(container.metadata["comment"])] == [2, 6]
