import asyncio
import io

import av
import numpy as np
import pytest

from simple_video_utils.streaming import slice_video_stream


def _streaming_video(frames=36, fps=24) -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mpegts") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            array = np.full((48, 64, 3), i * 7 % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return output.getvalue()


def _frames(video: bytes) -> list[np.ndarray]:
    with av.open(io.BytesIO(video)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_clips_are_copied_before_video_eof():
    data = _streaming_video()
    source_frames = _frames(data)

    async def run():
        release = asyncio.Event()

        async def chunks():
            yield data[:len(data) * 3 // 4]
            await release.wait()
            yield data[len(data) * 3 // 4:]

        clips = slice_video_stream(chunks(), duration=0.5)
        try:
            first = await asyncio.wait_for(anext(clips), 2)
        finally:
            release.set()
        return [first, *[clip async for clip in clips]]

    clips = asyncio.run(run())
    decoded = [_frames(clip) for clip in clips]
    assert len(decoded) == 3
    assert [frames[0].shape[:2] for frames in decoded] == [(48, 64)] * 3
    for index, frames in enumerate(decoded):
        np.testing.assert_array_equal(frames[0], source_frames[index * 12])


def test_abandoned_stream_stops_demuxer():
    data = _streaming_video()

    async def run():
        async def chunks():
            yield data[:len(data) * 3 // 4]
            await asyncio.Event().wait()

        clips = slice_video_stream(chunks())
        assert await anext(clips)
        await asyncio.wait_for(clips.aclose(), 1)

    asyncio.run(run())


@pytest.mark.parametrize("duration", [0, -1, float("inf")])
def test_duration_must_be_positive_and_finite(duration):
    async def chunks():
        yield b"video"

    async def run():
        with pytest.raises(ValueError, match="duration"):
            await anext(slice_video_stream(chunks(), duration=duration))

    asyncio.run(run())
