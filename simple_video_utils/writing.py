"""Encode frames into a video file or stream."""

import io
import os
from collections.abc import Iterable
from fractions import Fraction
from typing import BinaryIO

import av
import numpy as np
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

from simple_video_utils.video import quiet, refusing


def _chain(filters: str, frame: av.VideoFrame) -> av.filter.Graph:
    """`filters` as a graph that takes frames like `frame`."""
    graph = av.filter.Graph()
    source = graph.add_buffer(width=frame.width, height=frame.height, format=frame.format.name,
                              time_base=frame.time_base)
    # ponytail: a plain chain, split at its commas; a filter argument that holds a comma needs FFmpeg's own parser
    nodes = [graph.add(name, args or None) for name, _, args in (f.partition("=") for f in filters.split(","))]
    graph.link_nodes(source, *nodes, graph.add("buffersink")).configure()
    return graph


class VideoWriter:
    """Encodes frames into `dst`, a path or a writable binary stream, at `fps`. A context manager; nothing is written
    before the first frame.

    Frames are uint8 arrays, (H, W, 3) RGB, (H, W, 4) RGBA or (H, W) gray, all of one shape. `options` are the
    encoder's (`{"crf": "18", "preset": "fast"}`) and `container_options` the muxer's. `format` names the container
    when `dst` has no file name to tell. `filters` is an FFmpeg filter chain every frame passes before it is encoded
    (`"colorkey=0x00ff00:0.3:0.05,format=yuva420p"`). `tags` are the container's metadata.
    """

    def __init__(self, dst: str | os.PathLike | BinaryIO, fps: float | Fraction, *, codec: str = "h264",
                 pix_fmt: str = "yuv420p", options: dict[str, str] | None = None, format: str | None = None,
                 container_options: dict[str, str] | None = None, filters: str | None = None,
                 tags: dict[str, str] | None = None):
        quiet()
        self._rate = Fraction(fps).limit_denominator(65535)  # the finest clock MPEG-4 takes
        self._encoder, self._filters = (codec, pix_fmt, options), filters
        self._shape = self._stream = self._graph = None
        self._next = 0
        with refusing("Failed to open video for writing"):
            dst = os.fspath(dst) if isinstance(dst, os.PathLike) else dst
            self.container = av.open(dst, mode="w", format=format, options=container_options)
        self.container.metadata.update(tags or {})

    @refusing("Failed to write video")
    def write(self, frame: np.ndarray, index: int | None = None) -> None:
        """Add a frame, shown at `index / fps`: right after the one before it unless `index` leaves a gap."""
        if frame.shape != (self._shape or frame.shape):
            raise ValueError(f"frames have different shapes: {self._shape} and {frame.shape}")
        self._shape = frame.shape
        format = "gray" if frame.ndim == 2 else "rgba" if frame.shape[-1] == 4 else "rgb24"
        # image code hands over views (a BGR image flipped to RGB), which PyAV refuses
        image = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format=format)
        image.pts, image.time_base = self._next if index is None else index, 1 / self._rate
        self._next = image.pts + 1
        codec, pix_fmt, options = self._encoder
        if self._filters:
            # the chain ends in the encoder's format: a filter hands back RGB marked full range, which PyAV's own
            # conversion would then keep, writing a picture that decodes as black
            self._graph = self._graph or _chain(f"{self._filters},format={pix_fmt}", image)
            self._graph.push(image)
            image = self._graph.pull()
        if (stream := self._stream) is None:
            stream = self.container.add_stream(codec, rate=self._rate, options=options)
            stream.width, stream.height, stream.pix_fmt = image.width, image.height, pix_fmt
            if pix_fmt.startswith("yuv"):
                # The arrays are sRGB, and become YUV by BT.601 in limited range, here and in a filter chain alike.
                # Untagged, each player guesses a matrix and a transfer curve, and the colours shift.
                colour = stream.codec_context
                colour.colorspace, colour.color_range = Colorspace.ITU601, ColorRange.MPEG
                colour.color_primaries, colour.color_trc = ColorPrimaries.BT709, ColorTrc.IEC61966_2_1
        self.container.mux(stream.encode(image))
        self._stream = stream  # kept once it has encoded: PyAV crashes flushing an encoder that failed to open

    @refusing("Failed to write video")
    def close(self) -> None:
        """Flush the encoder and finish the file."""
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                self.container.mux(stream.encode())
        finally:
            self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def write_video(dst: str | os.PathLike | BinaryIO | None, frames: Iterable[np.ndarray], fps: float | Fraction,
                **settings) -> bytes | None:
    """Write `frames` through a `VideoWriter` with these `settings`. `dst=None` returns the file's bytes, as MP4
    unless `format` says otherwise."""
    buffer = io.BytesIO() if dst is None else None
    if buffer is not None:
        settings.setdefault("format", "mp4")
    with VideoWriter(dst if buffer is None else buffer, fps, **settings) as writer:
        for frame in frames:
            writer.write(frame)
    return buffer and buffer.getvalue()
