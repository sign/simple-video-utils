"""Temporary patch to support anomalous historical "packed" chapters.

Packed chapters remove source-time gaps: source spans 0–5s and 10–15s are stored
at file times 0–5s and 5–10s. Their payloads retain the source offsets, which this
module uses to translate between file time and original video time.

The new writer generates "gapped" chapters instead: those same spans stay at
0–5s and 10–15s in both chapter metadata and frame timestamps, with no filler
frames in the gap. File time already matches original video time.

Once no packed files remain, remove this entire helper and simplify all utils
that use it to map time: operate directly on native timestamps instead. Keep
timestamp-based range selection and gap-preserving output timing.

The introducing diff and migration context are in:
https://github.com/sign/simple-video-utils/pull/82
"""

import json
from bisect import bisect_right
from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np

from simple_video_utils.metadata import _open_video


def _seconds(value) -> Fraction:
    return Fraction(str(value)) if isinstance(value, (float, np.floating)) else Fraction(value)


class Timeline:
    """Map file seconds to source-video seconds using chapter payloads.

    ``chapters`` contains (file start, file end, source start) triples. Within a
    chapter only the offset changes. Gaps interpolate between chapter boundaries;
    a removed source gap maps to one file boundary. Without chapters the map is identity.
    """

    def __init__(self, chapters: Sequence[tuple] = ()):
        self.chapters = tuple(tuple(_seconds(t) for t in chapter) for chapter in chapters)
        if any(len(c) != 3 or c[1] <= c[0] for c in self.chapters):
            message = "chapters must have (file start, file end, source start) with positive duration"
            raise ValueError(message)
        if any(left[1] > right[0] for left, right in pairwise(self.chapters)):
            message = "chapters must not overlap on file"
            raise ValueError(message)
        self._file = tuple(c[0] for c in self.chapters)
        self._source = tuple(c[2] for c in self.chapters)
        self._forward = all(left <= right for left, right in pairwise(self._source))

    @classmethod
    def read(cls, src: str | Path | BinaryIO | av.container.InputContainer) -> "Timeline":
        """Read chapter offsets, without seeking or closing an already-open container."""
        if not isinstance(src, av.container.InputContainer):
            with _open_video(str(src) if isinstance(src, Path) else src) as container:
                return cls.read(container)
        rate = src.streams.video[0].average_rate
        chapters = sorted(src.chapters(), key=lambda c: c["start"] * c["time_base"])
        return cls([(c["start"] * c["time_base"], c["end"] * c["time_base"], _source_start(c, rate))
                    for c in chapters])

    def _map(self, value, source: bool):
        if isinstance(value, np.ndarray):
            return np.array([float(self._map(t, source)) for t in value.flat]).reshape(value.shape)
        t = _seconds(value)
        if not self.chapters:
            return t
        starts = self._source if source else self._file
        index = max(bisect_right(starts, t) - 1, 0)
        start, end, original = self.chapters[index]
        low, high, output = (original, original + end - start, start) if source else (start, end, original)
        if t <= high or index == len(self.chapters) - 1:
            return output + t - low
        next_start, _, next_source = self.chapters[index + 1]
        next_input, next_output = (next_source, next_start) if source else (next_start, next_source)
        output_end = output + high - low
        return output_end + (t - high) * (next_output - output_end) / (next_input - high)

    def to_file(self, time: float | Fraction | np.ndarray):
        """Source-video seconds to file seconds; removed gaps land at the next chapter."""
        if not self._forward:
            message = "source-to-file mapping needs chapters in source order"
            raise ValueError(message)
        return self._map(time, True)

    def to_source(self, time: float | Fraction | np.ndarray):
        """File seconds to source-video seconds; a shared boundary belongs to the next chapter."""
        return self._map(time, False)


def _source_start(chapter: dict, rate: Fraction | None) -> Fraction:
    start = chapter["start"] * chapter["time_base"]
    tags = {key.lower(): value for key, value in chapter["metadata"].items()}
    try:
        payload = json.loads(tags.get("payload", "null"))
        if isinstance(payload, dict) and "start_frame" in payload:
            return Fraction(payload["start_frame"]) / Fraction(payload.get("rate", rate))
    except (ValueError, TypeError, ZeroDivisionError):
        pass  # unrelated payload tags retain the native chapter timeline
    return start
