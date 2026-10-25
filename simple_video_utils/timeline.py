"""Seconds on a chaptered video <-> seconds on its source video.

Each chapter's payload ``start_frame`` places its clip on the source. Gapped files (written since 0.10) keep clips at
their source times, so the map is the identity; packed files (before 0.10) play clips back to back. Once no packed
files remain, every map is the identity and this module can go.
"""

import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np


class Timeline:
    """Source times before the first chapter or inside a gap land at the next chapter's start; past the last, at the
    file's end. A video without payload chapters maps to itself."""

    def __init__(self, chapters: Sequence[tuple[float, float, float]] = ()):
        """``chapters``: (file start, file end, source start) seconds each, e.g. ``Timeline.read(src).chapters``."""
        self.chapters = np.array(chapters, dtype=float).reshape(-1, 3)
        start, end, source = self.chapters.T
        if (end <= start).any() or (start[1:] < end[:-1]).any() or (np.diff(source) < 0).any():
            message = f"chapters must have positive length, not overlap, and run forward: {self.chapters.tolist()}"
            raise ValueError(message)

    @classmethod
    def read(cls, src: str | Path | BinaryIO | av.container.InputContainer) -> "Timeline":
        """Read the chapter table only; an open container is neither rewound nor decoded."""
        if not isinstance(src, av.container.InputContainer):
            with av.open(str(src) if isinstance(src, Path) else src) as container:
                return cls.read(container)
        rate = src.streams.video[0].average_rate
        chapters = sorted(src.chapters(), key=lambda chapter: chapter["start"] * chapter["time_base"])
        if not any(_payload_start(chapter, rate) is not None for chapter in chapters):
            return cls()  # ordinary chapter markers: the file is its own source
        return cls([(c["start"] * c["time_base"], c["end"] * c["time_base"], _source_start(c, rate)) for c in chapters])

    def to_file(self, t: float | np.ndarray) -> float | np.ndarray:
        """Source seconds on the file."""
        t = _finite(t)
        if not len(self.chapters):
            return t
        start, end, source = self.chapters.T
        t = np.clip(t, source[0], source[-1] + end[-1] - start[-1])
        # the first chapter ending at or after t; a gap before it collapses onto its start
        i = np.minimum(np.searchsorted(source + end - start, t), len(start) - 1)
        return _scalar(start[i] + np.maximum(t - source[i], 0))

    def to_source(self, t: float | np.ndarray) -> float | np.ndarray:
        """File seconds on the source."""
        t = _finite(t)
        if not len(self.chapters):
            return t
        start, end, source = self.chapters.T
        t = np.clip(t, start[0], end[-1])
        i = np.maximum(np.searchsorted(start, t, side="right") - 1, 0)
        return _scalar(source[i] + t - start[i])

    def window(self, start: float, end: float) -> tuple[float, float] | None:
        """A source window's file seconds, or None when it holds no video (wholly in a gap or outside the chapters)."""
        low, high = self.to_file(start), self.to_file(end)
        return (low, high) if low < high else None

    def squeeze(self, t: float) -> float:
        """File seconds with the gaps between chapters removed: a clock that runs only while there is video."""
        if not len(self.chapters):
            return t
        start, end, _ = self.chapters.T
        i = max(np.searchsorted(start, t, side="right") - 1, 0)
        return float(t - start[i] + (end[:i] - start[:i]).sum())

    def gaps(self) -> float:
        """Seconds between chapters that hold no frames."""
        start, end, _ = self.chapters.T
        return float((start[1:] - end[:-1]).sum())


def _finite(t: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(t, dtype=float)
    if not np.isfinite(array).all():
        message = f"times must be finite: {t}"
        raise ValueError(message)
    return _scalar(array)


def _scalar(t: np.ndarray) -> float | np.ndarray:
    return float(t) if np.ndim(t) == 0 else t


def _payload_start(chapter: dict, rate: Fraction | None) -> Fraction | None:
    tags = {key.lower(): value for key, value in chapter["metadata"].items()}
    try:
        payload = json.loads(tags.get("payload", "null"))
        if isinstance(payload, dict) and "start_frame" in payload:
            return Fraction(payload["start_frame"]) / Fraction(payload.get("rate", rate))
    except (ValueError, TypeError, ZeroDivisionError):
        pass  # unrelated payload tags retain the native chapter timeline
    return None


def _source_start(chapter: dict, rate: Fraction | None) -> Fraction:
    """A chapter's start on the source: its payload ``start_frame``, else its native start."""
    source = _payload_start(chapter, rate)
    return chapter["start"] * chapter["time_base"] if source is None else source
