"""Compare chapter export against a Git ref, outside setup/import/metadata costs.

Run from the repository: PYTHONPATH=. python benchmarks/chapter_export.py video.mp4
Add --size 256 to resize every chapter; otherwise retain varying native sizes.
"""

import argparse
import importlib.util
import inspect
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av

from simple_video_utils import chapters
from simple_video_utils.metadata import video_metadata


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--baseline", default="v0.9.4")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--chapters", type=int, default=36)
    parser.add_argument("--size", type=int)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, help="Native threads per stage; 1 for a single-CPU worker")
    args = parser.parse_args()
    if min(args.frames, args.chapters, args.runs) <= 0:
        parser.error("frames, chapters and runs must be positive")
    if args.size is not None and (args.size <= 0 or args.size % 16):
        parser.error("size must be a positive multiple of 16")
    if args.threads is not None and args.threads < 0:
        parser.error("threads must be nonnegative")
    return parser, args


def main():
    parser, args = _arguments()
    meta = video_metadata(args.source)
    total = min(args.frames, meta.nb_frames or 0)
    count = min(args.chapters, total)
    if not count:
        parser.error("source has no frames")
    width, height = meta.width, meta.height
    boxes = [
        (-32, -32, 288, 288),
        tuple(value // 2 * 2 for value in (width // 2 - 224, height // 2 - 224,
                                          width // 2 + 224, height // 2 + 224)),
        (width - 520, height - 480, width + 120, height + 160),
    ]
    resolution = (args.size, args.size) if args.size is not None else None
    clips = [chapters.ChapterClip(i * total // count, (i + 1) * total // count, boxes[i % 3], resolution=resolution)
             for i in range(count)]
    results = {"baseline": [], "current": []}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        baseline_path = path / "baseline.py"
        baseline_code = subprocess.check_output(
            ["git", "show", f"{args.baseline}:simple_video_utils/chapters.py"], text=True)
        if args.threads is not None:
            # Older releases have no thread-budget argument; apply the same native settings to both versions.
            for marker, setting in [
                ("stream.thread_type = thread_type", "        stream.codec_context.thread_count"),
                ("graph = av.filter.Graph()", "    graph.threads"),
                ('encoder = av.CodecContext.create("libx264", "w")', "    encoder.thread_count"),
            ]:
                baseline_code = baseline_code.replace(marker, f"{marker}\n{setting} = {args.threads}")
        baseline_path.write_text(baseline_code)
        spec = importlib.util.spec_from_file_location("baseline_chapters", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        modules = {"baseline": baseline, "current": chapters}
        for iteration in range(args.runs + 1):
            for name in list(modules) if iteration % 2 == 0 else list(reversed(modules)):
                output = path / f"{name}.mkv"
                output.unlink(missing_ok=True)
                writer = modules[name].write_chapters
                kwargs = {}
                if args.threads is not None:
                    kwargs["thread_type"] = "SLICE" if args.threads == 1 else "AUTO"
                    if "threads" in inspect.signature(writer).parameters:
                        kwargs["threads"] = args.threads
                before = resource.getrusage(resource.RUSAGE_SELF)
                start = time.perf_counter()
                writer(args.source, output, clips, preset="veryfast", **kwargs)
                wall = time.perf_counter() - start
                after = resource.getrusage(resource.RUSAGE_SELF)
                if iteration:
                    results[name].append({"wall": wall,
                                          "cpu": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime})
        for name in modules:
            with av.open(str(path / f"{name}.mkv")) as container:
                assert len(container.chapters()) == count
                assert sum(1 for _ in container.decode(video=0)) == total
    summary = {name: {key: {"median": statistics.median(v[key] for v in runs),
                            "min": min(v[key] for v in runs), "max": max(v[key] for v in runs)}
                      for key in ("wall", "cpu")} for name, runs in results.items()}
    print(json.dumps({"baseline_ref": args.baseline, "threads": args.threads, "frames": total, "chapters": count,
                      "timings": summary, "runs": results}, indent=2))


if __name__ == "__main__":
    main()
