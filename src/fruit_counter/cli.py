"""Command-line interface.

Examples::

    fruit-counter --input data/samples/fruit_bowl.jpg
    fruit-counter --input orchard.mp4 --conf 0.3 --device cuda:0
    fruit-counter --input 0 --show            # webcam
    fruit-counter --list-classes --weights models/custom.pt

Exit codes: 0 success, 1 runtime failure (model/output), 2 invalid config or input,
130 interrupted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from fruit_counter import __version__
from fruit_counter.config import AppConfig, ModelConfig, load_config
from fruit_counter.detection import Detector, YoloDetector
from fruit_counter.exceptions import ConfigError, FruitCounterError, InputError
from fruit_counter.logging_utils import configure_logging
from fruit_counter.runner import FruitCountingRunner, RunReport
from fruit_counter.sources import resolve_source

logger = logging.getLogger("fruit_counter.cli")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fruit-counter",
        description="Detect and count fruits in images, videos and camera streams.",
        epilog="Settings not given on the command line come from --config or built-in defaults.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io = parser.add_argument_group("input / output")
    io.add_argument(
        "-i",
        "--input",
        help="image file, directory of images, video file, or camera index (e.g. 0)",
    )
    io.add_argument("-c", "--config", help="YAML configuration file")
    io.add_argument("-o", "--output-dir", help="root directory for run outputs")
    io.add_argument("--run-name", help="fixed run directory name (default: <input>_<time>)")
    io.add_argument(
        "--no-save-annotated",
        dest="save_annotated",
        action="store_false",
        default=None,
        help="do not write annotated images/videos",
    )
    io.add_argument(
        "--no-frame-stats",
        dest="save_frame_stats",
        action="store_false",
        default=None,
        help="do not write per-frame CSV statistics",
    )

    model = parser.add_argument_group("model")
    model.add_argument("-w", "--weights", help="path to YOLO weights")
    model.add_argument("--device", help='inference device: "auto", "cpu", "cuda:0", "mps"')
    model.add_argument("--conf", type=float, help="minimum detection confidence")
    model.add_argument("--iou", type=float, help="NMS IoU threshold")
    model.add_argument("--imgsz", type=int, help="inference image size (multiple of 32)")
    model.add_argument(
        "--classes",
        nargs="*",
        metavar="NAME",
        help="class names to keep; pass the flag without names to keep all classes",
    )
    model.add_argument(
        "--list-classes", action="store_true", help="print the model's class names and exit"
    )

    video = parser.add_argument_group("video / tracking")
    video.add_argument("--stride", type=int, help="process every N-th frame")
    video.add_argument("--max-frames", type=int, help="stop after N processed frames")
    video.add_argument("--min-hits", type=int, help="matches needed before a fruit is counted")
    video.add_argument("--max-age", type=int, help="frames a lost track is kept alive")
    video.add_argument("--show", action="store_true", help="display annotated frames live")

    misc = parser.add_argument_group("output format")
    misc.add_argument("--json", action="store_true", help="print results as JSON on stdout")
    misc.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI flags to a nested config override mapping (``None`` = not given)."""
    classes = None if args.classes is None else [c for c in args.classes if c.strip()]
    return {
        "model": {
            "weights": args.weights,
            "device": args.device,
            "confidence": args.conf,
            "iou": args.iou,
            "image_size": args.imgsz,
            "classes": classes,
        },
        "tracking": {"min_hits": args.min_hits, "max_age": args.max_age},
        "video": {"frame_stride": args.stride, "max_frames": args.max_frames},
        "output": {
            "directory": args.output_dir,
            "save_annotated": args.save_annotated,
            "save_frame_stats": args.save_frame_stats,
        },
    }


def build_detector(config: ModelConfig) -> Detector:
    """Factory for the detector used by the CLI (a seam for tests and new backends)."""
    return YoloDetector(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if not args.list_classes and not args.input:
        parser.error("--input is required (or use --list-classes)")

    try:
        config = load_config(args.config, overrides_from_args(args))
        if args.list_classes:
            return _list_classes(config)
        source = resolve_source(args.input)
        runner = FruitCountingRunner(config, build_detector(config.model))
        report = runner.run(source, run_name=args.run_name, show=args.show)
    except (ConfigError, InputError) as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except FruitCounterError as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED

    if args.json:
        print(json.dumps(_report_as_dict(report), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return EXIT_OK


def _list_classes(config: AppConfig) -> int:
    # Class filtering is irrelevant here; don't fail on classes the model lacks.
    model_config = dataclasses.replace(config.model, classes=())
    detector = build_detector(model_config)
    for class_id, name in sorted(detector.class_names.items()):
        print(f"{class_id:>4}  {name}")
    return EXIT_OK


def _report_as_dict(report: RunReport) -> dict[str, Any]:
    return {
        "source": str(report.source),
        "kind": report.source.kind.value,
        "output_dir": str(report.output_dir),
        "files": {key: str(path) for key, path in report.files.items()},
        "results": report.results,
    }


def format_report(report: RunReport) -> str:
    """Human-readable run summary."""
    results = report.results
    lines = [f"Source:  {report.source} ({report.source.kind.value})"]
    if report.source.is_stream:
        lines += [
            f"Frames processed:      {results['frames_processed']}",
            f"Unique fruits counted: {results['unique_fruit_count']}",
            f"Max visible in frame:  {results['max_visible_in_frame']}",
            f"Processing speed:      {results['processing_fps']} fps",
        ]
        if results.get("interrupted"):
            lines.append("Note: processing was interrupted; results are partial.")
    else:
        lines.append(f"Fruits detected: {results['fruit_count']}")
        if "images_processed" in results:
            lines.append(f"Images processed: {results['images_processed']}")
    for name, count in results["counts_by_class"].items():
        lines.append(f"  - {name}: {count}")
    lines.append(f"Outputs: {report.output_dir}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
