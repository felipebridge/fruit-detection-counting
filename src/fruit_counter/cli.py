"""Command-line interface.

Exit codes: 0 success, 1 runtime failure (model/output), 2 invalid config or input,
130 interrupted.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from typing import Any

from fruit_counter import __version__
from fruit_counter.config import load_config
from fruit_counter.detector import YoloDetector
from fruit_counter.exceptions import ConfigError, FruitCounterError, InputError
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
    parser.add_argument(
        "-i", "--input", help="image, directory of images, video file, or camera index (e.g. 0)"
    )
    parser.add_argument("-c", "--config", help="YAML configuration file")
    parser.add_argument("-o", "--output-dir", help="root directory for run outputs")
    parser.add_argument(
        "--no-save-annotated",
        dest="save_annotated",
        action="store_false",
        default=None,
        help="do not write annotated images/videos",
    )
    parser.add_argument("-w", "--weights", help="path to YOLO weights")
    parser.add_argument("--device", help='inference device: "auto", "cpu", "cuda:0", "mps"')
    parser.add_argument("--conf", type=float, help="minimum detection confidence")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="inference image size (multiple of 32)")
    parser.add_argument(
        "--classes",
        nargs="*",
        metavar="NAME",
        help="class names to keep; pass the flag without names to keep all classes",
    )
    parser.add_argument(
        "--list-classes", action="store_true", help="print the model's class names and exit"
    )
    parser.add_argument("--stride", type=int, help="process every N-th video frame")
    parser.add_argument("--max-frames", type=int, help="stop after N processed frames")
    parser.add_argument("--min-hits", type=int, help="matches needed before a fruit is counted")
    parser.add_argument("--max-age", type=int, help="frames a lost track is kept alive")
    parser.add_argument("--show", action="store_true", help="display annotated frames live")
    parser.add_argument("-q", "--quiet", action="store_true", help="only log warnings and errors")
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Nested config overrides from CLI flags (``None`` means not given)."""
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
        "output": {"directory": args.output_dir, "save_annotated": args.save_annotated},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_classes and not args.input:
        parser.error("--input is required (or use --list-classes)")

    # Logs go to stderr so stdout only carries results.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%X"))
    package_logger = logging.getLogger("fruit_counter")
    package_logger.handlers[:] = [handler]
    package_logger.setLevel(logging.WARNING if args.quiet else logging.INFO)
    package_logger.propagate = False

    try:
        config = load_config(args.config, overrides_from_args(args))
        if args.list_classes:
            # The class filter is irrelevant here; don't fail on classes the model lacks.
            detector = YoloDetector(dataclasses.replace(config.model, classes=()))
            for class_id, name in sorted(detector.class_names.items()):
                print(f"{class_id:>4}  {name}")
            return EXIT_OK
        source = resolve_source(args.input)
        report = FruitCountingRunner(config, YoloDetector(config.model)).run(source, args.show)
    except (ConfigError, InputError) as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except FruitCounterError as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED

    print(format_report(report))
    return EXIT_OK


def format_report(report: RunReport) -> str:
    results = report.results
    lines = [f"Source: {report.source} ({report.source.kind.value})"]
    if report.source.is_stream:
        lines += [
            f"Frames processed:      {results['frames_processed']}",
            f"Unique fruits counted: {results['unique_fruit_count']}",
            f"Max visible in frame:  {results['max_visible_in_frame']}",
            f"Processing speed:      {results['processing_fps']} fps",
        ]
        if results["interrupted"]:
            lines.append("Note: processing was interrupted; results are partial.")
    else:
        lines.append(f"Fruits detected: {results['fruit_count']}")
        if "images_processed" in results:
            lines.append(f"Images processed: {results['images_processed']}")
    lines += [f"  - {name}: {count}" for name, count in results["counts_by_class"].items()]
    lines.append(f"Outputs: {report.output_dir}")
    return "\n".join(lines)
