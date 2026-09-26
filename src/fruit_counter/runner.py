"""Runs detection and counting on an input source and writes the results.

Each run gets its own directory::

    outputs/<source>_<YYYYmmdd-HHMMSS>/
        summary.json              counts, detections or per-fruit statistics, config
        <name>_annotated.<ext>    annotated image(s) or video
        frames.csv                per-frame statistics (video / camera only)
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fruit_counter.config import AppConfig
from fruit_counter.detector import Detector, YoloDetector
from fruit_counter.exceptions import InputError, OutputError
from fruit_counter.pipeline import ImagePipeline, VideoPipeline
from fruit_counter.sources import (
    InputSource,
    SourceKind,
    list_images,
    open_video,
    read_frames,
    read_image,
)
from fruit_counter.visualization import Annotator

logger = logging.getLogger(__name__)

LOG_EVERY = 50
WINDOW_NAME = "Fruit counter"


@dataclass(frozen=True)
class RunReport:
    source: InputSource
    output_dir: Path
    results: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def fruit_count(self) -> int:
        """Fruits in the image(s), or unique fruits in the stream."""
        return int(self.results["unique_fruit_count" if self.source.is_stream else "fruit_count"])


class FruitCountingRunner:
    def __init__(self, config: AppConfig, detector: Detector | None = None) -> None:
        self._config = config
        self._detector = detector if detector is not None else YoloDetector(config.model)
        self._annotator = Annotator()

    def run(self, source: InputSource, show: bool = False) -> RunReport:
        """Process ``source``; ``show`` displays annotated video frames live."""
        output_dir = create_run_dir(Path(self._config.output.directory), source.name)
        logger.info("Processing %s '%s'", source.kind.value, source)
        try:
            if source.is_stream:
                results, files = self._run_stream(source, output_dir, show)
            else:
                results, files = self._run_images(source, output_dir)
            summary = {
                "source": {"kind": source.kind.value, "location": str(source)},
                "config": self._config.to_dict(),
                "results": results,
            }
            files["summary"] = write_json(output_dir / "summary.json", summary)
        except BaseException:
            # The directory was created for this run, so only its partial output is lost.
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        logger.info("Results written to %s", output_dir)
        return RunReport(source, output_dir, results, files)

    def _run_images(
        self, source: InputSource, output_dir: Path
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        assert source.path is not None
        paths = [source.path] if source.kind is SourceKind.IMAGE else list_images(source.path)
        pipeline = ImagePipeline(self._detector, self._annotator)
        files: dict[str, Path] = {}
        entries: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        totals: Counter[str] = Counter()

        for path in paths:
            try:
                image = read_image(path)
            except InputError as exc:
                if source.kind is SourceKind.IMAGE:
                    raise
                logger.warning("Skipping %s: %s", path.name, exc)
                skipped.append({"file": path.name, "reason": str(exc)})
                continue

            result = pipeline.process(image)
            totals.update(result.counts_by_class)
            entry: dict[str, Any] = {"file": path.name, **result.to_dict()}
            if self._config.output.save_annotated:
                annotated_path = output_dir / f"{path.stem}_annotated{path.suffix.lower()}"
                write_image(annotated_path, pipeline.annotate(image, result))
                entry["annotated_file"] = annotated_path.name
                files.setdefault("annotated", annotated_path)
            entries.append(entry)
            logger.info(
                "%s: %d fruit(s) %s (%.0f ms)",
                path.name,
                result.fruit_count,
                result.counts_by_class,
                result.inference_ms,
            )

        if source.kind is SourceKind.IMAGE:
            return entries[0], files
        # Images are independent, so the total is the sum of per-image counts.
        results = {
            "fruit_count": sum(e["fruit_count"] for e in entries),
            "counts_by_class": dict(sorted(totals.items())),
            "images_processed": len(entries),
            "images_skipped": skipped,
            "images": entries,
        }
        return results, files

    def _run_stream(
        self, source: InputSource, output_dir: Path, show: bool
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        cfg = self._config
        stride = cfg.video.frame_stride
        pipeline = VideoPipeline(self._detector, cfg.tracking, self._annotator)
        csv_path = output_dir / "frames.csv"
        video_path = output_dir / f"{source.name}_annotated.mp4"
        video_writer: cv2.VideoWriter | None = None
        interrupted = False
        started = time.perf_counter()

        capture, fps = open_video(source)
        frames = read_frames(
            capture, fps, stride, cfg.video.max_frames, wall_clock=source.kind is SourceKind.CAMERA
        )
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer: csv.DictWriter[str] | None = None
                for index, timestamp, image in frames:
                    result = pipeline.process_frame(image, index, timestamp)
                    row = result.to_row()
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)

                    if cfg.output.save_annotated or show:
                        annotated = pipeline.annotate(image, result)
                        if cfg.output.save_annotated:
                            if video_writer is None:
                                video_writer = open_video_writer(video_path, fps / stride, image)
                            video_writer.write(annotated)
                        if show:
                            try:
                                cv2.imshow(WINDOW_NAME, annotated)
                                key = cv2.waitKey(1) & 0xFF
                            except cv2.error as exc:
                                logger.warning("Live preview unavailable: %s", exc)
                                show = False
                            else:
                                if key in (ord("q"), 27):  # q or Esc
                                    logger.info("Stopped by user")
                                    interrupted = True
                                    break

                    processed = pipeline.frames_processed
                    if processed % LOG_EVERY == 0:
                        logger.info(
                            "frame %d | visible %d | unique %d | %.1f fps",
                            processed,
                            result.visible_count,
                            result.unique_count,
                            processed / (time.perf_counter() - started),
                        )
        except KeyboardInterrupt:
            logger.warning("Interrupted; saving partial results")
            interrupted = True
        finally:
            capture.release()
            if video_writer is not None:
                video_writer.release()
            if show:
                with suppress(cv2.error):
                    cv2.destroyWindow(WINDOW_NAME)

        summary = pipeline.summary()
        if summary["frames_processed"] == 0:
            raise InputError(f"No frames could be read from {source}")
        elapsed = time.perf_counter() - started
        files = {"frame_stats": csv_path}
        if video_writer is not None:
            files["annotated"] = video_path
        results = {
            **summary,
            "source_fps": round(fps, 3),
            "frame_stride": stride,
            "processing_time_s": round(elapsed, 2),
            "processing_fps": round(summary["frames_processed"] / elapsed, 2),
            "interrupted": interrupted,
        }
        logger.info(
            "Done: %d frame(s), %d unique fruit(s) %s",
            summary["frames_processed"],
            summary["unique_fruit_count"],
            summary["counts_by_class"],
        )
        return results, files


def create_run_dir(root: Path, source_name: str) -> Path:
    """Create ``root/<source_name>_<timestamp>``, with a numeric suffix on collision."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._") or "run"
    base = f"{safe_name}_{datetime.now():%Y%m%d-%H%M%S}"
    path = root / base
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{base}-{suffix}"
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        raise OutputError(f"Cannot create output directory {path}: {exc}") from exc
    return path


def write_json(path: Path, data: dict[str, Any]) -> Path:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, TypeError) as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


def write_image(path: Path, image: np.ndarray) -> Path:
    # imencode + tofile instead of cv2.imwrite, which fails on non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise OutputError(f"Cannot encode image for {path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


def open_video_writer(path: Path, fps: float, first_frame: np.ndarray) -> cv2.VideoWriter:
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OutputError(f"Cannot open video writer for {path}")
    return writer
