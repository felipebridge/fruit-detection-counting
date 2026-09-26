"""Application layer: wires input sources, pipelines and output writers together.

The CLI is a thin wrapper around :class:`FruitCountingRunner`; the same class can be
used from notebooks, batch jobs or a web service.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fruit_counter import __version__
from fruit_counter.config import AppConfig
from fruit_counter.detection import Detector, YoloDetector
from fruit_counter.exceptions import InputError
from fruit_counter.outputs import (
    CsvStreamWriter,
    VideoFileWriter,
    create_run_dir,
    write_image,
    write_json,
)
from fruit_counter.pipeline import FrameResult, ImagePipeline, VideoPipeline
from fruit_counter.sources import InputSource, SourceKind, VideoStream, list_images, read_image
from fruit_counter.visualization import Annotator

logger = logging.getLogger(__name__)

SUMMARY_FILE = "summary.json"
FRAMES_FILE = "frames.csv"


@dataclass(frozen=True)
class RunReport:
    """What a run produced."""

    source: InputSource
    output_dir: Path
    results: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def fruit_count(self) -> int:
        """Headline number: fruits in the image(s), or unique fruits in the stream."""
        key = "unique_fruit_count" if self.source.is_stream else "fruit_count"
        return int(self.results[key])


class LivePreview:
    """Optional OpenCV window; degrades gracefully on headless systems."""

    WINDOW_NAME = "Fruit Detection Counting"

    def __init__(self) -> None:
        self.stop_requested = False
        self._available = True
        self._opened = False

    def show(self, frame: np.ndarray) -> None:
        if not self._available:
            return
        try:
            cv2.imshow(self.WINDOW_NAME, frame)
            self._opened = True
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            logger.warning("Live preview unavailable (%s); continuing without it", exc)
            self._available = False
            return
        if key in (ord("q"), 27):  # q or Esc
            self.stop_requested = True

    def close(self) -> None:
        if self._opened:
            with suppress(cv2.error):
                cv2.destroyWindow(self.WINDOW_NAME)


class FruitCountingRunner:
    """Runs detection and counting on an :class:`InputSource` and saves the results.

    Args:
        config: Application configuration.
        detector: Optional detector instance; a :class:`YoloDetector` is built from
            ``config.model`` when omitted.
    """

    def __init__(self, config: AppConfig, detector: Detector | None = None) -> None:
        self._config = config
        self._detector = detector if detector is not None else YoloDetector(config.model)
        self._annotator = Annotator(config.visualization)

    def run(
        self, source: InputSource, run_name: str | None = None, show: bool = False
    ) -> RunReport:
        """Process ``source`` and write all artefacts to a new run directory.

        Args:
            source: Validated input (see :func:`fruit_counter.sources.resolve_source`).
            run_name: Fixed run directory name; defaults to ``<source>_<timestamp>``.
            show: Display annotated frames live (video / camera only).
        """
        output_dir = create_run_dir(self._config.output.directory, source.name, run_name)
        logger.info("Processing %s '%s'", source.kind.value, source)
        try:
            if source.is_stream:
                report = self._run_stream(source, output_dir, show)
            else:
                assert source.path is not None
                paths = (
                    [source.path] if source.kind is SourceKind.IMAGE else list_images(source.path)
                )
                report = self._run_images(source, paths, output_dir)
        except BaseException:
            # Don't leave empty run directories behind (e.g. camera could not be opened).
            if not any(output_dir.iterdir()):
                output_dir.rmdir()
            raise
        logger.info("Results written to %s", output_dir)
        return report

    # ------------------------------------------------------------------ images

    def _run_images(self, source: InputSource, paths: list[Path], output_dir: Path) -> RunReport:
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
            results = entries[0]
        else:
            # Images are independent: the total is the sum of per-image counts.
            results = {
                "fruit_count": sum(e["fruit_count"] for e in entries),
                "counts_by_class": dict(sorted(totals.items())),
                "images_processed": len(entries),
                "images_skipped": skipped,
                "images": entries,
            }
        files["summary"] = write_json(output_dir / SUMMARY_FILE, self._envelope(source, results))
        return RunReport(source, output_dir, results, files)

    # ------------------------------------------------------------------ video / camera

    def _run_stream(self, source: InputSource, output_dir: Path, show: bool) -> RunReport:
        cfg = self._config
        pipeline = VideoPipeline(self._detector, cfg.tracking, self._annotator)
        preview = LivePreview() if show else None
        interrupted = False
        started = time.perf_counter()

        with ExitStack() as stack:
            stream = stack.enter_context(VideoStream(source))
            video_writer = stack.enter_context(
                VideoFileWriter(
                    output_dir / f"{source.name}_annotated.mp4",
                    stream.fps / cfg.video.frame_stride,
                    cfg.output.video_codec,
                )
            )
            csv_writer = stack.enter_context(CsvStreamWriter(output_dir / FRAMES_FILE))
            if preview is not None:
                stack.callback(preview.close)

            expected = self._expected_frames(stream.frame_count)
            logger.info(
                "Stream: %dx%d @ %.2f fps, %s frame(s) to process",
                stream.width,
                stream.height,
                stream.fps,
                expected if expected is not None else "unknown number of",
            )
            try:
                for frame in stream.frames(cfg.video.frame_stride, cfg.video.max_frames):
                    result = pipeline.process_frame(frame.image, frame.index, frame.timestamp_s)
                    if cfg.output.save_frame_stats:
                        csv_writer.write(result.to_row())
                    if cfg.output.save_annotated or preview is not None:
                        annotated = pipeline.annotate(frame.image, result)
                        if cfg.output.save_annotated:
                            video_writer.write(annotated)
                        if preview is not None:
                            preview.show(annotated)
                            if preview.stop_requested:
                                logger.info("Stopped by user")
                                interrupted = True
                                break
                    self._log_progress(pipeline, result, expected, started)
            except KeyboardInterrupt:
                logger.warning("Interrupted; saving partial results")
                interrupted = True
            source_fps = stream.fps

        summary = pipeline.summary()
        if summary.frames_processed == 0:
            raise InputError(f"No frames could be read from {source}")
        elapsed = time.perf_counter() - started
        files: dict[str, Path] = {}
        if video_writer.frames_written:
            files["annotated"] = video_writer.path
        if csv_writer.rows_written:
            files["frame_stats"] = csv_writer.path
        results = {
            **summary.to_dict(),
            "source_fps": round(source_fps, 3),
            "frame_stride": cfg.video.frame_stride,
            "processing_time_s": round(elapsed, 2),
            "processing_fps": round(summary.frames_processed / elapsed, 2),
            "interrupted": interrupted,
        }
        files["summary"] = write_json(output_dir / SUMMARY_FILE, self._envelope(source, results))
        logger.info(
            "Done: %d frame(s), %d unique fruit(s) %s",
            summary.frames_processed,
            summary.unique_fruit_count,
            summary.counts_by_class,
        )
        return RunReport(source, output_dir, results, files)

    def _expected_frames(self, frame_count: int | None) -> int | None:
        max_frames = self._config.video.max_frames
        if frame_count is None:
            return max_frames
        expected = -(-frame_count // self._config.video.frame_stride)  # ceil division
        return min(expected, max_frames) if max_frames else expected

    def _log_progress(
        self, pipeline: VideoPipeline, result: FrameResult, expected: int | None, started: float
    ) -> None:
        processed = pipeline.frames_processed
        if processed % self._config.video.log_every:
            return
        fps = processed / max(time.perf_counter() - started, 1e-9)
        logger.info(
            "frame %d%s | visible %d | unique %d | %.1f fps",
            processed,
            f"/{expected}" if expected else "",
            result.visible_count,
            result.unique_count,
            fps,
        )

    # ------------------------------------------------------------------ metadata

    def _envelope(self, source: InputSource, results: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool": {"name": "fruit-detection-counting", "version": __version__},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": {"kind": source.kind.value, "location": str(source)},
            "config": self._config.to_dict(),
            "results": results,
        }
