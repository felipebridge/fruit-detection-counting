"""In-memory pipelines for single images and video streams (no file I/O)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from fruit_counter.config import TrackingConfig
from fruit_counter.counting import UniqueCounter, count_by_class
from fruit_counter.detector import Detection, Detector
from fruit_counter.exceptions import InputError
from fruit_counter.tracker import FruitTracker, TrackedDetection
from fruit_counter.visualization import Annotator


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Return ``frame`` as a contiguous ``HxWx3`` uint8 array (grayscale/BGRA converted)."""
    if not isinstance(frame, np.ndarray):
        raise InputError(f"Expected a numpy array, got {type(frame).__name__}")
    if frame.size == 0:
        raise InputError("Received an empty frame")
    if frame.dtype != np.uint8:
        raise InputError(f"Expected a uint8 image, got dtype {frame.dtype}")
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    raise InputError(f"Unsupported frame shape {frame.shape}")


@dataclass(frozen=True)
class ImageResult:
    detections: list[Detection]
    width: int
    height: int
    inference_ms: float

    @property
    def fruit_count(self) -> int:
        return len(self.detections)

    @property
    def counts_by_class(self) -> dict[str, int]:
        return count_by_class(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fruit_count": self.fruit_count,
            "counts_by_class": self.counts_by_class,
            "image_width": self.width,
            "image_height": self.height,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [det.to_dict() for det in self.detections],
        }


class ImagePipeline:
    """Counts the fruits visible in independent images."""

    def __init__(self, detector: Detector, annotator: Annotator | None = None) -> None:
        self._detector = detector
        self._annotator = annotator or Annotator()

    def process(self, image: np.ndarray) -> ImageResult:
        frame = ensure_bgr(image)
        start = time.perf_counter()
        detections = self._detector.detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        height, width = frame.shape[:2]
        return ImageResult(detections, width, height, elapsed_ms)

    def annotate(self, image: np.ndarray, result: ImageResult) -> np.ndarray:
        """Draw ``result`` on a BGR ``image`` in place."""
        summary = [f"Fruits detected: {result.fruit_count}"]
        summary += [f"  {name}: {count}" for name, count in result.counts_by_class.items()]
        return self._annotator.draw(image, result.detections, summary_lines=summary)


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp_s: float
    tracked: list[TrackedDetection]
    unique_count: int  # distinct fruits counted so far in the stream
    newly_counted: list[int]
    inference_ms: float

    @property
    def visible_count(self) -> int:
        return len(self.tracked)

    @property
    def tracked_count(self) -> int:
        return sum(1 for item in self.tracked if item.track_id is not None)

    def to_row(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "visible_count": self.visible_count,
            "tracked_count": self.tracked_count,
            "unique_count": self.unique_count,
            "new_ids": " ".join(str(i) for i in self.newly_counted),
            "inference_ms": round(self.inference_ms, 2),
        }


class VideoPipeline:
    """Counts unique fruits in one stream of frames; use a new instance per stream."""

    def __init__(
        self,
        detector: Detector,
        tracking: TrackingConfig | None = None,
        annotator: Annotator | None = None,
    ) -> None:
        self._detector = detector
        self._annotator = annotator or Annotator()
        self._tracker = FruitTracker(tracking)
        self._counter = UniqueCounter()
        self.frames_processed = 0
        self._visible_sum = 0
        self._max_visible = 0
        self._inference_ms_sum = 0.0

    def process_frame(
        self, frame: np.ndarray, frame_index: int | None = None, timestamp_s: float = 0.0
    ) -> FrameResult:
        index = self.frames_processed if frame_index is None else frame_index
        image = ensure_bgr(frame)
        start = time.perf_counter()
        detections = self._detector.detect(image)
        inference_ms = (time.perf_counter() - start) * 1000

        tracked = self._tracker.update(detections)
        new_ids = self._counter.update(index, tracked)

        self.frames_processed += 1
        self._visible_sum += len(detections)
        self._max_visible = max(self._max_visible, len(detections))
        self._inference_ms_sum += inference_ms
        return FrameResult(index, timestamp_s, tracked, self._counter.total, new_ids, inference_ms)

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw ``result`` on a BGR ``frame`` in place."""
        summary = [
            f"Unique fruits counted: {result.unique_count}",
            f"Visible now: {result.visible_count}",
            f"Frame: {result.frame_index}",
        ]
        return self._annotator.draw(
            frame,
            [item.detection for item in result.tracked],
            track_ids=[item.track_id for item in result.tracked],
            summary_lines=summary,
        )

    def summary(self) -> dict[str, Any]:
        frames = self.frames_processed or 1
        return {
            "frames_processed": self.frames_processed,
            "unique_fruit_count": self._counter.total,
            "counts_by_class": self._counter.counts_by_class,
            "max_visible_in_frame": self._max_visible,
            "mean_visible_per_frame": round(self._visible_sum / frames, 3),
            # The naive per-frame total, reported to show why tracking is needed.
            # It is not a fruit count.
            "detections_summed_over_frames": self._visible_sum,
            "mean_inference_ms": round(self._inference_ms_sum / frames, 2),
            "objects": [obj.to_dict() for obj in self._counter.objects],
        }
