"""Streaming video pipeline: detection -> tracking -> unique counting -> visualisation.

The pipeline is fed one frame at a time, so the same code serves video files,
camera streams or frames pushed by an API. It holds the tracking state of one
stream; create a new instance (or call :meth:`VideoPipeline.reset`) per stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fruit_counter.config import TrackingConfig
from fruit_counter.counting import UniqueCounter
from fruit_counter.detection.base import Detector
from fruit_counter.preprocessing import ensure_bgr
from fruit_counter.structures import TrackedDetection
from fruit_counter.tracking import FruitTracker
from fruit_counter.visualization import Annotator


@dataclass(frozen=True)
class FrameResult:
    """Per-frame output.

    Attributes:
        visible_count: Fruits detected in this frame (detection count).
        tracked_count: Detections in this frame that belong to a confirmed track.
        unique_count: Distinct fruits counted from the start of the stream up to
            and including this frame.
    """

    frame_index: int
    timestamp_s: float
    tracked: list[TrackedDetection]
    unique_count: int
    newly_counted: list[int]
    inference_ms: float

    @property
    def visible_count(self) -> int:
        return len(self.tracked)

    @property
    def tracked_count(self) -> int:
        return sum(1 for item in self.tracked if item.track_id is not None)

    def to_row(self) -> dict[str, Any]:
        """Flat representation for per-frame CSV statistics."""
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "visible_count": self.visible_count,
            "tracked_count": self.tracked_count,
            "unique_count": self.unique_count,
            "new_ids": " ".join(str(i) for i in self.newly_counted),
            "inference_ms": round(self.inference_ms, 2),
        }


@dataclass(frozen=True)
class VideoSummary:
    """Aggregate results for a whole stream."""

    frames_processed: int
    unique_fruit_count: int
    counts_by_class: dict[str, int]
    max_visible: int
    mean_visible: float
    detections_summed_over_frames: int
    mean_inference_ms: float
    objects: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "unique_fruit_count": self.unique_fruit_count,
            "counts_by_class": self.counts_by_class,
            "max_visible_in_frame": self.max_visible,
            "mean_visible_per_frame": round(self.mean_visible, 3),
            # Reported to make the gap between naive per-frame counting and unique
            # counting explicit. It is NOT a fruit count.
            "detections_summed_over_frames": self.detections_summed_over_frames,
            "mean_inference_ms": round(self.mean_inference_ms, 2),
            "objects": self.objects,
        }


class VideoPipeline:
    """Counts unique fruits across a sequence of frames."""

    def __init__(
        self,
        detector: Detector,
        tracking: TrackingConfig | None = None,
        annotator: Annotator | None = None,
    ) -> None:
        self._detector = detector
        self._tracking = tracking or TrackingConfig()
        self._annotator = annotator or Annotator()
        self.reset()

    def reset(self) -> None:
        """Forget all tracking and counting state (start a new stream)."""
        self._tracker = FruitTracker(self._tracking)
        self._counter = UniqueCounter()
        self._frames = 0
        self._visible_sum = 0
        self._max_visible = 0
        self._inference_ms_sum = 0.0

    def process_frame(
        self, frame: np.ndarray, frame_index: int | None = None, timestamp_s: float = 0.0
    ) -> FrameResult:
        """Run detection, tracking and counting on the next frame of the stream.

        Args:
            frame: BGR image.
            frame_index: Index of the frame in the source (defaults to a running count;
                differs from it when frames are skipped).
            timestamp_s: Position of the frame in the source, in seconds.
        """
        index = self._frames if frame_index is None else frame_index
        image = ensure_bgr(frame)
        start = time.perf_counter()
        detections = self._detector.detect(image)
        inference_ms = (time.perf_counter() - start) * 1000

        tracked = self._tracker.update(detections)
        new_ids = self._counter.update(index, tracked)

        self._frames += 1
        self._visible_sum += len(detections)
        self._max_visible = max(self._max_visible, len(detections))
        self._inference_ms_sum += inference_ms
        return FrameResult(index, timestamp_s, tracked, self._counter.total, new_ids, inference_ms)

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        summary = [
            f"Unique fruits counted: {result.unique_count}",
            f"Visible now: {result.visible_count}",
            f"Frame: {result.frame_index}",
        ]
        return self._annotator.draw(
            ensure_bgr(frame),
            [item.detection for item in result.tracked],
            track_ids=[item.track_id for item in result.tracked],
            summary_lines=summary,
        )

    @property
    def unique_count(self) -> int:
        return self._counter.total

    @property
    def frames_processed(self) -> int:
        return self._frames

    def summary(self) -> VideoSummary:
        frames = self._frames
        return VideoSummary(
            frames_processed=frames,
            unique_fruit_count=self._counter.total,
            counts_by_class=self._counter.counts_by_class,
            max_visible=self._max_visible,
            mean_visible=self._visible_sum / frames if frames else 0.0,
            detections_summed_over_frames=self._visible_sum,
            mean_inference_ms=self._inference_ms_sum / frames if frames else 0.0,
            objects=[obj.to_dict() for obj in self._counter.objects],
        )
