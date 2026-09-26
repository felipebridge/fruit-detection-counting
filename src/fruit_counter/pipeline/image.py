"""Single-image pipeline: preprocessing -> detection -> counting -> visualisation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fruit_counter.counting import count_by_class
from fruit_counter.detector import Detection, Detector
from fruit_counter.preprocessing import ensure_bgr
from fruit_counter.visualization import Annotator


@dataclass(frozen=True)
class ImageResult:
    """Outcome of processing one image. In a still image, detections == fruit count."""

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
    """Detects and counts fruits in individual images.

    The pipeline is stateless and works on in-memory arrays, so it can be reused by
    the CLI, a batch job or a web service alike.
    """

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
        summary = [f"Fruits detected: {result.fruit_count}"]
        summary += [f"  {name}: {count}" for name, count in result.counts_by_class.items()]
        return self._annotator.draw(ensure_bgr(image), result.detections, summary_lines=summary)
