"""Core, framework-independent data structures shared across the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates (``x1, y1`` top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coords = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(c) for c in coords):
            raise ValueError(f"Bounding box coordinates must be finite: {coords}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Bounding box has negative size: {coords}")

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        w, h = max(w, 0.0), max(h, 0.0)
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        return float(iou_matrix(np.array([self.to_xyxy()]), np.array([other.to_xyxy()]))[0, 0])

    def clip(self, width: int, height: int) -> BoundingBox:
        """Clip the box to an image of the given size."""
        x1 = min(max(self.x1, 0.0), width)
        y1 = min(max(self.y1, 0.0), height)
        x2 = min(max(self.x2, x1), width)
        y2 = min(max(self.y2, y1), height)
        return BoundingBox(x1, y1, x2, y2)


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection in one frame."""

    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box_xyxy": [round(c, 1) for c in self.box.to_xyxy()],
        }


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection together with the identity assigned to it by the tracker.

    ``track_id`` is ``None`` while the underlying track is still tentative (not yet
    confirmed), or when the detection could not be associated with any track.
    """

    detection: Detection
    track_id: int | None


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Args:
        boxes_a: Array of shape ``(N, 4)``.
        boxes_b: Array of shape ``(M, 4)``.

    Returns:
        Array of shape ``(N, M)`` with values in ``[0, 1]``.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou
