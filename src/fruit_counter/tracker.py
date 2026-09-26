"""IoU tracker with constant-velocity motion prediction (ByteTrack-style association).

Per frame, every track is moved forward by its velocity, then:

1. Detections with confidence >= ``high_threshold`` are matched to all tracks
   (Hungarian algorithm on IoU between predicted boxes and detections).
2. Weaker detections are matched to the tracks left over. They keep partially
   occluded fruit tracked but never start a track, which suppresses false positives.
3. Unmatched confident detections start tentative tracks.

A tentative track is dropped on its first miss and confirmed after ``min_hits``
consecutive matches, at which point it receives the next sequential id. Only
confirmed tracks have ids, so the number of ids handed out is the unique count.
A confirmed track survives ``max_age`` frames without matches to bridge occlusions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from fruit_counter.config import TrackingConfig
from fruit_counter.detector import Detection

VELOCITY_SMOOTHING = 0.5  # weight of the newest velocity measurement


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    detection: Detection
    track_id: int | None  # None while the track is tentative


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` arrays of ``xyxy`` boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def match_by_iou(iou: np.ndarray, min_iou: float) -> tuple[list[tuple[int, int]], list[int]]:
    """Optimal one-to-one assignment; returns ``(row, col)`` matches and unmatched rows."""
    rows, cols = linear_sum_assignment(iou, maximize=True)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]
    matched_rows = {r for r, _ in matches}
    return matches, [r for r in range(iou.shape[0]) if r not in matched_rows]


def _xyxy_to_cxcywh(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


class Track:
    def __init__(self, detection: Detection) -> None:
        self.track_id: int | None = None
        self.hits = 1
        self.misses = 0
        self._state = _xyxy_to_cxcywh(detection.box)
        self._velocity = np.zeros(4)

    def predicted_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self._state + self._velocity * self.misses
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, detection: Detection) -> None:
        measured = _xyxy_to_cxcywh(detection.box)
        velocity = (measured - self._state) / max(self.misses, 1)
        if self.hits > 1:
            velocity = VELOCITY_SMOOTHING * velocity + (1 - VELOCITY_SMOOTHING) * self._velocity
        self._velocity = velocity
        self._state = measured
        self.hits += 1
        self.misses = 0


class FruitTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self._config = config or TrackingConfig()
        self.tracks: list[Track] = []
        self.confirmed_count = 0

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Process one frame; returns the detections, in order, with their track ids."""
        cfg = self._config
        for track in self.tracks:
            track.misses += 1

        assigned: dict[int, Track] = {}
        high = [i for i, d in enumerate(detections) if d.confidence >= cfg.high_threshold]
        low = [i for i, d in enumerate(detections) if d.confidence < cfg.high_threshold]
        remaining = self._associate(self.tracks, detections, high, assigned)
        self._associate(remaining, detections, low, assigned)

        for i in high:
            if i not in assigned:
                assigned[i] = Track(detections[i])
                self.tracks.append(assigned[i])

        alive = []
        for track in self.tracks:
            if track.track_id is None:
                if track.misses:
                    continue
                if track.hits >= cfg.min_hits:
                    self.confirmed_count += 1
                    track.track_id = self.confirmed_count
            elif track.misses > cfg.max_age:
                continue
            alive.append(track)
        self.tracks = alive

        return [
            TrackedDetection(det, assigned[i].track_id if i in assigned else None)
            for i, det in enumerate(detections)
        ]

    def _associate(
        self,
        tracks: list[Track],
        detections: Sequence[Detection],
        indices: list[int],
        assigned: dict[int, Track],
    ) -> list[Track]:
        """Match ``detections[indices]`` to ``tracks``; return the unmatched tracks."""
        if not tracks or not indices:
            return tracks
        predicted = np.array([t.predicted_xyxy() for t in tracks])
        boxes = np.array([detections[i].box for i in indices])
        matches, unmatched = match_by_iou(iou_matrix(predicted, boxes), self._config.match_iou)
        for t, d in matches:
            tracks[t].update(detections[indices[d]])
            assigned[indices[d]] = tracks[t]
        return [tracks[t] for t in unmatched]
