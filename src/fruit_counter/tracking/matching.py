"""Detection-to-track assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

Matches = list[tuple[int, int]]


def match_by_iou(iou: np.ndarray, min_iou: float) -> tuple[Matches, list[int], list[int]]:
    """Globally optimal one-to-one assignment on an IoU matrix (Hungarian algorithm).

    Args:
        iou: Matrix of shape ``(num_tracks, num_detections)``.
        min_iou: Pairs with a lower IoU are never matched.

    Returns:
        ``(matches, unmatched_track_indices, unmatched_detection_indices)`` where
        ``matches`` holds ``(track_index, detection_index)`` pairs.
    """
    num_tracks, num_detections = iou.shape
    if num_tracks == 0 or num_detections == 0:
        return [], list(range(num_tracks)), list(range(num_detections))

    rows, cols = linear_sum_assignment(-iou)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]
    matched_tracks = {r for r, _ in matches}
    matched_dets = {c for _, c in matches}
    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(num_detections) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets
