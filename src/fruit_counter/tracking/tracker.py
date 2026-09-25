"""IoU-based multi-object tracker with motion prediction (ByteTrack-style association).

Algorithm, per frame:

1. **Predict** every live track forward with a constant-velocity model.
2. **Associate high-confidence detections** (``>= high_threshold``) with all tracks,
   using the Hungarian algorithm on IoU between predicted boxes and detections.
3. **Associate low-confidence detections** with the tracks left unmatched. Weak
   detections (partially occluded or blurred fruit) keep existing tracks alive but
   never start new ones, which suppresses false positives.
4. **Start tentative tracks** from unmatched high-confidence detections.
5. **Manage lifecycles**: a tentative track that misses a frame is dropped; one that
   is matched ``min_hits`` consecutive times is *confirmed* and receives a public,
   sequential id. A confirmed track is kept for ``max_age`` frames without matches,
   bridging short occlusions and missed detections.

Only confirmed tracks carry an id, so ids are consecutive and the number of ids
handed out equals the number of unique objects observed.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np

from fruit_counter.config import TrackingConfig
from fruit_counter.structures import BoundingBox, Detection, TrackedDetection, iou_matrix
from fruit_counter.tracking.matching import match_by_iou

# Weight of the newest velocity measurement in the exponential moving average.
VELOCITY_SMOOTHING = 0.5


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REMOVED = "removed"


class Track:
    """State of one tracked object. State vector: ``(cx, cy, w, h)``."""

    def __init__(self, detection: Detection, frame_index: int) -> None:
        self.track_id: int | None = None
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = frame_index
        self.last_frame = frame_index
        self.last_detection = detection
        self._position = np.array(detection.box.to_cxcywh(), dtype=np.float64)
        self._velocity = np.zeros(4, dtype=np.float64)

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    def mark_missed(self) -> None:
        self.time_since_update += 1

    def predicted_box(self) -> BoundingBox:
        """Box extrapolated to the current frame."""
        cx, cy, w, h = self._position + self._velocity * self.time_since_update
        return BoundingBox.from_cxcywh(cx, cy, max(w, 1.0), max(h, 1.0))

    def update(self, detection: Detection, frame_index: int) -> None:
        """Incorporate a matched detection. Must be called after :meth:`mark_missed`."""
        measured = np.array(detection.box.to_cxcywh(), dtype=np.float64)
        frames_elapsed = max(self.time_since_update, 1)
        velocity = (measured - self._position) / frames_elapsed
        if self.hits == 1:
            self._velocity = velocity
        else:
            self._velocity = (
                VELOCITY_SMOOTHING * velocity + (1 - VELOCITY_SMOOTHING) * self._velocity
            )
        self._position = measured
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.last_detection = detection


class FruitTracker:
    """Associates per-frame detections into persistent identities."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self._config = config or TrackingConfig()
        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_index = -1

    @property
    def active_tracks(self) -> list[Track]:
        """Live tracks (tentative and confirmed)."""
        return list(self._tracks)

    @property
    def confirmed_count(self) -> int:
        """Number of ids handed out so far, i.e. unique confirmed objects."""
        return self._next_id - 1

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Process the detections of the next frame.

        Returns one :class:`TrackedDetection` per input detection, in input order.
        """
        self._frame_index += 1
        cfg = self._config
        for track in self._tracks:
            track.mark_missed()

        assigned: dict[int, Track] = {}
        high = [i for i, d in enumerate(detections) if d.confidence >= cfg.high_threshold]
        low = [i for i, d in enumerate(detections) if d.confidence < cfg.high_threshold]

        # Stage 1: high-confidence detections against every live track.
        remaining = self._associate(self._tracks, detections, high, assigned)
        # Stage 2: low-confidence detections against tracks still unmatched.
        self._associate(remaining, detections, low, assigned)

        # New tentative tracks from unmatched high-confidence detections.
        for i in high:
            if i not in assigned:
                track = Track(detections[i], self._frame_index)
                self._tracks.append(track)
                assigned[i] = track

        for track in self._tracks:
            self._update_lifecycle(track)
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

        return [
            TrackedDetection(det, assigned[i].track_id if i in assigned else None)
            for i, det in enumerate(detections)
        ]

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        indices: Sequence[int],
        assigned: dict[int, Track],
    ) -> list[Track]:
        """Match ``detections[indices]`` to ``tracks``; return the unmatched tracks."""
        if not tracks or not indices:
            return list(tracks)
        predicted = np.array([t.predicted_box().to_xyxy() for t in tracks])
        boxes = np.array([detections[i].box.to_xyxy() for i in indices])
        matches, unmatched_tracks, _ = match_by_iou(
            iou_matrix(predicted, boxes), self._config.match_iou
        )
        for track_index, det_index in matches:
            detection_index = indices[det_index]
            tracks[track_index].update(detections[detection_index], self._frame_index)
            assigned[detection_index] = tracks[track_index]
        return [tracks[i] for i in unmatched_tracks]

    def _update_lifecycle(self, track: Track) -> None:
        cfg = self._config
        if track.state is TrackState.TENTATIVE:
            if track.time_since_update > 0:
                track.state = TrackState.REMOVED
            elif track.hits >= cfg.min_hits:
                track.state = TrackState.CONFIRMED
                track.track_id = self._next_id
                self._next_id += 1
        elif track.time_since_update > cfg.max_age:
            track.state = TrackState.REMOVED
