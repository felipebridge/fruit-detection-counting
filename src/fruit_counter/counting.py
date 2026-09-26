"""Counting logic.

Two different quantities are computed in this project and must not be confused:

* **Detection count**: how many fruits are visible in a single frame/image. This is
  what :func:`count_by_class` computes.
* **Unique count**: how many *distinct* fruits have been observed across a video.
  It requires identity association (tracking) and is computed by
  :class:`UniqueCounter`: a fruit seen in 100 consecutive frames counts once.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fruit_counter.detector import Detection
from fruit_counter.tracker import TrackedDetection


def count_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    """Number of detections per class name, sorted by class name."""
    counts = Counter(det.class_name for det in detections)
    return dict(sorted(counts.items()))


@dataclass
class CountedObject:
    """Aggregated observations of one unique (tracked) fruit."""

    track_id: int
    first_frame: int
    last_frame: int
    observations: int = 0
    confidence_sum: float = 0.0
    class_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def class_name(self) -> str:
        """Class with the highest accumulated confidence over the object's lifetime.

        Voting smooths out frames where the detector confuses similar classes.
        """
        return max(sorted(self.class_scores), key=lambda name: self.class_scores[name])

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.observations if self.observations else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "observations": self.observations,
            "mean_confidence": round(self.mean_confidence, 4),
        }


class UniqueCounter:
    """Counts distinct objects from tracker output.

    Only detections with a ``track_id`` (i.e. confirmed tracks) contribute. Each id
    is counted exactly once no matter how many frames it appears in.
    """

    def __init__(self) -> None:
        self._objects: dict[int, CountedObject] = {}

    def update(self, frame_index: int, tracked: Iterable[TrackedDetection]) -> list[int]:
        """Record one frame of tracker output.

        Returns:
            The track ids counted for the first time in this frame.
        """
        new_ids = []
        for item in tracked:
            if item.track_id is None:
                continue
            obj = self._objects.get(item.track_id)
            if obj is None:
                obj = CountedObject(item.track_id, frame_index, frame_index)
                self._objects[item.track_id] = obj
                new_ids.append(item.track_id)
            obj.last_frame = frame_index
            obj.observations += 1
            obj.confidence_sum += item.detection.confidence
            obj.class_scores[item.detection.class_name] += item.detection.confidence
        return new_ids

    @property
    def total(self) -> int:
        return len(self._objects)

    @property
    def counts_by_class(self) -> dict[str, int]:
        counts = Counter(obj.class_name for obj in self._objects.values())
        return dict(sorted(counts.items()))

    @property
    def objects(self) -> list[CountedObject]:
        return sorted(self._objects.values(), key=lambda obj: obj.track_id)
