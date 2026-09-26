"""Counting: fruits visible in one image, and distinct fruits across a video.

A fruit that stays in view for 100 frames is one fruit, so video counts come from
confirmed tracker ids, not from summing per-frame detections.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fruit_counter.detector import Detection
from fruit_counter.tracker import TrackedDetection


def count_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    counts = Counter(det.class_name for det in detections)
    return dict(sorted(counts.items()))


@dataclass
class CountedObject:
    track_id: int
    first_frame: int
    last_frame: int
    observations: int = 0
    confidence_sum: float = 0.0
    class_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def class_name(self) -> str:
        # Confidence-weighted vote over the track's lifetime, which smooths out
        # frames where the detector confuses similar classes (apple vs. orange).
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
    """Counts each confirmed track id once; unconfirmed detections are ignored."""

    def __init__(self) -> None:
        self._objects: dict[int, CountedObject] = {}

    def update(self, frame_index: int, tracked: Iterable[TrackedDetection]) -> list[int]:
        """Record one frame of tracker output; returns the ids counted for the first time."""
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

    def class_of(self, track_id: int) -> str | None:
        """The voted class of a counted track, or ``None`` if it was never counted."""
        obj = self._objects.get(track_id)
        return obj.class_name if obj is not None else None

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
