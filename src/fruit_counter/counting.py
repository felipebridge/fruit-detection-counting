"""Counting logic.

Two different quantities are computed in this project and must not be confused:

* **Detection count**: how many fruits are visible in a single frame/image. This is
  what :func:`count_by_class` computes.
* **Unique count**: how many *distinct* fruits have been observed across a video.
  It requires identity association (tracking) and is computed by
  :class:`UniqueCounter`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from fruit_counter.structures import Detection


def count_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    """Number of detections per class name, sorted by class name."""
    counts = Counter(det.class_name for det in detections)
    return dict(sorted(counts.items()))
