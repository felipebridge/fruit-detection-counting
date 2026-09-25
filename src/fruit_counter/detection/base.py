"""Detector interface.

The rest of the pipeline depends only on this protocol, never on a concrete model
library. Any object with a ``class_names`` mapping and a ``detect`` method can be
plugged in: a different YOLO version, an ONNX/TensorRT runtime, or a test double.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from fruit_counter.structures import Detection


@runtime_checkable
class Detector(Protocol):
    """Produces fruit detections for a single BGR frame."""

    @property
    def class_names(self) -> Mapping[int, str]:
        """Mapping from class id to human-readable class name."""
        ...

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a ``HxWx3`` BGR ``uint8`` frame."""
        ...
