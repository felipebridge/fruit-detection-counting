from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from fruit_counter.detector import Detection

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "data" / "samples" / "fruit_bowl.jpg"
CLASS_NAMES = {0: "apple", 1: "banana", 2: "orange"}


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
    class_name: str = "apple",
) -> Detection:
    class_id = {name: i for i, name in CLASS_NAMES.items()}[class_name]
    return Detection((x1, y1, x2, y2), confidence, class_id, class_name)


class ScriptedDetector:
    """Test double that returns pre-defined detections, one list per call."""

    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self._script = list(script)
        self.calls = 0

    @property
    def class_names(self) -> Mapping[int, str]:
        return CLASS_NAMES

    def detect(self, frame: np.ndarray) -> list[Detection]:
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        result = list(self._script[self.calls]) if self.calls < len(self._script) else []
        self.calls += 1
        return result


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def write_video(
    path: Path, frames: int, fps: float = 10.0, size: tuple[int, int] = (160, 120)
) -> Path:
    """A small MJPG test clip of uniform grey frames."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40, dtype=np.uint8))
    writer.release()
    return path
