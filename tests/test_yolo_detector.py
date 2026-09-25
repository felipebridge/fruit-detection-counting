"""Tests for the YOLO adapter.

Most tests use a fake model with the same interface as ``ultralytics.YOLO`` so they
run without weights. ``test_real_model_*`` runs actual inference and is marked slow.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from conftest import REPO_ROOT, SAMPLE_IMAGE
from fruit_counter.config import ModelConfig
from fruit_counter.detection import Detector, YoloDetector
from fruit_counter.detection.yolo import boxes_to_detections, resolve_class_ids
from fruit_counter.exceptions import ConfigError, ModelError

COCO_SUBSET = {0: "person", 46: "banana", 47: "apple", 49: "orange"}


class FakeYolo:
    names = COCO_SUBSET

    def __init__(self, boxes: SimpleNamespace | None = None, error: Exception | None = None):
        self.boxes = boxes
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def predict(self, frame: np.ndarray, **kwargs: Any) -> list[SimpleNamespace]:
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=self.boxes)]


class Boxes(SimpleNamespace):
    """Mimics ``ultralytics.engine.results.Boxes`` (numpy instead of torch tensors)."""

    def __len__(self) -> int:
        return len(self.conf)


def make_boxes(rows: list[tuple[float, float, float, float, float, int]]) -> Boxes:
    arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
    return Boxes(xyxy=arr[:, :4], conf=arr[:, 4], cls=arr[:, 5])


def test_detector_satisfies_protocol() -> None:
    detector = YoloDetector(ModelConfig(device="cpu"), model=FakeYolo(make_boxes([])))
    assert isinstance(detector, Detector)
    assert detector.class_names[47] == "apple"


def test_inference_settings_come_from_config() -> None:
    model = FakeYolo(make_boxes([]))
    config = ModelConfig(device="cpu", confidence=0.4, iou=0.6, image_size=320)
    YoloDetector(config, model=model).detect(np.zeros((64, 64, 3), np.uint8))
    assert model.kwargs["conf"] == 0.4
    assert model.kwargs["iou"] == 0.6
    assert model.kwargs["imgsz"] == 320
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["classes"] == [46, 47, 49]  # fruit classes only, no "person"
    assert "half" not in model.kwargs  # half precision is only used on CUDA


def test_predictions_are_converted_to_detections() -> None:
    model = FakeYolo(make_boxes([(10, 20, 50, 80, 0.9, 47), (0, 0, 5, 5, 0.3, 49)]))
    detections = YoloDetector(ModelConfig(device="cpu"), model=model).detect(
        np.zeros((100, 100, 3), np.uint8)
    )
    assert [d.class_name for d in detections] == ["apple", "orange"]
    assert detections[0].box.to_xyxy() == (10, 20, 50, 80)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[1].class_id == 49


def test_empty_prediction_returns_no_detections() -> None:
    assert boxes_to_detections(None, COCO_SUBSET) == []
    assert boxes_to_detections(make_boxes([]), COCO_SUBSET) == []


def test_empty_class_filter_keeps_all_classes() -> None:
    model = FakeYolo(make_boxes([]))
    YoloDetector(ModelConfig(device="cpu", classes=()), model=model).detect(
        np.zeros((8, 8, 3), np.uint8)
    )
    assert model.kwargs["classes"] is None


def test_class_names_are_case_insensitive() -> None:
    assert resolve_class_ids(COCO_SUBSET, ["Apple", " ORANGE "]) == [47, 49]


def test_unknown_class_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="mango"):
        YoloDetector(ModelConfig(device="cpu", classes=("mango",)), model=FakeYolo())


def test_inference_errors_are_wrapped() -> None:
    model = FakeYolo(error=RuntimeError("CUDA out of memory"))
    detector = YoloDetector(ModelConfig(device="cpu"), model=model)
    with pytest.raises(ModelError, match="out of memory"):
        detector.detect(np.zeros((8, 8, 3), np.uint8))


def test_missing_weights_raise_model_error(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    with pytest.raises(ModelError, match="Could not load"):
        YoloDetector(ModelConfig(weights=str(tmp_path / "does_not_exist.pt"), device="cpu"))


WEIGHTS = Path(os.environ.get("FRUIT_COUNTER_TEST_WEIGHTS", REPO_ROOT / "models" / "yolo11n.pt"))


@pytest.mark.slow
@pytest.mark.skipif(not WEIGHTS.is_file(), reason=f"model weights not found at {WEIGHTS}")
def test_real_model_detects_fruit_in_sample_image() -> None:
    pytest.importorskip("ultralytics")
    detector = YoloDetector(ModelConfig(weights=str(WEIGHTS), device="cpu"))
    frame = cv2.imread(str(SAMPLE_IMAGE))
    detections = detector.detect(frame)

    assert detections, "expected at least one fruit in the sample image"
    height, width = frame.shape[:2]
    for det in detections:
        assert det.class_name in {"apple", "banana", "orange"}
        assert det.confidence >= 0.25
        assert 0 <= det.box.x1 < det.box.x2 <= width + 1
        assert 0 <= det.box.y1 < det.box.y2 <= height + 1
