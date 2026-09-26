"""Fruit detection with Ultralytics YOLO."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import numpy as np

from fruit_counter.config import ModelConfig
from fruit_counter.exceptions import ConfigError, ModelError
from fruit_counter.structures import BoundingBox, Detection

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, config: ModelConfig, model: Any | None = None) -> None:
        self._model = model if model is not None else load_yolo_model(config.weights)
        self.class_names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        class_ids = resolve_class_ids(self.class_names, config.classes)
        device = resolve_device(config.device)
        self._predict_args: dict[str, Any] = {
            "conf": config.confidence,
            "iou": config.iou,
            "imgsz": config.image_size,
            "classes": class_ids,
            # Class-agnostic NMS: otherwise one ambiguous fruit can yield both an
            # "apple" and an "orange" box and be counted twice.
            "agnostic_nms": True,
            "device": device,
            "verbose": False,
        }
        if config.half:
            if device.startswith("cuda"):
                self._predict_args["half"] = True
            else:
                logger.warning("Half precision requires CUDA; using full precision on %s", device)
        logger.info(
            "Detector ready: weights=%s device=%s classes=%s",
            config.weights,
            device,
            [self.class_names[i] for i in class_ids] if class_ids else "all",
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        try:
            results = self._model.predict(frame, **self._predict_args)
        except Exception as exc:
            raise ModelError(f"Inference failed: {exc}") from exc
        return boxes_to_detections(results[0].boxes, self.class_names) if results else []


def load_yolo_model(weights: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ModelError("The 'ultralytics' package is required: pip install ultralytics") from exc
    try:
        return YOLO(weights)
    except Exception as exc:
        raise ModelError(f"Could not load model weights '{weights}': {exc}") from exc


def resolve_class_ids(class_names: Mapping[int, str], wanted: Iterable[str]) -> list[int] | None:
    """Model class ids for the wanted class names; ``None`` (keep all) if none are given."""
    wanted = [name.strip().lower() for name in wanted]
    if not wanted:
        return None
    by_name = {name.lower(): class_id for class_id, name in class_names.items()}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ConfigError(
            f"Class(es) {missing} are not predicted by this model. Available classes: {available}"
        )
    return sorted({by_name[name] for name in wanted})


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def boxes_to_detections(boxes: Any, class_names: Mapping[int, str]) -> list[Detection]:
    """Convert an Ultralytics ``Boxes`` object into detections."""
    if boxes is None or len(boxes) == 0:
        return []
    boxes = boxes.cpu().numpy()
    return [
        Detection(BoundingBox(*xyxy), conf, int(cls), class_names.get(int(cls), str(int(cls))))
        for xyxy, conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
        )
    ]
