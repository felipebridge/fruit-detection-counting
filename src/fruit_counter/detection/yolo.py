"""Ultralytics YOLO detector adapter."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from fruit_counter.config import ModelConfig
from fruit_counter.exceptions import ConfigError, ModelError
from fruit_counter.structures import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """Wraps an Ultralytics YOLO model behind the :class:`Detector` protocol.

    Class filtering, confidence and NMS thresholds are applied by Ultralytics during
    inference, so the pipeline only ever sees the configured fruit classes.

    Args:
        config: Model configuration.
        model: Optional pre-built model object exposing ``names`` and ``predict``
            (mainly for testing). Loaded from ``config.weights`` when omitted.
    """

    def __init__(self, config: ModelConfig, model: Any | None = None) -> None:
        self._config = config
        self._model = model if model is not None else load_yolo_model(config.weights)
        self._class_names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        self._class_ids = resolve_class_ids(self._class_names, config.classes)
        self._device = resolve_device(config.device)
        self._half = config.half and self._device.startswith("cuda")
        if config.half and not self._half:
            logger.warning(
                "Half precision requires CUDA; running in full precision on %s", self._device
            )
        logger.info(
            "Detector ready: weights=%s device=%s classes=%s",
            config.weights,
            self._device,
            [self._class_names[i] for i in self._class_ids] if self._class_ids else "all",
        )

    @property
    def class_names(self) -> Mapping[int, str]:
        return dict(self._class_names)

    @property
    def device(self) -> str:
        return self._device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        options: dict[str, Any] = {
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.image_size,
            "classes": self._class_ids,
            # Class-agnostic NMS: otherwise one ambiguous fruit can yield both an
            # "apple" and an "orange" box and be counted twice.
            "agnostic_nms": True,
            "device": self._device,
            "verbose": False,
        }
        if self._half:
            # Only passed when enabled: recent Ultralytics versions warn on any use.
            options["half"] = True
        try:
            results = self._model.predict(frame, **options)
        except Exception as exc:  # the underlying library raises many exception types
            raise ModelError(f"Inference failed: {exc}") from exc
        if not results:
            return []
        return boxes_to_detections(results[0].boxes, self._class_names)


def load_yolo_model(weights: str) -> Any:
    """Load a YOLO model, downloading official checkpoints if needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModelError(
            "The 'ultralytics' package is required for YOLO models: pip install ultralytics"
        ) from exc
    try:
        return YOLO(weights)
    except Exception as exc:
        raise ModelError(f"Could not load model weights '{weights}': {exc}") from exc


def resolve_class_ids(class_names: Mapping[int, str], wanted: Iterable[str]) -> list[int] | None:
    """Map configured class names to model class ids.

    Returns ``None`` (keep every class) when ``wanted`` is empty.

    Raises:
        ConfigError: If a requested class does not exist in the model.
    """
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
    """Resolve ``"auto"`` to the best available torch device."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with ultralytics
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def boxes_to_detections(boxes: Any, class_names: Mapping[int, str]) -> list[Detection]:
    """Convert an Ultralytics ``Boxes`` object into :class:`Detection` instances."""
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    class_ids = _to_numpy(boxes.cls).reshape(-1).astype(int)

    detections = []
    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confidences, class_ids, strict=True):
        detections.append(
            Detection(
                box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_names.get(int(class_id), str(class_id)),
            )
        )
    return detections


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)
