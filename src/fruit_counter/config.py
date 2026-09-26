"""Configuration: code defaults, overridden by an optional YAML file, then CLI flags."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from fruit_counter.exceptions import ConfigError

# COCO classes that are fruits. The default YOLO checkpoints are trained on COCO.
COCO_FRUIT_CLASSES: tuple[str, ...] = ("apple", "banana", "orange")


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be within [0, 1], got {value}")


def _check_positive(name: str, value: int) -> None:
    if value < 1:
        raise ConfigError(f"{name} must be >= 1, got {value}")


@dataclass(frozen=True)
class ModelConfig:
    """Detector settings."""

    weights: str = "models/yolo11n.pt"
    device: str = "auto"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    # Class names to keep. An empty tuple keeps every class the model predicts,
    # which is the right choice for a custom, fruit-only model.
    classes: tuple[str, ...] = COCO_FRUIT_CLASSES
    # Suppress overlapping boxes across classes. Essential for counting: with per-class
    # NMS an ambiguous fruit can yield both an "apple" and an "orange" box, which
    # would be counted as two objects.
    agnostic_nms: bool = True
    half: bool = False

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigError("model.weights must not be empty")
        if isinstance(self.classes, str):
            raise ConfigError("model.classes must be a list of class names")
        object.__setattr__(self, "classes", tuple(self.classes))
        _check_unit_interval("model.confidence", self.confidence)
        _check_unit_interval("model.iou", self.iou)
        if self.image_size < 32 or self.image_size % 32:
            raise ConfigError(
                f"model.image_size must be a multiple of 32 and >= 32, got {self.image_size}"
            )


@dataclass(frozen=True)
class TrackingConfig:
    """Multi-object tracker settings (used for video and camera sources)."""

    # Detections at or above this confidence can start new tracks; weaker ones can
    # only extend existing tracks (ByteTrack-style two-stage association).
    high_threshold: float = 0.5
    # Minimum IoU between a predicted track box and a detection to associate them.
    match_iou: float = 0.3
    # Consecutive matched frames before a track is confirmed (and counted).
    min_hits: int = 3
    # Frames a confirmed track survives without a matching detection.
    max_age: int = 30

    def __post_init__(self) -> None:
        _check_unit_interval("tracking.high_threshold", self.high_threshold)
        _check_unit_interval("tracking.match_iou", self.match_iou)
        _check_positive("tracking.min_hits", self.min_hits)
        _check_positive("tracking.max_age", self.max_age)


@dataclass(frozen=True)
class VideoConfig:
    """Video / stream processing settings."""

    # Process every N-th frame. Values > 1 trade accuracy for speed.
    frame_stride: int = 1
    # Stop after this many processed frames (None = whole video).
    max_frames: int | None = None
    # Log progress every N processed frames.
    log_every: int = 50

    def __post_init__(self) -> None:
        _check_positive("video.frame_stride", self.frame_stride)
        _check_positive("video.log_every", self.log_every)
        if self.max_frames is not None:
            _check_positive("video.max_frames", self.max_frames)


@dataclass(frozen=True)
class VisualizationConfig:
    """Annotation rendering settings."""

    show_confidence: bool = True
    show_track_ids: bool = True
    show_summary: bool = True
    line_thickness: int = 2
    font_scale: float = 0.6

    def __post_init__(self) -> None:
        _check_positive("visualization.line_thickness", self.line_thickness)
        if self.font_scale <= 0:
            raise ConfigError(f"visualization.font_scale must be > 0, got {self.font_scale}")


@dataclass(frozen=True)
class OutputConfig:
    """Where and what to write."""

    directory: str = "outputs"
    save_annotated: bool = True
    save_frame_stats: bool = True
    video_codec: str = "mp4v"

    def __post_init__(self) -> None:
        if not self.directory:
            raise ConfigError("output.directory must not be empty")
        if len(self.video_codec) != 4:
            raise ConfigError(
                f"output.video_codec must be a 4-character FourCC, got {self.video_codec!r}"
            )


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["model"]["classes"] = list(self.model.classes)
        return data


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "tracking": TrackingConfig,
    "video": VideoConfig,
    "visualization": VisualizationConfig,
    "output": OutputConfig,
}
_T = TypeVar("_T")


def load_config(
    path: str | Path | None = None, overrides: Mapping[str, Mapping[str, Any]] | None = None
) -> AppConfig:
    """Build the configuration from defaults, an optional YAML file and overrides.

    ``overrides`` is a nested mapping such as ``{"model": {"confidence": 0.4}}``;
    ``None`` values are ignored so unset CLI flags keep the file or default value.
    """
    data = _read_yaml(Path(path)) if path is not None else {}
    for section, values in (overrides or {}).items():
        given = {key: value for key, value in values.items() if value is not None}
        if given:
            current = data.get(section)
            data[section] = {**current, **given} if isinstance(current, Mapping) else given

    unknown = sorted(set(data) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(unknown)}")
    return AppConfig(
        **{name: _build(cls, name, data.get(name, {})) for name, cls in _SECTIONS.items()}
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return content


def _build(cls: type[_T], name: str, values: Any) -> _T:
    if not isinstance(values, Mapping):
        raise ConfigError(f"Config section '{name}' must be a mapping")
    unknown = sorted(set(values) - {f.name for f in dataclasses.fields(cls)})  # type: ignore[arg-type]
    if unknown:
        raise ConfigError(f"Unknown config key(s) in '{name}': {', '.join(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"Invalid value in '{name}': {exc}") from None
