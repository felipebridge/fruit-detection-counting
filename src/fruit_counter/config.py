"""Application configuration.

Configuration is a tree of frozen dataclasses. Values are resolved with the following
precedence (lowest to highest):

1. Defaults defined in this module.
2. A YAML file (see ``configs/default.yaml``).
3. Environment variables named ``FRUIT_COUNTER_<SECTION>__<KEY>``,
   e.g. ``FRUIT_COUNTER_MODEL__CONFIDENCE=0.4``.
4. Explicit overrides (typically CLI flags), as a nested mapping.

Every section validates itself on construction, so an invalid value fails fast with a
:class:`~fruit_counter.exceptions.ConfigError` instead of deep inside the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from fruit_counter.exceptions import ConfigError

ENV_PREFIX = "FRUIT_COUNTER_"
ENV_SECTION_SEPARATOR = "__"

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
        """Return a plain, JSON/YAML-serialisable representation."""
        data = dataclasses.asdict(self)
        data["model"]["classes"] = list(self.model.classes)
        return data


# --------------------------------------------------------------------------- loading

_T = TypeVar("_T")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from defaults, a YAML file, env vars and overrides.

    Args:
        path: Optional YAML file.
        overrides: Nested mapping (``{"model": {"confidence": 0.4}}``) applied last.
            ``None`` values are ignored so that unset CLI flags do not override anything.
        env: Environment mapping; defaults to :data:`os.environ`.
    """
    data: dict[str, Any] = {}
    if path is not None:
        _deep_merge(data, _read_yaml(Path(path)))
    _deep_merge(data, _env_overrides(os.environ if env is None else env))
    if overrides:
        _deep_merge(data, _drop_none(overrides))
    return _build(AppConfig, data, "")


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


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = name[len(ENV_PREFIX) :].lower().split(ENV_SECTION_SEPARATOR)
        if len(parts) != 2 or not all(parts):
            raise ConfigError(
                f"Environment variable {name} must look like "
                f"{ENV_PREFIX}<SECTION>{ENV_SECTION_SEPARATOR}<KEY>"
            )
        section, key = parts
        result.setdefault(section, {})[key] = value
    return result


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        elif isinstance(value, Mapping):
            target[key] = dict(value)
        else:
            target[key] = value


def _drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            result[key] = _drop_none(value)
        elif value is not None:
            result[key] = value
    return result


def _build(cls: type[_T], data: Mapping[str, Any], prefix: str) -> _T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"Section '{prefix.rstrip('.') or 'root'}' must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        where = prefix.rstrip(".") or "root"
        raise ConfigError(f"Unknown config key(s) in '{where}': {', '.join(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        qualified = f"{prefix}{name}"
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{qualified}.")  # type: ignore[arg-type]
        else:
            kwargs[name] = _coerce(value, hint, qualified)
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, name: str) -> Any:
    """Convert ``value`` (possibly a string from the environment) to type ``hint``."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin in (typing.Union, types.UnionType):
        is_null = value is None or (
            isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
        )
        if is_null and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], name)

    if origin is tuple:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise ConfigError(f"{name} must be a list, got {value!r}")
        return tuple(_coerce(item, args[0], name) for item in items)

    try:
        if hint is bool:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError(value)
        if hint is int:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError(value)
            return int(value)
        if hint is float:
            if isinstance(value, bool):
                raise ValueError(value)
            return float(value)
        if hint is str:
            if not isinstance(value, (str, int, float)):
                raise ValueError(value)
            return str(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"Invalid value for {name}: expected {hint.__name__}, got {value!r}"
        ) from None
    raise ConfigError(f"Unsupported config type for {name}: {hint}")  # pragma: no cover
