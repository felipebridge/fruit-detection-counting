from pathlib import Path

import pytest

from fruit_counter.config import COCO_FRUIT_CLASSES, AppConfig, load_config
from fruit_counter.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_valid_and_target_coco_fruits() -> None:
    config = load_config(env={})
    assert config == AppConfig()
    assert config.model.classes == COCO_FRUIT_CLASSES
    assert 0 < config.model.confidence < config.tracking.high_threshold


def test_shipped_default_yaml_matches_code_defaults() -> None:
    assert load_config(REPO_ROOT / "configs" / "default.yaml", env={}) == AppConfig()


def test_yaml_values_override_defaults(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        "model:\n  confidence: 0.4\n  classes: [apple]\ntracking:\n  max_age: 5\n",
    )
    config = load_config(path, env={})
    assert config.model.confidence == 0.4
    assert config.model.classes == ("apple",)
    assert config.tracking.max_age == 5
    # Untouched values keep their defaults.
    assert config.model.iou == AppConfig().model.iou


def test_empty_yaml_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(write_yaml(tmp_path, ""), env={}) == AppConfig()


def test_env_vars_are_coerced_to_field_types() -> None:
    env = {
        "FRUIT_COUNTER_MODEL__CONFIDENCE": "0.35",
        "FRUIT_COUNTER_MODEL__IMAGE_SIZE": "1280",
        "FRUIT_COUNTER_MODEL__HALF": "yes",
        "FRUIT_COUNTER_MODEL__CLASSES": "apple, orange",
        "FRUIT_COUNTER_VIDEO__MAX_FRAMES": "100",
        "UNRELATED_VARIABLE": "ignored",
    }
    config = load_config(env=env)
    assert config.model.confidence == 0.35
    assert config.model.image_size == 1280
    assert config.model.half is True
    assert config.model.classes == ("apple", "orange")
    assert config.video.max_frames == 100


def test_empty_class_list_from_env_means_all_classes() -> None:
    config = load_config(env={"FRUIT_COUNTER_MODEL__CLASSES": ""})
    assert config.model.classes == ()


def test_precedence_is_yaml_then_env_then_overrides(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "model:\n  confidence: 0.3\n  iou: 0.6\n  device: cpu\n")
    env = {"FRUIT_COUNTER_MODEL__CONFIDENCE": "0.4", "FRUIT_COUNTER_MODEL__IOU": "0.5"}
    config = load_config(path, overrides={"model": {"confidence": 0.5}}, env=env)
    assert config.model.confidence == 0.5  # override wins
    assert config.model.iou == 0.5  # env beats YAML
    assert config.model.device == "cpu"  # YAML beats default


def test_none_overrides_are_ignored() -> None:
    config = load_config(overrides={"model": {"confidence": None, "device": "cpu"}}, env={})
    assert config.model.confidence == AppConfig().model.confidence
    assert config.model.device == "cpu"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": {"confidence": 1.5}}, "model.confidence"),
        ({"model": {"iou": -0.1}}, "model.iou"),
        ({"model": {"image_size": 500}}, "multiple of 32"),
        ({"model": {"weights": ""}}, "model.weights"),
        ({"tracking": {"min_hits": 0}}, "tracking.min_hits"),
        ({"tracking": {"max_age": 0}}, "tracking.max_age"),
        ({"video": {"frame_stride": 0}}, "video.frame_stride"),
        ({"output": {"video_codec": "h264x"}}, "FourCC"),
        ({"model": {"unknown_key": 1}}, "Unknown config key"),
        ({"not_a_section": {}}, "Unknown config key"),
        ({"model": {"image_size": "large"}}, "expected int"),
        ({"model": {"image_size": 640.5}}, "expected int"),
        ({"model": {"half": "maybe"}}, "expected bool"),
        ({"model": "cpu"}, "must be a mapping"),
    ],
)
def test_invalid_values_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides, env={})


def test_malformed_env_var_name_is_rejected() -> None:
    with pytest.raises(ConfigError, match="FRUIT_COUNTER_"):
        load_config(env={"FRUIT_COUNTER_CONFIDENCE": "0.3"})


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml", env={})


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_yaml(tmp_path, "- just\n- a list\n"), env={})


def test_to_dict_round_trips() -> None:
    config = load_config(overrides={"model": {"classes": ["apple"]}}, env={})
    assert load_config(overrides=config.to_dict(), env={}) == config
