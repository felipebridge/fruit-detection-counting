from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import SAMPLE_IMAGE, ScriptedDetector, make_detection
from fruit_counter import cli
from fruit_counter.config import ModelConfig
from fruit_counter.exceptions import ModelError


@pytest.fixture
def fake_detector(monkeypatch: pytest.MonkeyPatch) -> list[ModelConfig]:
    """Replace the YOLO detector; record the model config the CLI built."""
    seen: list[ModelConfig] = []

    def factory(config: ModelConfig) -> ScriptedDetector:
        seen.append(config)
        return ScriptedDetector([[make_detection(10, 10, 100, 100, 0.9, "orange")]])

    monkeypatch.setattr(cli, "build_detector", factory)
    return seen


def run_cli(args: list[str], tmp_path: Path) -> int:
    return cli.main([*args, "--output-dir", str(tmp_path / "out"), "--log-level", "ERROR"])


def test_image_run_prints_human_summary(
    fake_detector: list[ModelConfig], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(["--input", str(SAMPLE_IMAGE), "--run-name", "demo"], tmp_path)
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "Fruits detected: 1" in out
    assert "orange: 1" in out
    assert (tmp_path / "out" / "demo" / "summary.json").is_file()


def test_json_output_is_machine_readable(
    fake_detector: list[ModelConfig], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_cli(["-i", str(SAMPLE_IMAGE), "--json"], tmp_path)
    data = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK
    assert data["kind"] == "image"
    assert data["results"]["fruit_count"] == 1
    assert Path(data["files"]["summary"]).is_file()


def test_flags_override_config_file(fake_detector: list[ModelConfig], tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("model:\n  confidence: 0.6\n  iou: 0.7\n", encoding="utf-8")
    args = ["-i", str(SAMPLE_IMAGE), "-c", str(config_file), "--conf", "0.35"]
    args += ["--weights", "models/custom.pt", "--device", "cpu", "--classes", "apple"]
    assert run_cli(args, tmp_path) == cli.EXIT_OK

    (model,) = fake_detector
    assert model.confidence == 0.35  # CLI beats file
    assert model.iou == 0.7  # file beats default
    assert model.weights == "models/custom.pt"
    assert model.device == "cpu"
    assert model.classes == ("apple",)


def test_bare_classes_flag_keeps_all_classes(
    fake_detector: list[ModelConfig], tmp_path: Path
) -> None:
    assert run_cli(["-i", str(SAMPLE_IMAGE), "--classes"], tmp_path) == cli.EXIT_OK
    assert fake_detector[0].classes == ()


def test_missing_input_argument_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_cli([], tmp_path)
    assert exc_info.value.code == cli.EXIT_USAGE


@pytest.mark.parametrize(
    "args",
    [
        ["-i", "does/not/exist.jpg"],
        ["-i", str(SAMPLE_IMAGE), "--conf", "2"],
        ["-i", str(SAMPLE_IMAGE), "--config", "missing.yaml"],
    ],
)
def test_invalid_input_or_config_exit_with_usage_code(
    fake_detector: list[ModelConfig], tmp_path: Path, args: list[str]
) -> None:
    assert run_cli(args, tmp_path) == cli.EXIT_USAGE
    assert fake_detector == []  # fails before the model is loaded


def test_model_errors_exit_with_failure_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(config: ModelConfig) -> None:
        raise ModelError("Could not load model weights 'x.pt'")

    monkeypatch.setattr(cli, "build_detector", broken)
    assert run_cli(["-i", str(SAMPLE_IMAGE), "--log-level", "ERROR"], tmp_path) == 1
    assert "Could not load" in capsys.readouterr().err


def test_list_classes_ignores_class_filter(
    fake_detector: list[ModelConfig], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(["--list-classes", "--classes", "mango"], tmp_path) == cli.EXIT_OK
    assert fake_detector[0].classes == ()
    assert capsys.readouterr().out.split() == ["0", "apple", "1", "banana", "2", "orange"]


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    assert "fruit-counter 0.1.0" in capsys.readouterr().out
