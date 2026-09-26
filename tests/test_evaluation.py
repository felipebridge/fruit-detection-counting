from pathlib import Path

import pytest

from conftest import SAMPLE_IMAGE, ScriptedDetector, make_detection
from fruit_counter import cli
from fruit_counter.config import load_config
from fruit_counter.evaluation import check_coverage, evaluate_counts, load_ground_truth
from fruit_counter.exceptions import InputError
from fruit_counter.runner import FruitCountingRunner
from fruit_counter.sources import resolve_source


def write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "counts.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_ground_truth(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "file,count,notes\na.jpg,3,x\n b.jpg , 0 ,\n")
    assert load_ground_truth(path) == {"a.jpg": 3, "b.jpg": 0}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name,count\na.jpg,3\n", "header row"),
        ("file,count\na.jpg,-1\n", "line 2"),
        ("file,count\na.jpg,three\n", "line 2"),
        ("file,count\n,3\n", "line 2"),
        ("file,count\n", "no counts"),
    ],
)
def test_invalid_ground_truth_is_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(InputError, match=message):
        load_ground_truth(write_csv(tmp_path, text))


def test_missing_ground_truth_file(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="Cannot read"):
        load_ground_truth(tmp_path / "missing.csv")


def test_evaluate_counts() -> None:
    result = evaluate_counts({"a.jpg": 5, "b.jpg": 2, "c.jpg": 1}, {"a.jpg": 4, "b.jpg": 4})
    assert result["files_evaluated"] == 2
    assert result["files_without_ground_truth"] == ["c.jpg"]
    assert result["mean_absolute_error"] == 1.5
    assert result["mean_error"] == -0.5
    assert (result["total_true"], result["total_predicted"]) == (8, 7)
    assert result["items"][1] == {
        "file": "b.jpg",
        "true_count": 4,
        "predicted_count": 2,
        "error": -2,
    }


def test_coverage_is_checked_before_processing() -> None:
    with pytest.raises(InputError, match="camera"):
        check_coverage(resolve_source("0"), {"a.jpg": 1})
    with pytest.raises(InputError, match="None of the input files"):
        check_coverage(resolve_source(SAMPLE_IMAGE), {"other.jpg": 1})
    check_coverage(resolve_source(SAMPLE_IMAGE), {"fruit_bowl.jpg": 9})


def test_runner_adds_evaluation_to_results(tmp_path: Path) -> None:
    config = load_config(overrides={"output": {"directory": str(tmp_path / "out")}})
    detector = ScriptedDetector([[make_detection(0, 0, 10, 10)] * 2])
    report = FruitCountingRunner(config, detector).run(
        resolve_source(SAMPLE_IMAGE), ground_truth={"fruit_bowl.jpg": 3}
    )
    evaluation = report.results["evaluation"]
    assert evaluation["mean_error"] == -1
    assert evaluation["items"][0]["predicted_count"] == 2


def test_cli_prints_counting_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "YoloDetector", lambda config: ScriptedDetector([[]]))
    truth = write_csv(tmp_path, "file,count\nfruit_bowl.jpg,2\n")
    args = ["-i", str(SAMPLE_IMAGE), "--ground-truth", str(truth), "-o", str(tmp_path), "-q"]
    assert cli.main(args) == cli.EXIT_OK
    assert "Counting error over 1 file(s): MAE 2.0, mean error -2.0" in capsys.readouterr().out


def test_cli_rejects_uncovered_input_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = []
    monkeypatch.setattr(cli, "YoloDetector", lambda config: loaded.append(config))
    truth = write_csv(tmp_path, "file,count\nother.jpg,2\n")
    args = ["-i", str(SAMPLE_IMAGE), "--ground-truth", str(truth), "-o", str(tmp_path), "-q"]
    assert cli.main(args) == cli.EXIT_USAGE
    assert loaded == []
