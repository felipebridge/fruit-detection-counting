from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import SAMPLE_IMAGE, ScriptedDetector, make_detection
from fruit_counter.config import AppConfig, load_config
from fruit_counter.exceptions import InputError
from fruit_counter.runner import FruitCountingRunner
from fruit_counter.sources import resolve_source


def make_config(tmp_path: Path, **sections: dict) -> AppConfig:
    overrides = {"output": {"directory": str(tmp_path / "outputs")}, **sections}
    return load_config(overrides=overrides)


def write_video(path: Path, frames: int, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (160, 120))
    for _ in range(frames):
        writer.write(np.full((120, 160, 3), 40, dtype=np.uint8))
    writer.release()
    return path


def two_fruits_script(frames: int) -> list[list]:
    """An apple crossing the whole clip and an orange appearing halfway."""
    script = []
    for f in range(frames):
        dets = [make_detection(2 * f, 10, 2 * f + 30, 40, 0.9, "apple")]
        if f >= frames // 2:
            dets.append(make_detection(2 * f, 70, 2 * f + 30, 100, 0.8, "orange"))
        script.append(dets)
    return script


def test_single_image_run_writes_annotated_image_and_summary(tmp_path: Path) -> None:
    detections = [
        make_detection(100, 100, 300, 300, 0.9, "orange"),
        make_detection(400, 400, 600, 600, 0.7, "apple"),
    ]
    runner = FruitCountingRunner(make_config(tmp_path), ScriptedDetector([detections]))
    report = runner.run(resolve_source(SAMPLE_IMAGE))

    assert report.output_dir.parent == tmp_path / "outputs"
    assert report.output_dir.name.startswith("fruit_bowl_")
    assert report.fruit_count == 2
    assert report.files["annotated"].name == "fruit_bowl_annotated.jpg"
    annotated = cv2.imread(str(report.files["annotated"]))
    assert annotated.shape == cv2.imread(str(SAMPLE_IMAGE)).shape

    summary = json.loads(report.files["summary"].read_text(encoding="utf-8"))
    assert summary["source"]["kind"] == "image"
    assert summary["results"]["fruit_count"] == 2
    assert summary["results"]["counts_by_class"] == {"apple": 1, "orange": 1}
    assert len(summary["results"]["detections"]) == 2
    assert summary["config"]["model"]["classes"] == ["apple", "banana", "orange"]


def test_image_directory_aggregates_and_skips_broken_files(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        (images / name).write_bytes(SAMPLE_IMAGE.read_bytes())
    (images / "c_broken.jpg").write_bytes(b"corrupt")

    script = [[make_detection(0, 0, 50, 50)], [make_detection(0, 0, 50, 50, class_name="banana")]]
    runner = FruitCountingRunner(make_config(tmp_path), ScriptedDetector(script))
    report = runner.run(resolve_source(images))

    assert report.fruit_count == 2
    assert report.results["counts_by_class"] == {"apple": 1, "banana": 1}
    assert report.results["images_processed"] == 2
    assert [s["file"] for s in report.results["images_skipped"]] == ["c_broken.jpg"]
    assert {p.name for p in report.output_dir.iterdir()} == {
        "a_annotated.jpg",
        "b_annotated.jpg",
        "summary.json",
    }


def test_failed_runs_leave_no_empty_output_directory(tmp_path: Path) -> None:
    broken_image = tmp_path / "broken.jpg"
    broken_image.write_bytes(b"corrupt")
    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"corrupt")
    runner = FruitCountingRunner(make_config(tmp_path), ScriptedDetector([]))

    for source in (broken_image, broken_video):
        with pytest.raises(InputError):
            runner.run(resolve_source(source))
    assert list((tmp_path / "outputs").iterdir()) == []


def test_video_without_readable_frames_leaves_no_output(tmp_path: Path) -> None:
    video = write_video(tmp_path / "empty.avi", frames=0)
    runner = FruitCountingRunner(make_config(tmp_path), ScriptedDetector([]))
    if not cv2.VideoCapture(str(video)).isOpened():
        pytest.skip("this OpenCV build cannot open an empty video")
    with pytest.raises(InputError, match="No frames"):
        runner.run(resolve_source(video))
    assert list((tmp_path / "outputs").iterdir()) == []


def test_video_run_counts_unique_fruits_and_writes_all_outputs(tmp_path: Path) -> None:
    video = write_video(tmp_path / "orchard.avi", frames=30, fps=10)
    runner = FruitCountingRunner(make_config(tmp_path), ScriptedDetector(two_fruits_script(30)))
    report = runner.run(resolve_source(video))

    results = report.results
    assert report.fruit_count == 2
    assert results["frames_processed"] == 30
    assert results["counts_by_class"] == {"apple": 1, "orange": 1}
    assert results["detections_summed_over_frames"] == 45  # naive count would say 45
    assert results["source_fps"] == pytest.approx(10)
    assert results["interrupted"] is False

    capture = cv2.VideoCapture(str(report.files["annotated"]))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (
        160,
        120,
    )
    capture.release()

    with report.files["frame_stats"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert rows[-1]["unique_count"] == "2"
    assert rows[-1]["visible_count"] == "2"
    assert json.loads(report.files["summary"].read_text(encoding="utf-8"))["results"] == results


def test_video_stride_and_max_frames_are_applied(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.avi", frames=30, fps=10)
    config = make_config(tmp_path, video={"frame_stride": 2, "max_frames": 5})
    detector = ScriptedDetector([[] for _ in range(30)])
    report = FruitCountingRunner(config, detector).run(resolve_source(video))

    assert detector.calls == 5
    assert report.results["frames_processed"] == 5
    capture = cv2.VideoCapture(str(report.files["annotated"]))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(5)  # source fps / stride
    capture.release()
    with report.files["frame_stats"].open(newline="", encoding="utf-8") as f:
        assert [r["frame_index"] for r in csv.DictReader(f)] == ["0", "2", "4", "6", "8"]


def test_annotated_video_can_be_disabled(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.avi", frames=5)
    config = make_config(tmp_path, output={"save_annotated": False})
    report = FruitCountingRunner(config, ScriptedDetector([])).run(resolve_source(video))
    assert set(report.files) == {"summary", "frame_stats"}
    assert sorted(p.name for p in report.output_dir.iterdir()) == ["frames.csv", "summary.json"]


def test_keyboard_interrupt_keeps_partial_results(tmp_path: Path) -> None:
    class InterruptingDetector(ScriptedDetector):
        def detect(self, frame):
            if self.calls == 4:
                raise KeyboardInterrupt
            return super().detect(frame)

    video = write_video(tmp_path / "clip.avi", frames=20)
    runner = FruitCountingRunner(make_config(tmp_path), InterruptingDetector([]))
    report = runner.run(resolve_source(video))
    assert report.results["interrupted"] is True
    assert report.results["frames_processed"] == 4
    assert report.files["summary"].is_file()
