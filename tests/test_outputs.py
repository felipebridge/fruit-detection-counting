import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fruit_counter.exceptions import OutputError
from fruit_counter.outputs import (
    CsvStreamWriter,
    VideoFileWriter,
    create_run_dir,
    write_image,
    write_json,
)


def test_run_dirs_are_unique_per_invocation(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path / "outputs", "fruit bowl")
    second = create_run_dir(tmp_path / "outputs", "fruit bowl")
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.name.startswith("fruit_bowl_")
    assert second.name == f"{first.name}-2"


def test_explicit_run_name_is_sanitised_and_reused(tmp_path: Path) -> None:
    path = create_run_dir(tmp_path, "ignored", run_name="../demo run")
    assert path == tmp_path / "demo_run"
    assert create_run_dir(tmp_path, "ignored", run_name="demo run") == path


def test_run_dir_creation_failure_is_reported(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(OutputError, match="Cannot create"):
        create_run_dir(blocker, "x", run_name="run")


def test_write_json_round_trips_unicode(tmp_path: Path) -> None:
    path = write_json(tmp_path / "r.json", {"fruit": "manzana ñ", "count": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"fruit": "manzana ñ", "count": 3}


def test_write_json_rejects_unserialisable_data(tmp_path: Path) -> None:
    with pytest.raises(OutputError):
        write_json(tmp_path / "r.json", {"bad": object()})


def test_write_image_supports_non_ascii_paths(tmp_path: Path) -> None:
    image = np.full((20, 30, 3), 128, dtype=np.uint8)
    path = write_image(tmp_path / "resultado_ñ.png", image)
    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, image)


def test_csv_stream_writer_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "frames.csv"
    with CsvStreamWriter(path) as writer:
        writer.write({"frame": 0, "count": 1})
        writer.write({"frame": 1, "count": 2})
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"frame": "0", "count": "1"}, {"frame": "1", "count": "2"}]
    assert writer.rows_written == 2


def test_csv_stream_writer_creates_no_file_without_rows(tmp_path: Path) -> None:
    with CsvStreamWriter(tmp_path / "frames.csv"):
        pass
    assert not (tmp_path / "frames.csv").exists()


def test_video_writer_produces_readable_video(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    with VideoFileWriter(path, fps=15, codec="mp4v") as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    assert writer.frames_written == 10

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(15)
    capture.release()


def test_video_writer_rejects_size_changes(tmp_path: Path) -> None:
    with VideoFileWriter(tmp_path / "out.mp4", fps=10) as writer:
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        with pytest.raises(OutputError, match="differs"):
            writer.write(np.zeros((40, 64, 3), dtype=np.uint8))
