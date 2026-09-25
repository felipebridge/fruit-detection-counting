"""Output management: run directories and result writers.

Every invocation writes into its own run directory so results never overwrite each
other and all artefacts of a run stay together::

    outputs/
    └── <source-name>_<YYYYmmdd-HHMMSS>/
        ├── summary.json          # counts, detections/objects, config, metadata
        ├── <name>_annotated.jpg  # image sources
        ├── <name>_annotated.mp4  # video / camera sources
        └── frames.csv            # per-frame statistics (video / camera)
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

import cv2
import numpy as np

from fruit_counter.exceptions import OutputError

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def create_run_dir(root: str | Path, source_name: str, run_name: str | None = None) -> Path:
    """Create and return a fresh run directory under ``root``.

    With ``run_name`` the directory name is fixed (and reused if it exists);
    otherwise it is ``<source_name>_<timestamp>`` with a numeric suffix on collision.
    """
    root = Path(root)
    if run_name:
        path = root / _safe_name(run_name)
    else:
        base = f"{_safe_name(source_name)}_{datetime.now():%Y%m%d-%H%M%S}"
        path = root / base
        suffix = 1
        while path.exists():
            suffix += 1
            path = root / f"{base}-{suffix}"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"Cannot create output directory {path}: {exc}") from exc
    return path


def _safe_name(name: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", name).strip("._")
    return cleaned or "run"


def write_json(path: Path, data: Mapping[str, Any]) -> Path:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, TypeError) as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


def write_image(path: Path, image: np.ndarray) -> Path:
    """Write an image (supports non-ASCII paths, unlike ``cv2.imwrite`` on Windows)."""
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise OutputError(f"Cannot encode image for {path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise OutputError(f"Cannot write {path}: {exc}") from exc
    return path


class CsvStreamWriter:
    """Appends dict rows to a CSV file as they are produced (context manager).

    Rows are flushed incrementally, so long videos and camera streams do not keep
    per-frame statistics in memory, and partial results survive interruptions.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows_written = 0
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def write(self, row: Mapping[str, Any]) -> None:
        if self._writer is None:
            try:
                self._file = self.path.open("w", newline="", encoding="utf-8")
            except OSError as exc:
                raise OutputError(f"Cannot write {self.path}: {exc}") from exc
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> CsvStreamWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class VideoFileWriter:
    """Writes frames to a video file; opens lazily on the first frame (context manager)."""

    def __init__(self, path: Path, fps: float, codec: str = "mp4v") -> None:
        self.path = path
        self.frames_written = 0
        self._fps = fps
        self._codec = codec
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter.fourcc(*self._codec)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self._fps, (width, height))
            if not self._writer.isOpened():
                raise OutputError(
                    f"Cannot open video writer for {self.path} with codec '{self._codec}'"
                )
            self._size = (width, height)
        elif (width, height) != self._size:
            raise OutputError(f"Frame size {(width, height)} differs from video size {self._size}")
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoFileWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
