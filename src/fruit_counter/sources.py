"""Input sources: image files, image directories, video files and cameras."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from fruit_counter.exceptions import InputError

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"})
DEFAULT_FPS = 30.0


class SourceKind(str, Enum):
    IMAGE = "image"
    IMAGE_DIR = "image_dir"
    VIDEO = "video"
    CAMERA = "camera"


@dataclass(frozen=True)
class InputSource:
    kind: SourceKind
    path: Path | None = None
    camera_index: int | None = None

    @property
    def name(self) -> str:
        if self.path is None:
            return f"camera{self.camera_index}"
        return self.path.stem or self.path.resolve().name

    @property
    def is_stream(self) -> bool:
        return self.kind in (SourceKind.VIDEO, SourceKind.CAMERA)

    def __str__(self) -> str:
        return self.name if self.path is None else str(self.path)


def resolve_source(spec: str | Path) -> InputSource:
    """Classify an input: camera index (``"0"``), image directory, image or video file."""
    text = str(spec).strip()
    if not text:
        raise InputError("No input given")
    if text.isdigit():
        return InputSource(SourceKind.CAMERA, camera_index=int(text))

    path = Path(text)
    if path.is_dir():
        if not list_images(path):
            raise InputError(f"Directory contains no supported images: {path}")
        return InputSource(SourceKind.IMAGE_DIR, path=path)
    if not path.exists():
        raise InputError(f"Input not found: {path}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return InputSource(SourceKind.IMAGE, path=path)
    if suffix in VIDEO_EXTENSIONS:
        return InputSource(SourceKind.VIDEO, path=path)
    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise InputError(f"Unsupported file type '{suffix}' for {path}. Supported: {supported}")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode instead of cv2.imread, which fails on non-ASCII paths on Windows.
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise InputError(f"Cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise InputError(f"Cannot decode image {path} (corrupt or unsupported format)")
    return image


def open_video(source: InputSource) -> tuple[cv2.VideoCapture, float]:
    """Open a video file or camera; returns the capture and its frame rate."""
    if source.path is None:
        capture = cv2.VideoCapture(source.camera_index or 0)
    else:
        capture = cv2.VideoCapture(str(source.path))
    if not capture.isOpened():
        capture.release()
        raise InputError(f"Cannot open {source.kind.value} source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    return capture, fps if math.isfinite(fps) and 0 < fps < 1000 else DEFAULT_FPS


def read_frames(
    capture: cv2.VideoCapture,
    fps: float,
    stride: int = 1,
    max_frames: int | None = None,
    wall_clock: bool = False,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(index, timestamp_s, frame)`` for every ``stride``-th frame.

    Timestamps come from the frame index, or from the wall clock for live cameras.
    """
    start = time.monotonic()
    index = -1
    yielded = 0
    while max_frames is None or yielded < max_frames:
        index += 1
        if index % stride:
            if not capture.grab():  # skip without decoding
                return
            continue
        ok, frame = capture.read()
        if not ok or frame is None:
            return
        yielded += 1
        yield index, time.monotonic() - start if wall_clock else index / fps, frame
