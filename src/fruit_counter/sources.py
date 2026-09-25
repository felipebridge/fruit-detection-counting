"""Input sources: image files, image directories, video files and cameras."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from fruit_counter.exceptions import InputError

logger = logging.getLogger(__name__)

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
    """A validated input specification."""

    kind: SourceKind
    path: Path | None = None
    camera_index: int | None = None

    @property
    def name(self) -> str:
        """Short, filesystem-friendly name used for output naming."""
        if self.kind is SourceKind.CAMERA:
            return f"camera{self.camera_index}"
        assert self.path is not None
        return self.path.stem if self.path.stem else self.path.resolve().name

    @property
    def is_stream(self) -> bool:
        """True for sources processed frame-by-frame with tracking."""
        return self.kind in (SourceKind.VIDEO, SourceKind.CAMERA)

    def __str__(self) -> str:
        return self.name if self.kind is SourceKind.CAMERA else str(self.path)


def resolve_source(spec: str | Path) -> InputSource:
    """Turn a user-provided input string into an :class:`InputSource`.

    * A non-negative integer (``"0"``) selects a camera by index.
    * A directory selects every supported image inside it.
    * A file is classified by extension as image or video.

    Raises:
        InputError: If the path does not exist or its type is not supported.
    """
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
    """Supported images directly inside ``directory``, sorted by name."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR. Works with non-ASCII paths (unlike ``cv2.imread`` on Windows)."""
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise InputError(f"Cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image is None:
        raise InputError(f"Cannot decode image {path} (corrupt or unsupported format)")
    return image


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_s: float
    image: np.ndarray


class VideoStream:
    """Sequential frame reader for video files and cameras (context manager)."""

    def __init__(self, source: InputSource) -> None:
        if not source.is_stream:
            raise InputError(f"{source} is not a video or camera source")
        self._source = source
        if source.kind is SourceKind.CAMERA:
            assert source.camera_index is not None
            self._capture = cv2.VideoCapture(source.camera_index)
        else:
            self._capture = cv2.VideoCapture(str(source.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise InputError(f"Cannot open {source.kind.value} source: {source}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and 0 < fps < 1000 else DEFAULT_FPS
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count: int | None = (
            count if source.kind is SourceKind.VIDEO and count > 0 else None
        )

    def frames(self, stride: int = 1, max_frames: int | None = None) -> Iterator[VideoFrame]:
        """Yield every ``stride``-th frame, at most ``max_frames`` of them.

        Timestamps come from the frame position for files and from wall-clock time
        for cameras.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")
        is_camera = self._source.kind is SourceKind.CAMERA
        start = time.monotonic()
        index = -1
        yielded = 0
        while max_frames is None or yielded < max_frames:
            index += 1
            if index % stride:
                if not self._capture.grab():  # skip without decoding
                    break
                continue
            ok, image = self._capture.read()
            if not ok or image is None:
                break
            timestamp = time.monotonic() - start if is_camera else index / self.fps
            yielded += 1
            yield VideoFrame(index, timestamp, image)

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
