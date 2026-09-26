"""Annotated video files: H.264 through ffmpeg when it is installed, else OpenCV.

H.264 in yuv420p with ``+faststart`` plays everywhere and is accepted by social
platforms; OpenCV's built-in ``mp4v`` (MPEG-4 Part 2) is not. With ffmpeg, the
source's audio track is copied into the output too.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import IO, Protocol

import cv2
import numpy as np

from fruit_counter.exceptions import OutputError

logger = logging.getLogger(__name__)

# x264 constant-quality setting: 18 is visually close to lossless.
H264_CRF = 18


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def release(self) -> None: ...


def open_video_writer(
    path: Path, fps: float, first_frame: np.ndarray, audio_from: Path | None = None
) -> VideoWriter:
    """Writer for frames the size of ``first_frame``; ``audio_from`` adds its audio track."""
    height, width = first_frame.shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return FfmpegWriter(ffmpeg, path, fps, (width, height), audio_from)
    logger.warning(
        "ffmpeg not found: writing MPEG-4 Part 2 video without audio, which some players "
        "and platforms reject. Install ffmpeg for H.264 output."
    )
    return OpenCvWriter(path, fps, (width, height))


class OpenCvWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise OutputError(f"Cannot open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


class FfmpegWriter:
    """Pipes raw BGR frames into an ffmpeg process that encodes H.264 once."""

    def __init__(
        self,
        ffmpeg: str,
        path: Path,
        fps: float,
        size: tuple[int, int],
        audio_from: Path | None = None,
    ) -> None:
        self._path = path
        width, height = size
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        command += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}"]
        command += ["-r", f"{fps:.6g}", "-i", "-"]
        if audio_from is not None:
            command += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?"]
            command += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        # yuv420p needs even dimensions; pad by at most one pixel.
        command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", str(H264_CRF)]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        # A separate process group, so Ctrl+C stops the run but not the encoder,
        # which then finalises the partial video.
        if sys.platform == "win32":
            creationflags, new_session = subprocess.CREATE_NEW_PROCESS_GROUP, False
        else:
            creationflags, new_session = 0, True
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                start_new_session=new_session,
            )
        except OSError as exc:
            raise OutputError(f"Cannot start ffmpeg for {path}: {exc}") from exc

    def write(self, frame: np.ndarray) -> None:
        stdin = self._stdin()
        try:
            stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise OutputError(
                f"ffmpeg stopped while writing {self._path}: {self._errors()}"
            ) from exc

    def release(self) -> None:
        if self._process.poll() is None:
            with contextlib.suppress(OSError):
                self._stdin().close()
        code = self._process.wait()
        if code != 0:
            raise OutputError(f"ffmpeg failed to write {self._path}: {self._errors()}")

    def _stdin(self) -> IO[bytes]:
        assert self._process.stdin is not None
        return self._process.stdin

    def _errors(self) -> str:
        self._process.wait()
        stderr = self._process.stderr.read() if self._process.stderr else b""
        return stderr.decode(errors="replace").strip() or f"exit code {self._process.returncode}"
