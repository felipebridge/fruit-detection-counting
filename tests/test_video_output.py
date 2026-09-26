import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from fruit_counter import video_output
from fruit_counter.video_output import FfmpegWriter, OpenCvWriter, open_video_writer

FRAME = np.full((120, 160, 3), 90, dtype=np.uint8)
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def write_frames(path: Path, frames: int, audio_from: Path | None = None) -> None:
    writer = open_video_writer(path, 10.0, FRAME, audio_from)
    for _ in range(frames):
        writer.write(FRAME)
    writer.release()


def test_falls_back_to_opencv_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(video_output.shutil, "which", lambda name: None)
    write_frames(tmp_path / "out.mp4", 5)
    capture = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()
    writer = open_video_writer(tmp_path / "other.mp4", 10, FRAME)
    assert isinstance(writer, OpenCvWriter)
    writer.release()


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg is not installed")
def test_ffmpeg_writes_h264_and_keeps_the_source_audio(tmp_path: Path) -> None:
    assert FFMPEG is not None and FFPROBE is not None
    source = tmp_path / "source.mp4"
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x120:d=1",
         "-f", "lavfi", "-i", "sine=d=1", "-shortest", str(source)],
        check=True,
    )  # fmt: skip
    output = tmp_path / "out.mp4"
    write_frames(output, 10, audio_from=source)

    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert streams["video"]["codec_name"] == "h264"
    assert streams["video"]["pix_fmt"] == "yuv420p"
    assert (streams["video"]["width"], streams["video"]["height"]) == (160, 120)
    assert "audio" in streams


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_odd_frame_sizes_are_padded_for_h264(tmp_path: Path) -> None:
    odd = np.zeros((121, 161, 3), dtype=np.uint8)
    writer = open_video_writer(tmp_path / "odd.mp4", 10, odd)
    assert isinstance(writer, FfmpegWriter)
    for _ in range(3):
        writer.write(odd)
    writer.release()
    capture = cv2.VideoCapture(str(tmp_path / "odd.mp4"))
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (
        162,
        122,
    )
    capture.release()
