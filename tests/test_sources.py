from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import SAMPLE_IMAGE
from fruit_counter.exceptions import InputError
from fruit_counter.sources import (
    InputSource,
    SourceKind,
    VideoStream,
    list_images,
    read_image,
    resolve_source,
)


def write_video(path: Path, frames: int, fps: float = 10.0, size: tuple[int, int] = (64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ------------------------------------------------------------------ resolution


def test_resolves_image_file() -> None:
    source = resolve_source(str(SAMPLE_IMAGE))
    assert source.kind is SourceKind.IMAGE
    assert source.name == "fruit_bowl"
    assert not source.is_stream


def test_resolves_video_by_extension_case_insensitively(tmp_path: Path) -> None:
    video = tmp_path / "clip.MP4"
    video.write_bytes(b"not decoded during resolution")
    source = resolve_source(video)
    assert source.kind is SourceKind.VIDEO
    assert source.is_stream


def test_digits_select_a_camera() -> None:
    source = resolve_source("1")
    assert source == InputSource(SourceKind.CAMERA, camera_index=1)
    assert source.name == "camera1"
    assert source.is_stream


def test_directory_of_images(tmp_path: Path) -> None:
    for name in ["b.png", "a.jpg", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    source = resolve_source(tmp_path)
    assert source.kind is SourceKind.IMAGE_DIR
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.png"]


@pytest.mark.parametrize(
    ("name", "message"),
    [("missing.jpg", "not found"), ("data.csv", "Unsupported file type")],
)
def test_invalid_inputs(tmp_path: Path, name: str, message: str) -> None:
    if name == "data.csv":
        (tmp_path / name).write_text("a,b")
    with pytest.raises(InputError, match=message):
        resolve_source(tmp_path / name)


def test_empty_directory_and_blank_input_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="no supported images"):
        resolve_source(tmp_path)
    with pytest.raises(InputError, match="No input"):
        resolve_source("  ")


# ------------------------------------------------------------------ image reading


def test_read_image_supports_non_ascii_paths(tmp_path: Path) -> None:
    target = tmp_path / "manzanas_ñ.jpg"
    target.write_bytes(SAMPLE_IMAGE.read_bytes())
    image = read_image(target)
    assert image.ndim == 3 and image.shape[2] == 3


def test_read_image_rejects_corrupt_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"definitely not a jpeg")
    with pytest.raises(InputError, match="Cannot decode"):
        read_image(broken)
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(InputError, match="Cannot decode"):
        read_image(empty)


# ------------------------------------------------------------------ video reading


def test_video_stream_reads_all_frames_with_timestamps(tmp_path: Path) -> None:
    source = resolve_source(write_video(tmp_path / "v.avi", frames=12, fps=10))
    with VideoStream(source) as stream:
        assert stream.fps == pytest.approx(10)
        assert (stream.width, stream.height) == (64, 48)
        assert stream.frame_count == 12
        frames = list(stream.frames())
    assert [f.index for f in frames] == list(range(12))
    assert frames[5].timestamp_s == pytest.approx(0.5)
    assert frames[0].image.shape == (48, 64, 3)


def test_video_stream_stride_and_max_frames(tmp_path: Path) -> None:
    source = resolve_source(write_video(tmp_path / "v.avi", frames=12))
    with VideoStream(source) as stream:
        assert [f.index for f in stream.frames(stride=3)] == [0, 3, 6, 9]
    with VideoStream(source) as stream:
        assert [f.index for f in stream.frames(stride=2, max_frames=2)] == [0, 2]


def test_unreadable_video_raises_input_error(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"garbage")
    with pytest.raises(InputError, match="Cannot open"):
        VideoStream(resolve_source(fake))


def test_video_stream_rejects_image_sources() -> None:
    with pytest.raises(InputError, match="not a video"):
        VideoStream(resolve_source(SAMPLE_IMAGE))
