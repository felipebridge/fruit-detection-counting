import importlib.util

import numpy as np
import pytest

from conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "make_demo_video", REPO_ROOT / "scripts" / "make_demo_video.py"
)
assert _spec is not None and _spec.loader is not None
make_demo_video = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_demo_video)


def test_pan_covers_image_from_top_to_bottom() -> None:
    # Each row holds its own index, so a crop reveals its vertical position.
    image = np.repeat(np.arange(200, dtype=np.uint8)[:, None, None], 100, axis=1)
    image = np.repeat(image, 3, axis=2)
    frames = make_demo_video.pan_frames(image, width=80, height=50, num_frames=20)

    assert len(frames) == 20
    assert all(f.shape == (50, 80, 3) for f in frames)
    assert frames[0][0, 0, 0] == 0  # starts at the top
    assert frames[-1][-1, 0, 0] == 199  # ends at the bottom
    tops = [int(f[0, 0, 0]) for f in frames]
    assert tops == sorted(tops)  # moves monotonically downwards


def test_window_larger_than_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="larger than the image"):
        make_demo_video.pan_frames(np.zeros((10, 10, 3), np.uint8), 20, 5, 3)
