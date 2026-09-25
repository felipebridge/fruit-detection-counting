import numpy as np
import pytest

from fruit_counter.exceptions import InputError
from fruit_counter.preprocessing import ensure_bgr


def test_bgr_frame_passes_through_unchanged() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    out = ensure_bgr(frame)
    assert out.shape == (48, 64, 3)
    np.testing.assert_array_equal(out, frame)


@pytest.mark.parametrize("shape", [(48, 64), (48, 64, 1)])
def test_grayscale_is_expanded_to_three_channels(shape: tuple[int, ...]) -> None:
    frame = np.full(shape, 120, dtype=np.uint8)
    out = ensure_bgr(frame)
    assert out.shape == (48, 64, 3)
    assert (out == 120).all()


def test_alpha_channel_is_dropped() -> None:
    frame = np.zeros((10, 10, 4), dtype=np.uint8)
    frame[..., 2] = 200  # red
    frame[..., 3] = 7  # alpha
    out = ensure_bgr(frame)
    assert out.shape == (10, 10, 3)
    assert (out[..., 2] == 200).all()


def test_non_contiguous_input_becomes_contiguous() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)[:, ::2]
    assert ensure_bgr(frame).flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((0, 0, 3), dtype=np.uint8), "empty"),
        (np.zeros((10, 10, 3), dtype=np.float32), "uint8"),
        (np.zeros((10, 10, 2), dtype=np.uint8), "shape"),
        (np.zeros((2, 10, 10, 3), dtype=np.uint8), "shape"),
        ([[0, 0, 0]], "numpy array"),
    ],
)
def test_invalid_frames_are_rejected(frame: object, message: str) -> None:
    with pytest.raises(InputError, match=message):
        ensure_bgr(frame)  # type: ignore[arg-type]
