"""Frame validation and normalisation before inference.

Resizing and letterboxing are handled by the detector itself; this module only
guarantees that every frame entering the pipeline is a contiguous 3-channel BGR
``uint8`` image, whatever the source produced.
"""

from __future__ import annotations

import cv2
import numpy as np

from fruit_counter.exceptions import InputError


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Return ``frame`` as a contiguous ``HxWx3`` BGR ``uint8`` array.

    Grayscale (``HxW`` or ``HxWx1``) and BGRA (``HxWx4``) frames are converted.

    Raises:
        InputError: If the frame is empty or has an unsupported shape or dtype.
    """
    if not isinstance(frame, np.ndarray):
        raise InputError(f"Expected a numpy array, got {type(frame).__name__}")
    if frame.size == 0:
        raise InputError("Received an empty frame")
    if frame.dtype != np.uint8:
        raise InputError(f"Expected a uint8 image, got dtype {frame.dtype}")

    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif not (frame.ndim == 3 and frame.shape[2] == 3):
        raise InputError(f"Unsupported frame shape {frame.shape}")
    return np.ascontiguousarray(frame)
