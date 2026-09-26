"""Generate a demo video by panning a virtual camera across a still image.

Fruits enter the view, stay visible for many frames and leave again, which is
exactly the situation where naive per-frame counting overcounts. Because the video
is derived from a single photo, the expected unique count is known: it should match
the number of fruits detected in the full image.

Usage:
    python scripts/make_demo_video.py
    python scripts/make_demo_video.py --image my_photo.jpg --seconds 8 --output demo.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "data" / "samples" / "fruit_bowl.jpg"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "samples" / "fruit_bowl_pan.mp4"


def pan_frames(image: np.ndarray, width: int, height: int, num_frames: int) -> list[np.ndarray]:
    """Crop a ``width x height`` window moving from the top to the bottom of ``image``.

    The window follows an eased vertical path with a gentle horizontal sway, so the
    apparent fruit motion has varying speed like a hand-held camera.
    """
    img_h, img_w = image.shape[:2]
    if width > img_w or height > img_h:
        raise ValueError(f"Window {width}x{height} is larger than the image {img_w}x{img_h}")
    max_x, max_y = img_w - width, img_h - height
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        eased = (1 - math.cos(math.pi * t)) / 2  # slow start and end
        y = round(eased * max_y)
        x = round(max_x / 2 * (1 + 0.6 * math.sin(2 * math.pi * t)))
        frames.append(image[y : y + height, x : x + width].copy())
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    frames = pan_frames(image, args.width, args.height, round(args.fps * args.seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot open video writer for {args.output}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(
        f"Wrote {len(frames)} frames ({args.width}x{args.height} @ {args.fps} fps) to {args.output}"
    )


if __name__ == "__main__":
    main()
