"""Drawing of boxes, labels, track ids and a summary panel."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from fruit_counter.detector import Detection

Color = tuple[int, int, int]

# Colour-blind-friendly Okabe-Ito palette (BGR).
PALETTE: tuple[Color, ...] = (
    (178, 114, 0),
    (0, 159, 230),
    (115, 158, 0),
    (167, 121, 204),
    (66, 228, 240),
    (233, 180, 86),
    (0, 94, 213),
)
TENTATIVE_COLOR: Color = (160, 160, 160)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REFERENCE_SIDE = 720  # larger images get proportionally thicker lines and text


class Annotator:
    def __init__(self) -> None:
        self._class_colors: dict[int, Color] = {}

    def color_for(self, class_id: int) -> Color:
        # Assigned in order of first appearance: with class_id % len(PALETTE), COCO's
        # apple (47) and orange (49) would get near-identical colours.
        if class_id not in self._class_colors:
            self._class_colors[class_id] = PALETTE[len(self._class_colors) % len(PALETTE)]
        return self._class_colors[class_id]

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        track_ids: Sequence[int | None] | None = None,
        summary_lines: Sequence[str] = (),
    ) -> np.ndarray:
        """Draw on ``frame`` in place and return it.

        With ``track_ids``, detections whose id is ``None`` (unconfirmed tracks) are
        drawn in grey.
        """
        if track_ids is not None and len(track_ids) != len(detections):
            raise ValueError("track_ids must have the same length as detections")

        scale = max(1.0, min(frame.shape[:2]) / _REFERENCE_SIDE)
        thickness = max(1, round(2 * scale))
        font_scale = 0.6 * scale

        for index, det in enumerate(detections):
            track_id = track_ids[index] if track_ids is not None else None
            unconfirmed = track_ids is not None and track_id is None
            color = TENTATIVE_COLOR if unconfirmed else self.color_for(det.class_id)
            x1, y1, x2, y2 = (round(c) for c in det.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            label = f"{det.class_name} {det.confidence:.2f}"
            if track_id is not None:
                label = f"#{track_id} {label}"
            _draw_label(frame, label, (x1, y1), color, font_scale)

        if summary_lines:
            _draw_panel(frame, summary_lines, font_scale)
        return frame


def _draw_label(
    frame: np.ndarray, text: str, anchor: tuple[int, int], color: Color, font_scale: float
) -> None:
    thickness = max(1, round(font_scale * 1.5))
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = anchor
    # Above the box, or inside it when there is no room above.
    top = y - th - baseline - 4 if y - th - baseline - 4 >= 0 else y
    cv2.rectangle(frame, (x, top), (x + tw + 4, top + th + baseline + 4), color, cv2.FILLED)
    b, g, r = color
    text_color = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)
    cv2.putText(
        frame, text, (x + 2, top + th + 2), _FONT, font_scale, text_color, thickness, cv2.LINE_AA
    )


def _draw_panel(frame: np.ndarray, lines: Sequence[str], font_scale: float) -> None:
    thickness = max(1, round(font_scale * 1.5))
    sizes = [cv2.getTextSize(line, _FONT, font_scale, thickness)[0] for line in lines]
    pad = round(10 * font_scale)
    line_height = max(h for _, h in sizes) + pad
    width = max(w for w, _ in sizes) + 2 * pad
    height = line_height * len(lines) + pad

    region = frame[0:height, 0:width]
    region[:] = (region * 0.4).astype(np.uint8)  # darken the background
    for i, line in enumerate(lines):
        y = pad + line_height * i + sizes[i][1]
        cv2.putText(
            frame, line, (pad, y), _FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
