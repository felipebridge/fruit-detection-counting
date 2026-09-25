"""Rendering of detections, track identities and summary overlays."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from fruit_counter.config import VisualizationConfig
from fruit_counter.structures import Detection

Color = tuple[int, int, int]

# Distinct, colour-blind-friendly BGR colours (Okabe-Ito palette).
PALETTE: tuple[Color, ...] = (
    (0, 159, 230),  # orange
    (233, 180, 86),  # sky blue
    (115, 158, 0),  # bluish green
    (66, 228, 240),  # yellow
    (178, 114, 0),  # blue
    (0, 94, 213),  # vermillion
    (167, 121, 204),  # reddish purple
)
TENTATIVE_COLOR: Color = (160, 160, 160)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REFERENCE_SIDE = 720  # images larger than this get proportionally thicker drawings


def color_for_class(class_id: int) -> Color:
    """Stable colour for a class id."""
    return PALETTE[class_id % len(PALETTE)]


class Annotator:
    """Draws detections and summary panels onto copies of frames."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self._config = config or VisualizationConfig()

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        track_ids: Sequence[int | None] | None = None,
        summary_lines: Sequence[str] = (),
    ) -> np.ndarray:
        """Return an annotated copy of ``frame``.

        Args:
            frame: BGR image; it is not modified.
            detections: Detections to draw.
            track_ids: Optional identity per detection. ``None`` marks a detection
                whose track is not confirmed yet; it is drawn in grey without an id.
            summary_lines: Text rendered in a panel at the top-left corner.
        """
        if track_ids is not None and len(track_ids) != len(detections):
            raise ValueError("track_ids must have the same length as detections")

        canvas = frame.copy()
        scale = max(1.0, min(canvas.shape[:2]) / _REFERENCE_SIDE)
        thickness = max(1, round(self._config.line_thickness * scale))
        font_scale = self._config.font_scale * scale

        for index, det in enumerate(detections):
            track_id = track_ids[index] if track_ids is not None else None
            unconfirmed = track_ids is not None and track_id is None
            color = TENTATIVE_COLOR if unconfirmed else color_for_class(det.class_id)
            x1, y1, x2, y2 = (round(c) for c in det.box.to_xyxy())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            self._draw_label(canvas, self._label(det, track_id), (x1, y1), color, font_scale)

        if self._config.show_summary and summary_lines:
            self._draw_panel(canvas, summary_lines, font_scale)
        return canvas

    def _label(self, det: Detection, track_id: int | None) -> str:
        parts = []
        if self._config.show_track_ids and track_id is not None:
            parts.append(f"#{track_id}")
        parts.append(det.class_name)
        if self._config.show_confidence:
            parts.append(f"{det.confidence:.2f}")
        return " ".join(parts)

    @staticmethod
    def _draw_label(
        canvas: np.ndarray, text: str, anchor: tuple[int, int], color: Color, font_scale: float
    ) -> None:
        thickness = max(1, round(font_scale * 1.5))
        (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
        x, y = anchor
        # Put the label above the box, or inside it when there is no room above.
        top = y - th - baseline - 4 if y - th - baseline - 4 >= 0 else y
        cv2.rectangle(canvas, (x, top), (x + tw + 4, top + th + baseline + 4), color, cv2.FILLED)
        cv2.putText(
            canvas,
            text,
            (x + 2, top + th + 2),
            _FONT,
            font_scale,
            _text_color(color),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_panel(canvas: np.ndarray, lines: Sequence[str], font_scale: float) -> None:
        thickness = max(1, round(font_scale * 1.5))
        sizes = [cv2.getTextSize(line, _FONT, font_scale, thickness)[0] for line in lines]
        line_height = max(h for _, h in sizes) + round(10 * font_scale)
        pad = round(10 * font_scale)
        width = max(w for w, _ in sizes) + 2 * pad
        height = line_height * len(lines) + pad

        region = canvas[0:height, 0:width]
        overlay = np.zeros_like(region)
        cv2.addWeighted(overlay, 0.6, region, 0.4, 0, dst=region)
        for i, line in enumerate(lines):
            y = pad + line_height * i + sizes[i][1]
            cv2.putText(
                canvas, line, (pad, y), _FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )


def _text_color(background: Color) -> Color:
    b, g, r = background
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)
