"""Drawing: fruit boxes with labels, and a count card in the top-left corner.

Everything scales with the frame, so a 4K video gets the same layout as a 720p one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from fruit_counter.detector import Detection

Color = tuple[int, int, int]
Box = tuple[float, float, float, float]

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
# Known fruits get a colour close to their own, so classes read at a glance.
FRUIT_COLORS: dict[str, Color] = {
    "apple": (80, 50, 225),  # crimson
    "orange": (0, 159, 230),  # orange
    "banana": (66, 228, 240),  # yellow
}
TENTATIVE_COLOR: Color = (160, 160, 160)
WHITE: Color = (255, 255, 255)
MUTED: Color = (190, 190, 190)
ACCENT: Color = (115, 200, 60)  # counter highlight (green)
CARD_COLOR: Color = (24, 24, 24)
CARD_OPACITY = 0.62
BOX_FILL_OPACITY = 0.10

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
_REFERENCE_SIDE = 720  # a 720p frame is drawn at scale 1


def frame_scale(frame: np.ndarray) -> float:
    return max(0.6, min(frame.shape[:2]) / _REFERENCE_SIDE)


@dataclass(frozen=True)
class BoxStyle:
    label: str
    color: Color
    emphasis: float = 0.0  # 0..1, a fading white halo for a fruit that was just counted


class Annotator:
    def __init__(self) -> None:
        self._colors: dict[int | str, Color] = {}

    def color_for(self, key: int | str) -> Color:
        """Colour for a class id or name: fixed for known fruits, else by first appearance.

        First-appearance order avoids ``class_id % len(PALETTE)``, which would give
        COCO's apple (47) and orange (49) near-identical colours.
        """
        if isinstance(key, str) and key.lower() in FRUIT_COLORS:
            return FRUIT_COLORS[key.lower()]
        if key not in self._colors:
            self._colors[key] = PALETTE[len(self._colors) % len(PALETTE)]
        return self._colors[key]

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        track_ids: Sequence[int | None] | None = None,
        summary_lines: Sequence[str] = (),
    ) -> np.ndarray:
        """Draw detections (and optional text lines) on ``frame`` in place and return it.

        With ``track_ids``, detections whose id is ``None`` (unconfirmed tracks) are
        drawn in grey.
        """
        if track_ids is not None and len(track_ids) != len(detections):
            raise ValueError("track_ids must have the same length as detections")
        for index, det in enumerate(detections):
            track_id = track_ids[index] if track_ids is not None else None
            if track_ids is not None and track_id is None:
                style = BoxStyle(det.class_name, TENTATIVE_COLOR)
            else:
                label = f"{det.class_name.capitalize()} {det.confidence:.0%}"
                if track_id is not None:
                    label = f"{det.class_name.capitalize()} #{track_id}"
                style = BoxStyle(label, self.color_for(det.class_name))
            draw_box(frame, det.box, style)
        if summary_lines:
            draw_text_card(frame, summary_lines)
        return frame

    def draw_counter(
        self,
        frame: np.ndarray,
        title: str,
        total: int,
        counts_by_class: Mapping[str, int],
        footer: str | None = None,
        highlight: float = 0.0,
    ) -> np.ndarray:
        """Draw the count card: title, a large total, one row per class, optional footer.

        ``highlight`` (0..1) tints the total with the accent colour, e.g. right after a
        new fruit is counted.
        """
        s = frame_scale(frame)
        margin, pad = round(24 * s), round(18 * s)
        bar = max(3, round(5 * s))
        title_font = (_FONT, 0.5 * s, max(1, round(1.2 * s)))
        total_font = (_FONT_BOLD, 2.0 * s, max(2, round(3.2 * s)))
        row_font = (_FONT, 0.6 * s, max(1, round(1.5 * s)))

        rows = [(name.capitalize(), str(count)) for name, count in counts_by_class.items()]
        title_w, title_h = _text_size(title.upper(), title_font)
        total_w, total_h = _text_size(max(str(total), "88", key=len), total_font)
        row_h = _text_size("Ag", row_font)[1]
        dot = round(5 * s)
        name_w = max((_text_size(n, row_font)[0] for n, _ in rows), default=0)
        count_w = max((_text_size(c, row_font)[0] for _, c in rows), default=0)
        rows_w = 2 * dot + round(10 * s) + name_w + round(24 * s) + count_w
        footer_w = _text_size(footer, title_font)[0] if footer else 0

        width = bar + 2 * pad + max(title_w, total_w, rows_w, footer_w, round(150 * s))
        total_gap, rows_gap, row_gap, footer_gap = (round(v * s) for v in (16, 10, 12, 14))
        height = pad + title_h + total_gap + total_h
        if rows:
            height += rows_gap + len(rows) * (row_gap + row_h)
        if footer:
            height += footer_gap + title_h
        height += pad

        x1, y1 = margin, margin
        x2 = min(frame.shape[1] - 1, x1 + width)
        y2 = min(frame.shape[0] - 1, y1 + height)
        _blend_rounded_rect(frame, (x1, y1, x2, y2), round(10 * s), CARD_COLOR, CARD_OPACITY)
        accent = _mix(MUTED, ACCENT, 0.35 + 0.65 * highlight)
        cv2.rectangle(frame, (x1, y1 + round(10 * s)), (x1 + bar, y2 - round(10 * s)), accent, -1)

        x = x1 + bar + pad
        y = y1 + pad + title_h
        _put(frame, title.upper(), (x, y), title_font, MUTED)
        y += total_gap + total_h
        _put(frame, str(total), (x, y), total_font, _mix(WHITE, ACCENT, highlight))
        if rows:
            y += rows_gap
        for (name, count), key in zip(rows, counts_by_class, strict=True):
            y += row_gap + row_h
            color = self.color_for(key)
            cv2.circle(frame, (x + dot, y - row_h // 2), dot, color, -1, cv2.LINE_AA)
            _put(frame, name, (x + 2 * dot + round(10 * s), y), row_font, WHITE)
            count_x = x2 - pad - _text_size(count, row_font)[0]
            _put(frame, count, (count_x, y), row_font, WHITE)
        if footer:
            y += footer_gap + title_h
            _put(frame, footer, (x, y), title_font, MUTED)
        return frame


def draw_box(frame: np.ndarray, box: Box, style: BoxStyle) -> None:
    """A thin box with bold corners, a faint fill and a label tab above it."""
    s = frame_scale(frame)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (round(c) for c in box)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1 : y2 + 1, x1 : x2 + 1]
    region[:] = cv2.addWeighted(
        region, 1 - BOX_FILL_OPACITY, np.full_like(region, style.color), BOX_FILL_OPACITY, 0
    )
    if style.emphasis > 0:
        # A white halo that shrinks onto the box and fades out.
        grow = round(12 * s * (1 - style.emphasis))
        thick = max(1, round(3 * s))
        hx1, hy1 = max(0, x1 - grow - thick), max(0, y1 - grow - thick)
        hx2, hy2 = min(width, x2 + grow + thick + 1), min(height, y2 + grow + thick + 1)
        area = frame[hy1:hy2, hx1:hx2]
        overlay = area.copy()
        cv2.rectangle(
            overlay,
            (x1 - grow - hx1, y1 - grow - hy1),
            (x2 + grow - hx1, y2 + grow - hy1),
            WHITE,
            thick,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, style.emphasis, area, 1 - style.emphasis, 0, dst=area)

    thin = max(1, round(1.5 * s))
    cv2.rectangle(frame, (x1, y1), (x2, y2), style.color, thin, cv2.LINE_AA)
    corner = max(4, round(min(x2 - x1, y2 - y1) * 0.22))
    bold = max(2, round(4 * s))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), style.color, bold, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), style.color, bold, cv2.LINE_AA)

    font = (_FONT, 0.55 * s, max(1, round(1.4 * s)))
    tw, th = _text_size(style.label, font)
    pad = round(6 * s)
    tab_h = th + 2 * pad
    top = y1 - tab_h - round(3 * s)
    if top < 0:  # no room above: put the tab inside the box
        top = y1 + round(3 * s)
    left = min(x1, max(0, width - tw - 2 * pad))
    _blend_rounded_rect(
        frame, (left, top, left + tw + 2 * pad, top + tab_h), round(5 * s), style.color, 1.0
    )
    _put(frame, style.label, (left + pad, top + pad + th), font, _text_color_on(style.color))


def draw_text_card(frame: np.ndarray, lines: Sequence[str]) -> None:
    """Plain lines of white text on a translucent card in the top-left corner."""
    s = frame_scale(frame)
    font = (_FONT, 0.6 * s, max(1, round(1.5 * s)))
    sizes = [_text_size(line, font) for line in lines]
    pad = round(12 * s)
    line_h = max(h for _, h in sizes) + pad
    width = max(w for w, _ in sizes) + 2 * pad
    height = line_h * len(lines) + pad
    margin = round(16 * s)
    box = (margin, margin, margin + width, margin + height)
    _blend_rounded_rect(frame, box, round(8 * s), CARD_COLOR, CARD_OPACITY)
    for i, line in enumerate(lines):
        _put(frame, line, (margin + pad, margin + pad + line_h * i + sizes[i][1]), font, WHITE)


class BoxSmoother:
    """Steadies track boxes for display; counting never sees these values.

    Boxes follow detections with an exponential moving average, and a track that
    misses a few frames keeps moving at its last velocity instead of blinking out.
    """

    def __init__(self, alpha: float = 0.6, hold_frames: int = 4) -> None:
        self._alpha = alpha
        self._hold = hold_frames
        self._state: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}  # box, velocity, misses

    def update(self, observed: Mapping[int, Box]) -> dict[int, Box]:
        """Feed this frame's boxes per track id; returns the boxes to draw."""
        shown: dict[int, Box] = {}
        for track_id, raw in observed.items():
            measured = np.asarray(raw, dtype=np.float64)
            previous = self._state.get(track_id)
            if previous is None:
                box, velocity = measured, np.zeros(4)
            else:
                old_box, old_velocity, misses = previous
                predicted = old_box + old_velocity * (misses + 1)
                box = self._alpha * measured + (1 - self._alpha) * predicted
                velocity = 0.5 * (box - old_box) / (misses + 1) + 0.5 * old_velocity
            self._state[track_id] = (box, velocity, 0)
            shown[track_id] = _as_box(box)
        for track_id in list(self._state):
            if track_id in observed:
                continue
            box, velocity, misses = self._state[track_id]
            if misses >= self._hold:
                del self._state[track_id]
                continue
            self._state[track_id] = (box + velocity, velocity, misses + 1)
            shown[track_id] = _as_box(box + velocity)
        return shown


def _as_box(values: np.ndarray) -> Box:
    x1, y1, x2, y2 = (float(v) for v in values)
    return x1, y1, x2, y2


def _text_size(text: str, font: tuple[int, float, int]) -> tuple[int, int]:
    face, scale, thickness = font
    (w, h), _ = cv2.getTextSize(text, face, scale, thickness)
    return w, h


def _put(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: tuple[int, float, int],
    color: Color,
) -> None:
    face, scale, thickness = font
    cv2.putText(frame, text, origin, face, scale, color, thickness, cv2.LINE_AA)


def _mix(a: Color, b: Color, t: float) -> Color:
    t = min(1.0, max(0.0, t))
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def _text_color_on(color: Color) -> Color:
    b, g, r = color
    return (20, 20, 20) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else WHITE


def _blend_rounded_rect(
    frame: np.ndarray, box: tuple[int, int, int, int], radius: int, color: Color, opacity: float
) -> None:
    """Fill a rounded rectangle, blended into ``frame`` (only its region is touched)."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(width - 1, box[2]), min(height - 1, box[3])
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1 : y2 + 1, x1 : x2 + 1]
    h, w = region.shape[:2]
    r = max(0, min(radius, h // 2, w // 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    alpha = (mask.astype(np.float32) / 255.0 * opacity)[..., None]
    blended = region * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    region[:] = blended.astype(np.uint8)
