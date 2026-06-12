import cv2
import numpy as np


# Colour palette (BGR)
_GREEN = (50, 200, 50)
_CYAN = (200, 200, 0)
_RED = (50, 50, 230)
_WHITE = (240, 240, 240)
_DARK = (20, 20, 20)
_YELLOW = (0, 210, 255)

_GESTURE_COLOURS = {
    'move_cursor':   _CYAN,
    'left_click':    _GREEN,
    'right_click':   _YELLOW,
    'scroll_up':     _WHITE,
    'scroll_down':   _WHITE,
    'volume_up':     _GREEN,
    'volume_down':   _RED,
    'screenshot':    _YELLOW,
    'none':          (100, 100, 100),
}

_GESTURE_LABELS = {
    'move_cursor':  'MOVE',
    'left_click':   'LEFT CLICK',
    'right_click':  'RIGHT CLICK',
    'scroll_up':    'SCROLL UP',
    'scroll_down':  'SCROLL DOWN',
    'volume_up':    'VOL +',
    'volume_down':  'VOL -',
    'screenshot':   'SCREENSHOT',
    'none':         '',
}

_CHEATSHEET = [
    ('Index only',         'Move cursor'),
    ('Pinch (thumb+idx)',  'Left click'),
    ('Thumb+middle close', 'Right click'),
    ('Index+middle up',    'Scroll'),
    ('Rock on',            'Volume up'),
    ('Fist',               'Volume down'),
    ('Open palm',          'Screenshot'),
]


def draw_hud(frame, gesture: str, volume_pct: int, fps: float, show_help: bool):
    h, w = frame.shape[:2]

    # --- semi-transparent top bar ---
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), _DARK, -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f'FPS {fps:.0f}', (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _WHITE, 2)

    # Gesture label
    label = _GESTURE_LABELS.get(gesture, gesture.upper())
    colour = _GESTURE_COLOURS.get(gesture, _WHITE)
    if label:
        cv2.putText(frame, label, (w // 2 - 80, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    # Volume bar (right side)
    _draw_volume_bar(frame, volume_pct, w - 180, 10, 160, 32)

    # Help hint
    hint = 'H - hide help' if show_help else 'H - show help'
    cv2.putText(frame, hint, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    if show_help:
        _draw_cheatsheet(frame)


def draw_cursor_ring(frame, tip_px: tuple[int, int]):
    cv2.circle(frame, tip_px, 14, _CYAN, 2)
    cv2.circle(frame, tip_px, 3, _CYAN, -1)


def draw_pinch_line(frame, thumb_px: tuple[int, int], idx_px: tuple[int, int], active: bool):
    colour = _GREEN if active else (100, 100, 100)
    mid = ((thumb_px[0] + idx_px[0]) // 2, (thumb_px[1] + idx_px[1]) // 2)
    cv2.line(frame, thumb_px, idx_px, colour, 2)
    cv2.circle(frame, mid, 8, colour, -1)


def draw_flash(frame, text: str, colour=_YELLOW):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 180, h // 2 - 40), (w // 2 + 180, h // 2 + 40), _DARK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (w // 2 - 160, h // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _draw_volume_bar(frame, pct: int, x: int, y: int, bar_w: int, bar_h: int):
    filled = int(bar_w * pct / 100)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), _GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), _WHITE, 1)
    cv2.putText(frame, f'Vol {pct}%', (x, y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)


def _draw_cheatsheet(frame):
    x0, y0 = 12, 70
    line_h = 24
    pad = 10
    box_h = len(_CHEATSHEET) * line_h + pad * 2
    box_w = 310

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad), (x0 + box_w, y0 + box_h), _DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i, (gesture, action) in enumerate(_CHEATSHEET):
        y = y0 + i * line_h + line_h // 2
        cv2.putText(frame, gesture, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _CYAN, 1)
        cv2.putText(frame, f'→ {action}', (x0 + 168, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1)
