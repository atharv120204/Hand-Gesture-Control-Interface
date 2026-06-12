from collections import deque
from hand_tracker import HandTracker
import config


class Gesture:
    NONE = "none"
    MOVE_CURSOR = "move_cursor"     # index up only
    LEFT_CLICK = "left_click"       # pinch (thumb + index close)
    RIGHT_CLICK = "right_click"     # thumb + middle pinch
    SCROLL_UP = "scroll_up"         # index + middle up, move up
    SCROLL_DOWN = "scroll_down"     # index + middle up, move down
    VOLUME_UP = "volume_up"         # thumb + pinky spread (rock-on open)
    VOLUME_DOWN = "volume_down"     # fist
    SCREENSHOT = "screenshot"       # open palm (all fingers up)
    DRAG = "drag"                   # index + middle pinch + move


class GestureRecognizer:
    """
    Classifies hand landmarks into named gestures with temporal debouncing.
    """

    def __init__(self):
        self._history: deque[str] = deque(maxlen=config.GESTURE_HOLD_FRAMES)
        self._scroll_y_history: deque[int] = deque(maxlen=6)

    def classify(self, hand: dict) -> tuple[str, dict]:
        """
        Returns (gesture_name, extra_data).
        extra_data varies per gesture:
          - MOVE_CURSOR: {'tip_px': (x, y)}
          - LEFT_CLICK / RIGHT_CLICK: {}
          - SCROLL_*: {'delta': int}
          - VOLUME_*: {}
          - SCREENSHOT: {}
          - NONE: {}
        """
        fingers = HandTracker.fingers_up(hand)  # [idx, mid, ring, pinky, thumb]
        idx, mid, ring, pinky, thumb = fingers
        pinch_dist = HandTracker.pinch_distance(hand)

        # --- prioritise pinch gestures first ---

        # Left click: thumb + index pinched, others relaxed
        if pinch_dist < config.PINCH_THRESHOLD and not mid:
            return self._debounce(Gesture.LEFT_CLICK, {})

        # Right click: thumb + middle close (use tip distance)
        thumb_mid_dist = self._tip_dist(hand, 4, 12)
        if thumb_mid_dist < config.PINCH_THRESHOLD and not idx:
            return self._debounce(Gesture.RIGHT_CLICK, {})

        # Screenshot: all five fingers up
        if all(fingers):
            return self._debounce(Gesture.SCREENSHOT, {})

        # Volume down: full fist (no fingers up)
        if not any(fingers):
            return self._debounce(Gesture.VOLUME_DOWN, {})

        # Volume up: thumb + pinky up, middle three down (rock-on)
        if thumb and pinky and not idx and not mid and not ring:
            return self._debounce(Gesture.VOLUME_UP, {})

        # Scroll: index + middle up, ring + pinky down
        if idx and mid and not ring and not pinky:
            tip_y = hand['points'][8][1]
            self._scroll_y_history.append(tip_y)
            if len(self._scroll_y_history) >= 4:
                delta = self._scroll_y_history[0] - self._scroll_y_history[-1]
                if abs(delta) > 12:
                    gesture = Gesture.SCROLL_UP if delta > 0 else Gesture.SCROLL_DOWN
                    return self._debounce(gesture, {'delta': abs(delta)})
            return self._debounce(Gesture.NONE, {})

        # Move cursor: only index up
        if idx and not mid and not ring and not pinky:
            tip_px = hand['points'][8]
            return self._debounce(Gesture.MOVE_CURSOR, {'tip_px': tip_px})

        self._scroll_y_history.clear()
        return self._debounce(Gesture.NONE, {})

    def _debounce(self, gesture: str, data: dict) -> tuple[str, dict]:
        self._history.append(gesture)
        if len(self._history) == config.GESTURE_HOLD_FRAMES and len(set(self._history)) == 1:
            return gesture, data
        return Gesture.NONE, {}

    @staticmethod
    def _tip_dist(hand: dict, a: int, b: int) -> float:
        import numpy as np
        norm = hand['norm']
        return float(np.hypot(norm[a][0] - norm[b][0], norm[a][1] - norm[b][1]))
