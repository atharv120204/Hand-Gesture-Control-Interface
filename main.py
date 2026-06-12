"""
Hand Gesture Control Interface
-------------------------------
Controls:
  Q / ESC  - Quit
  H        - Toggle gesture cheatsheet
  P        - Pause / Resume gesture recognition
"""

import time
import cv2

import config
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, Gesture
from actions import ActionController
from overlay import (
    draw_hud, draw_cursor_ring, draw_pinch_line, draw_flash
)


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {config.CAMERA_INDEX}')

    tracker = HandTracker(
        max_hands=config.MAX_HANDS,
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    recognizer = GestureRecognizer()
    controller = ActionController(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    show_help = True
    paused = False
    flash_text = ''
    flash_until = 0.0

    fps_buf = []
    prev_t = time.time()

    print('[INFO] Starting — press Q or ESC to quit, H for help, P to pause.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[WARN] Frame grab failed, retrying…')
            continue

        # FPS
        now = time.time()
        fps_buf.append(1.0 / max(now - prev_t, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)
        prev_t = now

        frame, hands = tracker.process(frame)

        gesture = Gesture.NONE
        extra = {}

        if hands and not paused:
            hand = hands[0]
            gesture, extra = recognizer.classify(hand)

            pts = hand['points']
            thumb_px = pts[4]
            idx_px = pts[8]
            mid_px = pts[12]

            # --- visual feedback ---
            if gesture == Gesture.MOVE_CURSOR:
                draw_cursor_ring(frame, idx_px)

            pinch_active = gesture in (Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK)
            draw_pinch_line(frame, thumb_px, idx_px, pinch_active)

            # --- dispatch actions ---
            if gesture == Gesture.MOVE_CURSOR:
                controller.move_cursor(extra['tip_px'])

            elif gesture == Gesture.LEFT_CLICK:
                controller.left_click()

            elif gesture == Gesture.RIGHT_CLICK:
                controller.right_click()

            elif gesture == Gesture.SCROLL_UP:
                controller.scroll('up', extra.get('delta', config.SCROLL_SENSITIVITY))

            elif gesture == Gesture.SCROLL_DOWN:
                controller.scroll('down', extra.get('delta', config.SCROLL_SENSITIVITY))

            elif gesture == Gesture.VOLUME_UP:
                controller.volume_up()

            elif gesture == Gesture.VOLUME_DOWN:
                controller.volume_down()

            elif gesture == Gesture.SCREENSHOT:
                path = controller.screenshot()
                flash_text = f'Saved: {path}'
                flash_until = time.time() + 2.0

        volume_pct = controller.get_volume_pct()
        draw_hud(frame, gesture if not paused else 'none',
                 volume_pct, fps, show_help)

        if paused:
            draw_flash(frame, 'PAUSED', colour=(180, 180, 180))
        elif flash_text and time.time() < flash_until:
            draw_flash(frame, flash_text)

        cv2.imshow('Hand Gesture Control', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('p'):
            paused = not paused

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print('[INFO] Exited cleanly.')


if __name__ == '__main__':
    main()
