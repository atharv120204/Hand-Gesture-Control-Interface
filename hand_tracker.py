import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

# Fixed hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

_JOINT_COLOUR = (0, 220, 0)
_BONE_COLOUR  = (0, 160, 0)


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print('[INFO] Downloading hand landmark model (~8 MB)…')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('[INFO] Model saved to', MODEL_PATH)


class HandTracker:
    """MediaPipe Tasks-based hand tracker (compatible with mediapipe >= 0.10)."""

    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12
    RING_TIP   = 16
    PINKY_TIP  = 20

    FINGER_PAIRS = [
        (8,  6),   # index
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18),  # pinky
    ]

    def __init__(self, max_hands=1, detection_confidence=0.8, tracking_confidence=0.8):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process(self, frame):
        """
        Returns (annotated_frame, landmarks_list).
        Each entry in landmarks_list:
            'points'     : list of (x_px, y_px) for all 21 landmarks
            'norm'       : list of (x, y, z) normalised coords
            'handedness' : 'Left' or 'Right'
        """
        h, w = frame.shape[:2]
        self._ts_ms += 33  # synthetic timestamp — must be monotonically increasing

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._ts_ms)

        annotated = frame.copy()
        landmarks_list = []

        if not result.hand_landmarks:
            return annotated, landmarks_list

        for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
            norm   = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            side   = handedness[0].category_name  # 'Left' or 'Right'

            _draw_hand(annotated, points)

            landmarks_list.append({
                'points': points,
                'norm': norm,
                'handedness': side,
            })

        return annotated, landmarks_list

    @staticmethod
    def fingers_up(hand: dict) -> list:
        """Returns [index_up, middle_up, ring_up, pinky_up, thumb_up]."""
        norm = hand['norm']
        side = hand['handedness']

        result = []
        for tip, pip in HandTracker.FINGER_PAIRS:
            result.append(norm[tip][1] < norm[pip][1])

        if side == 'Right':
            thumb_up = norm[HandTracker.THUMB_TIP][0] < norm[3][0]
        else:
            thumb_up = norm[HandTracker.THUMB_TIP][0] > norm[3][0]

        result.append(thumb_up)
        return result  # [index, middle, ring, pinky, thumb]

    @staticmethod
    def pinch_distance(hand: dict) -> float:
        norm = hand['norm']
        tx, ty = norm[HandTracker.THUMB_TIP][:2]
        ix, iy = norm[HandTracker.INDEX_TIP][:2]
        return float(np.hypot(tx - ix, ty - iy))

    def release(self):
        self._detector.close()


def _draw_hand(frame, points):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], _BONE_COLOUR, 2)
    for pt in points:
        cv2.circle(frame, pt, 4, _JOINT_COLOUR, -1)
