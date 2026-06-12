import time
import pyautogui
import numpy as np
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import config

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class ActionController:
    """Translates recognised gestures into OS-level actions."""

    def __init__(self, frame_w: int, frame_h: int):
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._screen_w, self._screen_h = pyautogui.size()

        # Smoothing buffer for cursor movement
        self._cursor_buf: deque[tuple[float, float]] = deque(maxlen=config.MOUSE_SMOOTHING)

        # Click debouncing
        self._last_click_time = 0.0

        # Volume interface
        self._volume = self._init_volume()
        self._vol_min, self._vol_max, _ = self._volume.GetVolumeRange()

    # ------------------------------------------------------------------
    # Public action methods
    # ------------------------------------------------------------------

    def move_cursor(self, tip_px: tuple[int, int]):
        """Smoothly move the cursor to the position indicated by the index tip."""
        # Flip x so movement feels natural (mirror image)
        raw_x = np.interp(tip_px[0], [0, self._frame_w], [self._screen_w, 0])
        raw_y = np.interp(tip_px[1], [0, self._frame_h], [0, self._screen_h])

        self._cursor_buf.append((raw_x, raw_y))
        smooth_x = int(np.mean([p[0] for p in self._cursor_buf]))
        smooth_y = int(np.mean([p[1] for p in self._cursor_buf]))

        cur_x, cur_y = pyautogui.position()
        if abs(smooth_x - cur_x) > config.MOVE_DEADZONE or abs(smooth_y - cur_y) > config.MOVE_DEADZONE:
            pyautogui.moveTo(smooth_x, smooth_y)

    def left_click(self):
        if self._click_allowed():
            pyautogui.click(button='left')
            self._last_click_time = time.time()

    def right_click(self):
        if self._click_allowed():
            pyautogui.click(button='right')
            self._last_click_time = time.time()

    def scroll(self, direction: str, delta: int):
        ticks = max(1, delta // config.SCROLL_SENSITIVITY)
        amount = ticks if direction == 'up' else -ticks
        pyautogui.scroll(amount)

    def volume_up(self):
        current = self._volume.GetMasterVolumeLevel()
        new_vol = min(self._vol_max, current + 1.5)
        self._volume.SetMasterVolumeLevel(new_vol, None)

    def volume_down(self):
        current = self._volume.GetMasterVolumeLevel()
        new_vol = max(self._vol_min, current - 1.5)
        self._volume.SetMasterVolumeLevel(new_vol, None)

    def screenshot(self):
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = f'screenshot_{ts}.png'
        pyautogui.screenshot(path)
        return path

    def get_volume_pct(self) -> int:
        current = self._volume.GetMasterVolumeLevel()
        pct = (current - self._vol_min) / (self._vol_max - self._vol_min) * 100
        return int(pct)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _click_allowed(self) -> bool:
        return (time.time() - self._last_click_time) > config.CLICK_COOLDOWN

    @staticmethod
    def _init_volume() -> IAudioEndpointVolume:
        device = AudioUtilities.GetSpeakers()
        # pycaw >= 20240210 wraps IMMDevice in AudioDevice; unwrap to get COM object
        imm_device = getattr(device, '_dev', device)
        interface = imm_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
