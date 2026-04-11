"""
camera.py

Camera interface module for Raspberry Pi using Picamera2.

Responsibilities:
- Initialize and configure the camera
- Capture frames in RGB or BGR format
- Apply orientation correction (rotation and flipping)

The module supports optional configuration via `config.py`,
allowing hardware-specific settings to be adjusted without
modifying the core logic.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

# Optional configuration import
try:
    import config as cfg
except Exception:
    cfg = None


def _get_config(name: str, default):
    """
    Safely read a configuration value from config.py.
    Falls back to a default if not defined.
    """
    return getattr(cfg, name, default) if cfg is not None else default


# =========================
# Configuration parameters
# =========================

FRAME_SIZE: Tuple[int, int] = _get_config("FRAME_SIZE", (1280, 720))  # (width, height)

UPSIDE_DOWN: bool = _get_config("UPSIDE_DOWN", True)
ROTATE_180: bool = _get_config("ROTATE_180", True)

FLIP_H: bool = _get_config("FLIP_H", False)
FLIP_V: bool = _get_config("FLIP_V", False)


class Camera:
    """
    Wrapper class for PiCamera2 capture.

    Provides:
    - start/stop control
    - RGB frame capture
    - BGR frame conversion for OpenCV

    Designed to isolate hardware-specific camera handling from
    the rest of the vision pipeline.
    """

    def __init__(self) -> None:
        self._camera = Picamera2()

        camera_config = self._camera.create_still_configuration(
            main={"size": FRAME_SIZE}
        )
        self._camera.configure(camera_config)

    def start(self) -> None:
        """
        Start the camera stream.
        """
        self._camera.start()

    def stop(self) -> None:
        """
        Stop the camera stream.
        """
        self._camera.stop()

    def _apply_orientation(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply rotation and flipping to correct camera orientation.
        """
        # Rotate 180 degrees if mounted upside down
        if UPSIDE_DOWN and ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Optional flips
        if FLIP_H:
            frame = cv2.flip(frame, 1)
        if FLIP_V:
            frame = cv2.flip(frame, 0)

        return frame

    def read_rgb(self) -> np.ndarray:
        """
        Capture a frame in RGB format.

        Returns:
            np.ndarray: RGB image (uint8)
        """
        frame_rgb = self._camera.capture_array("main")

        # Apply orientation correction BEFORE any cropping/processing
        frame_rgb = self._apply_orientation(frame_rgb)

        return frame_rgb

    def read_bgr(self) -> np.ndarray:
        """
        Capture a frame in BGR format (for OpenCV pipelines).

        Returns:
            np.ndarray: BGR image (uint8)
        """
        frame_rgb = self.read_rgb()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)