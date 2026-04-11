"""
calibration.py

Image preprocessing and calibration utilities for the vision pipeline.

Responsibilities:
- Optional frame cropping
- Optional Gaussian blur
- Automatic brightness normalization using gamma correction
- LAB-based color cast correction with smoothing

This module is intentionally independent of camera handling, detection,
and visualization so that preprocessing can be tested and tuned in isolation.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Optional configuration import
try:
    import config as cfg
except Exception:
    cfg = None


def _get_config(name: str, default: Any) -> Any:
    """
    Safely read a configuration value from config.py.
    Falls back to the provided default if not defined.
    """
    return getattr(cfg, name, default) if cfg is not None else default


# ==========================================
# Internal state for temporal smoothing
# ==========================================
_prev_a_shift = 0.0
_prev_b_shift = 0.0


# ==========================================
# Calibration parameters
# ==========================================

# Brightness normalization
TARGET_BRIGHTNESS = _get_config("TARGET_BRIGHTNESS", 130)
GAMMA_MIN = _get_config("GAMMA_MIN", 0.6)
GAMMA_MAX = _get_config("GAMMA_MAX", 2.2)

# White balance / LAB neutralization
WB_DEADZONE = _get_config("WB_DEADZONE", 2.0)
WB_SMOOTHING = _get_config("WB_SMOOTHING", 0.85)
WB_ROI_RATIO = _get_config("WB_ROI_RATIO", 0.7)

WB_STRENGTH = _get_config("WB_STRENGTH", 0.6)
WB_MAX_SHIFT = _get_config("WB_MAX_SHIFT", 18.0)

# Optional preprocessing
CROP_TOP_RATIO = _get_config("CROP_TOP_RATIO", 0.25)
BLUR_KSIZE = _get_config("BLUR_KSIZE", (5, 5))


def crop_top(frame_bgr: np.ndarray, top_ratio: float) -> np.ndarray:
    """
    Crop the top portion of the frame.

    Args:
        frame_bgr: Input image in BGR format.
        top_ratio: Fraction of the top of the image to remove.

    Returns:
        Cropped BGR image.
    """
    if top_ratio <= 0:
        return frame_bgr

    height = frame_bgr.shape[0]
    y_start = int(height * top_ratio)
    return frame_bgr[y_start:, :, :]


def apply_blur(frame_bgr: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.

    Args:
        frame_bgr: Input image in BGR format.
        ksize: Blur kernel size. (0, 0) disables blur.

    Returns:
        Blurred BGR image.
    """
    if not ksize or ksize == (0, 0):
        return frame_bgr

    kernel_x = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
    kernel_y = ksize[1] if ksize[1] % 2 == 1 else ksize[1] + 1
    return cv2.GaussianBlur(frame_bgr, (kernel_x, kernel_y), 0)


def lab_neutralize_ab(frame_bgr: np.ndarray, strength: float) -> tuple[np.ndarray, float, float, float, float]:
    """
    Apply LAB color correction by shifting the a and b channels toward neutral.

    Design features:
    - Uses a central ROI to reduce sensitivity to colored edges or walls
    - Applies deadzones to avoid reacting to small fluctuations
    - Uses exponential smoothing to reduce flicker
    - Supports optional disabling of blue-yellow correction

    Args:
        frame_bgr: Input image in BGR format.
        strength: Base correction strength.

    Returns:
        Tuple containing:
        - corrected frame
        - measured a-channel mean
        - measured b-channel mean
        - applied smoothed a shift
        - applied smoothed b shift
    """
    global _prev_a_shift, _prev_b_shift

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Measure color cast in a central region to reduce environmental bias
    height, width = a_channel.shape
    roi_ratio = float(np.clip(WB_ROI_RATIO, 0.2, 1.0))
    roi_height, roi_width = int(height * roi_ratio), int(width * roi_ratio)
    y_start = (height - roi_height) // 2
    x_start = (width - roi_width) // 2

    a_roi = a_channel[y_start:y_start + roi_height, x_start:x_start + roi_width]
    b_roi = b_channel[y_start:y_start + roi_height, x_start:x_start + roi_width]

    a_mean = float(a_roi.mean())
    b_mean = float(b_roi.mean())

    # Per-axis overrides
    wb_a_strength = _get_config("WB_A_STRENGTH", strength)
    wb_b_strength = _get_config("WB_B_STRENGTH", strength)

    wb_a_max_shift = _get_config("WB_A_MAX_SHIFT", WB_MAX_SHIFT)
    wb_b_max_shift = _get_config("WB_B_MAX_SHIFT", WB_MAX_SHIFT)

    wb_a_deadzone = _get_config("WB_A_DEADZONE", WB_DEADZONE)
    wb_b_deadzone = _get_config("WB_B_DEADZONE", WB_DEADZONE)

    # LAB neutral point is approximately 128 for both a and b channels
    delta_a = 128.0 - a_mean
    delta_b = 128.0 - b_mean

    # Optionally disable blue-yellow correction if it is too reactive
    if _get_config("WB_DISABLE_B_CHANNEL", False):
        delta_b = 0.0

    if abs(delta_a) < wb_a_deadzone:
        delta_a = 0.0
    if abs(delta_b) < wb_b_deadzone:
        delta_b = 0.0

    a_shift = float(np.clip(delta_a * wb_a_strength, -wb_a_max_shift, wb_a_max_shift))
    b_shift = float(np.clip(delta_b * wb_b_strength, -wb_b_max_shift, wb_b_max_shift))

    # Exponential smoothing to reduce frame-to-frame instability
    smoothing = float(np.clip(WB_SMOOTHING, 0.0, 0.98))
    a_shift_smoothed = smoothing * _prev_a_shift + (1.0 - smoothing) * a_shift
    b_shift_smoothed = smoothing * _prev_b_shift + (1.0 - smoothing) * b_shift

    _prev_a_shift = a_shift_smoothed
    _prev_b_shift = b_shift_smoothed

    a_channel = np.clip(a_channel + a_shift_smoothed, 0, 255)
    b_channel = np.clip(b_channel + b_shift_smoothed, 0, 255)

    lab[:, :, 1] = a_channel
    lab[:, :, 2] = b_channel

    corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return corrected, a_mean, b_mean, a_shift_smoothed, b_shift_smoothed


def auto_gamma(frame_bgr: np.ndarray, target: float) -> tuple[np.ndarray, float, float]:
    """
    Normalize image brightness using gamma correction.

    Args:
        frame_bgr: Input image in BGR format.
        target: Desired average brightness level in grayscale space.

    Returns:
        Tuple containing:
        - brightness-corrected frame
        - original grayscale mean brightness
        - gamma value used
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    mean_brightness = max(1.0, mean_brightness)

    mean_normalized = mean_brightness / 255.0
    target_normalized = float(np.clip(target / 255.0, 1e-3, 0.999))

    gamma = np.log(target_normalized) / np.log(mean_normalized)
    gamma = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

    inverse_gamma = 1.0 / gamma
    lookup_table = ((np.arange(256) / 255.0) ** inverse_gamma * 255.0).astype(np.uint8)
    corrected = cv2.LUT(frame_bgr, lookup_table)

    return corrected, mean_brightness, gamma


def calibrate(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Run the full calibration pipeline.

    Processing order:
    1. Optional top crop
    2. Optional Gaussian blur
    3. Automatic brightness normalization
    4. LAB-based color correction

    Args:
        frame_bgr: Input image in BGR format.

    Returns:
        Tuple containing:
        - calibrated BGR frame
        - dictionary of calibration statistics
    """
    stats: dict = {}

    # 1) Optional crop
    frame = crop_top(frame_bgr, CROP_TOP_RATIO)

    # 2) Optional blur
    frame = apply_blur(frame, BLUR_KSIZE)

    # 3) Brightness normalization
    frame, mean_brightness, gamma = auto_gamma(frame, TARGET_BRIGHTNESS)
    stats["mean_brightness"] = mean_brightness
    stats["gamma"] = gamma

    # 4) LAB color correction
    frame, a_mean, b_mean, a_shift, b_shift = lab_neutralize_ab(frame, WB_STRENGTH)
    stats["lab_a_mean"] = a_mean
    stats["lab_b_mean"] = b_mean
    stats["lab_a_shift"] = a_shift
    stats["lab_b_shift"] = b_shift

    # Optional diagnostic RGB channel means after correction
    blue_channel, green_channel, red_channel = cv2.split(frame)
    stats["red_mean"] = float(red_channel.mean())
    stats["green_mean"] = float(green_channel.mean())
    stats["blue_mean"] = float(blue_channel.mean())

    return frame, stats