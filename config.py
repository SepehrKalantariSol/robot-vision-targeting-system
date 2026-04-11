"""
config.py

Central configuration file for the vision and robotics pipeline.

This module contains ONLY tunable parameters (no logic).
It allows easy adjustment of system behavior without modifying core code.

Sections:
1. Camera settings
2. Calibration (brightness + color correction)
3. Classical detector (HSV-based)
4. Visualization / debugging
"""

import numpy as np

# ============================================================
# 1) CAMERA SETTINGS
# ============================================================

# Frame resolution (width, height)
FRAME_SIZE = (1280, 720)

# Camera orientation settings
UPSIDE_DOWN = True        # Set True if camera is physically inverted
ROTATE_180 = True        # Apply 180° rotation if UPSIDE_DOWN
FLIP_H = False           # Optional horizontal flip (mirror)
FLIP_V = False           # Optional vertical flip


# ============================================================
# 2) CALIBRATION (BRIGHTNESS + COLOR CORRECTION)
# ============================================================

# --- Brightness normalization (gamma correction) ---
TARGET_BRIGHTNESS = 130   # Desired mean grayscale brightness (0–255)
GAMMA_MIN = 0.6
GAMMA_MAX = 2.2

# --- LAB color correction (neutralizing color cast) ---
# Per-axis control allows more stable tuning

WB_A_STRENGTH = 0.45      # Green ↔ Magenta correction
WB_B_STRENGTH = 0.10      # Blue ↔ Yellow correction (kept low for stability)

WB_DISABLE_B_CHANNEL = True  # Disable blue-yellow correction if unstable

WB_A_MAX_SHIFT = 10.0
WB_B_MAX_SHIFT = 4.0

WB_A_DEADZONE = 2.0
WB_B_DEADZONE = 8.0       # Larger deadzone to avoid overreaction

WB_SMOOTHING = 0.88       # Temporal smoothing factor (0–1)

# Region used for measuring color cast (center ROI)
WB_ROI_RATIO = 0.7

# --- Optional preprocessing ---
CROP_TOP_RATIO = 0.25     # Ignore top portion of image (0 disables)
BLUR_KSIZE = (0, 0)       # Gaussian blur kernel (0,0 disables)


# ============================================================
# 3) CLASSICAL DETECTOR (HSV + CONTOUR FILTERING)
# ============================================================

# Area filtering (pixel-based)
MIN_AREA = 400
MAX_AREA = 8000

# Shape filtering
MIN_CIRCULARITY = 0.70

# --- Color thresholds (HSV space) ---

# Silver detection (low saturation, high brightness)
SILVER_LOWER = np.array([0, 0, 190], dtype=np.uint8)
SILVER_UPPER = np.array([180, 60, 255], dtype=np.uint8)

# Black detection (low brightness)
BLACK_LOWER = np.array([0, 0, 0], dtype=np.uint8)
BLACK_UPPER = np.array([180, 255, 65], dtype=np.uint8)

# --- Morphological operations ---
KERNEL_OPEN = (3, 3)

SILVER_DILATE_ITERS = 2

BLACK_ERODE_ITERS = 1
BLACK_DILATE_ITERS = 2

# --- Silver highlight validation ---
# Helps distinguish reflective objects from plain white surfaces
USE_SILVER_HIGHLIGHT_CHECK = True
SILVER_HIGHLIGHT_GRAY_THRESH = 240
SILVER_HIGHLIGHT_MIN_PIXELS = 25


# ============================================================
# 4) VISUALIZATION / DEBUGGING
# ============================================================

WINDOW_MAIN = "Detection Output"

SHOW_MASKS = True
SHOW_AREA_ON_LABEL = False  # If True: displays object area in label