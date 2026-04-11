"""
detector.py

Classical computer vision ball detector.

This module performs object detection using traditional image processing methods:
- HSV thresholding
- Morphological cleanup
- Contour extraction
- Circularity filtering
- Optional highlight validation for reflective objects

It was used as an earlier development stage before the YOLO-based detector and
remains useful as:
- a lightweight baseline detector
- a debugging tool
- a demonstration of lower-level computer vision understanding
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
# Detection thresholds and defaults
# ==========================================

MIN_AREA = _get_config("MIN_AREA", 400)
MAX_AREA = _get_config("MAX_AREA", 8000)
MIN_CIRCULARITY = _get_config("MIN_CIRCULARITY", 0.70)

# Silver: low saturation + high brightness
SILVER_LOWER = _get_config("SILVER_LOWER", np.array([0, 0, 190], dtype=np.uint8))
SILVER_UPPER = _get_config("SILVER_UPPER", np.array([180, 60, 255], dtype=np.uint8))

# Black: low brightness
BLACK_LOWER = _get_config("BLACK_LOWER", np.array([0, 0, 0], dtype=np.uint8))
BLACK_UPPER = _get_config("BLACK_UPPER", np.array([180, 255, 65], dtype=np.uint8))

# Morphology
KERNEL_OPEN = _get_config("KERNEL_OPEN", (3, 3))
SILVER_DILATE_ITERS = _get_config("SILVER_DILATE_ITERS", 2)
BLACK_ERODE_ITERS = _get_config("BLACK_ERODE_ITERS", 1)
BLACK_DILATE_ITERS = _get_config("BLACK_DILATE_ITERS", 2)

# Crack healing to bridge thin mask gaps caused by reflections or shadows
USE_CRACK_HEALING = _get_config("USE_CRACK_HEALING", True)
CLOSE_KERNEL = _get_config("CLOSE_KERNEL", (7, 7))
CLOSE_ITERS_SILVER = _get_config("CLOSE_ITERS_SILVER", 1)
CLOSE_ITERS_BLACK = _get_config("CLOSE_ITERS_BLACK", 1)

# Optional filling of internal mask holes
FILL_HOLES = _get_config("FILL_HOLES", False)

# Optional specular highlight validation for silver objects
USE_SILVER_HIGHLIGHT_CHECK = _get_config("USE_SILVER_HIGHLIGHT_CHECK", True)
SILVER_HIGHLIGHT_GRAY_THRESH = _get_config("SILVER_HIGHLIGHT_GRAY_THRESH", 240)
SILVER_HIGHLIGHT_MIN_PIXELS = _get_config("SILVER_HIGHLIGHT_MIN_PIXELS", 25)


# ==========================================
# Geometry helpers
# ==========================================

def circularity(contour) -> float:
    """
    Compute contour circularity.

    A perfect circle has circularity close to 1.0.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return 0.0

    return float(4.0 * np.pi * area / (perimeter * perimeter))


def centroid(contour) -> tuple[int, int] | None:
    """
    Compute contour centroid.

    Returns:
        (cx, cy) if valid, otherwise None
    """
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def min_enclosing_circle(contour) -> tuple[int, int, int]:
    """
    Compute the minimum enclosing circle for a contour.

    Returns:
        (x, y, radius)
    """
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return int(round(x)), int(round(y)), int(round(radius))


def silver_has_highlight(gray: np.ndarray, contour) -> bool:
    """
    Check whether a candidate silver object contains enough very bright pixels.

    This helps distinguish reflective silver balls from plain white or light-colored
    regions that may satisfy the HSV thresholds but do not exhibit specular highlights.
    """
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    bright_mask = cv2.inRange(gray, SILVER_HIGHLIGHT_GRAY_THRESH, 255)
    bright_inside = cv2.bitwise_and(bright_mask, bright_mask, mask=contour_mask)

    bright_pixel_count = int(cv2.countNonZero(bright_inside))
    return bright_pixel_count >= SILVER_HIGHLIGHT_MIN_PIXELS


# ==========================================
# Mask post-processing helpers
# ==========================================

def heal_cracks(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Bridge thin internal gaps in a mask using morphological closing.

    This is useful when reflections, shadows, or lighting streaks split the
    segmented region of a ball into disconnected parts.
    """
    if not USE_CRACK_HEALING or iterations <= 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes by redrawing external contours as solid regions.

    Useful when segmentation produces ring-like objects or interior gaps.
    """
    if not FILL_HOLES:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask


# ==========================================
# Core detection
# ==========================================

def detect(frame_bgr: np.ndarray) -> tuple[list[dict], dict]:
    """
    Detect black and silver ball candidates using classical image processing.

    Args:
        frame_bgr: Input frame in BGR format

    Returns:
        A tuple of:
        - detections: list of dictionaries with fields:
            {
                "type": "silver" or "black",
                "x": center_x,
                "y": center_y,
                "radius": radius,
                "area": area
            }
        - masks: dictionary of debug masks:
            {
                "Silver Mask": ...,
                "Black Mask": ...
            }
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    kernel_open = np.ones(KERNEL_OPEN, np.uint8)

    # ------------------------------------------
    # Silver mask
    # ------------------------------------------
    silver_mask = cv2.inRange(hsv, SILVER_LOWER, SILVER_UPPER)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel_open)
    silver_mask = cv2.dilate(silver_mask, None, iterations=int(SILVER_DILATE_ITERS))
    silver_mask = heal_cracks(silver_mask, CLOSE_ITERS_SILVER)
    silver_mask = fill_holes(silver_mask)

    # ------------------------------------------
    # Black mask
    # ------------------------------------------
    black_mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open)
    black_mask = cv2.erode(black_mask, None, iterations=int(BLACK_ERODE_ITERS))
    black_mask = cv2.dilate(black_mask, None, iterations=int(BLACK_DILATE_ITERS))
    black_mask = heal_cracks(black_mask, CLOSE_ITERS_BLACK)
    black_mask = fill_holes(black_mask)

    detections: list[dict] = []

    # ------------------------------------------
    # Silver contours
    # ------------------------------------------
    silver_contours, _ = cv2.findContours(silver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in silver_contours:
        area = float(cv2.contourArea(contour))
        if not (MIN_AREA < area < MAX_AREA):
            continue

        if circularity(contour) < MIN_CIRCULARITY:
            continue

        if USE_SILVER_HIGHLIGHT_CHECK and not silver_has_highlight(gray, contour):
            continue

        center = centroid(contour)
        if center is None:
            continue

        cx, cy = center
        _, _, radius = min_enclosing_circle(contour)

        detections.append(
            {
                "type": "silver",
                "x": cx,
                "y": cy,
                "radius": radius,
                "area": int(area),
            }
        )

    # ------------------------------------------
    # Black contours
    # ------------------------------------------
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in black_contours:
        area = float(cv2.contourArea(contour))
        if not (MIN_AREA < area < MAX_AREA):
            continue

        if circularity(contour) < MIN_CIRCULARITY:
            continue

        center = centroid(contour)
        if center is None:
            continue

        cx, cy = center
        _, _, radius = min_enclosing_circle(contour)

        detections.append(
            {
                "type": "black",
                "x": cx,
                "y": cy,
                "radius": radius,
                "area": int(area),
            }
        )

    debug_masks = {
        "Silver Mask": silver_mask,
        "Black Mask": black_mask,
    }

    return detections, debug_masks