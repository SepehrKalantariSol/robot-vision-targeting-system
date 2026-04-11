"""
visual.py

Visualization and debugging utilities for the vision pipeline.

Responsibilities:
- Draw detected objects and labels
- Display diagnostic statistics on frames
- Show main and mask windows
- Provide a lightweight FPS helper

This module contains no detection, calibration, or camera logic.
It is designed to keep presentation/debugging concerns separate from
core processing.
"""

from __future__ import annotations

import time

import cv2


# ==========================================
# Display colors (BGR format)
# ==========================================
COLOR_SILVER = (0, 255, 255)   # yellow
COLOR_BLACK = (0, 255, 0)      # green
COLOR_TEXT_SHADOW = (0, 0, 0)  # black


def get_type_color(ball_type: str) -> tuple[int, int, int]:
    """
    Return the display color for a detected object type.
    """
    object_type = (ball_type or "").lower()

    if object_type == "silver":
        return COLOR_SILVER
    if object_type == "black":
        return COLOR_BLACK

    return (255, 255, 255)  # default: white


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """
    Clamp a point so it remains within the image bounds.
    """
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def assign_ids(detections: list[dict]) -> list[dict]:
    """
    Assign simple per-frame IDs to detections.

    ID convention:
    - Silver objects: S1, S2, ...
    - Black objects:  B1, B2, ...

    Objects are ordered left-to-right within each class.

    Args:
        detections: List of detection dictionaries. Each detection is expected
            to include at least:
            {
                "type": "silver" or "black",
                "x": int,
                "y": int,
                "radius": int or float
            }

    Returns:
        List of detections with added "id" fields.
    """
    silver_detections = [d for d in detections if d.get("type", "").lower() == "silver"]
    black_detections = [d for d in detections if d.get("type", "").lower() == "black"]

    silver_detections.sort(key=lambda d: d.get("x", 0))
    black_detections.sort(key=lambda d: d.get("x", 0))

    for index, detection in enumerate(silver_detections, start=1):
        detection["id"] = f"S{index}"

    for index, detection in enumerate(black_detections, start=1):
        detection["id"] = f"B{index}"

    return silver_detections + black_detections


def draw_detections(frame_bgr, detections: list[dict], show_area: bool = False) -> None:
    """
    Draw detected objects on a frame.

    Each detection may contain:
        {
            "type": "silver" or "black",
            "x": center_x,
            "y": center_y,
            "radius": radius,
            "area": area
        }

    Draws:
    - enclosing circle
    - center point
    - object label
    """
    if frame_bgr is None:
        return

    height, width = frame_bgr.shape[:2]
    detections_with_ids = assign_ids([dict(d) for d in detections])  # copy before modification

    for detection in detections_with_ids:
        ball_type = detection.get("type", "unknown")
        color = get_type_color(ball_type)

        x = int(detection.get("x", 0))
        y = int(detection.get("y", 0))
        radius = int(round(float(detection.get("radius", 0)))) if detection.get("radius") is not None else 0

        x, y = clamp_point(x, y, width, height)
        radius = max(2, radius)

        # Circle outline
        cv2.circle(frame_bgr, (x, y), radius, color, 2)

        # Center point
        cv2.circle(frame_bgr, (x, y), 5, color, -1)

        # Label
        label = detection.get("id", ball_type[:1].upper())
        if show_area and "area" in detection and detection["area"] is not None:
            label += f" A={int(detection['area'])}"

        text_x, text_y = clamp_point(x + 10, y - 10, width, height)

        cv2.putText(
            frame_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT_SHADOW,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_stats(frame_bgr, stats: dict) -> None:
    """
    Draw diagnostic statistics on the top-left of the frame.

    Supported keys may include:
    - fps
    - meanY
    - gamma
    - g_ratio
    - lab_shift
    """
    if frame_bgr is None or not stats:
        return

    lines = []
    if "fps" in stats:
        lines.append(f"FPS: {stats['fps']:.1f}")
    if "meanY" in stats:
        lines.append(f"Y mean: {stats['meanY']:.1f}")
    if "gamma" in stats:
        lines.append(f"Gamma: {stats['gamma']:.2f}")
    if "g_ratio" in stats:
        lines.append(f"Green cast ratio: {stats['g_ratio']:.2f}")
    if "lab_shift" in stats:
        lines.append(f"LAB a shift: {stats['lab_shift']:.2f}")

    x, y = 10, 25
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT_SHADOW,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22


def show_windows(
    main_name: str,
    frame_bgr,
    masks: dict | None = None,
    wait_ms: int = 1,
) -> bool:
    """
    Show the main output window and optional mask/debug windows.

    Args:
        main_name: Window title for the main frame
        frame_bgr: Main image to display
        masks: Optional dictionary of additional debug windows
        wait_ms: Delay passed to cv2.waitKey()

    Returns:
        True if the user pressed 'q', otherwise False
    """
    cv2.imshow(main_name, frame_bgr)

    if masks:
        for window_name, image in masks.items():
            cv2.imshow(window_name, image)

    key = cv2.waitKey(wait_ms) & 0xFF
    return key == ord("q")


class FPSCounter:
    """
    Lightweight moving-window FPS estimator.
    """

    def __init__(self, avg_over: int = 15) -> None:
        self.avg_over = max(1, int(avg_over))
        self.timestamps: list[float] = []

    def tick(self) -> float:
        """
        Record a frame timestamp and return the estimated FPS.
        """
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) > self.avg_over:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        delta_time = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / delta_time if delta_time > 0 else 0.0


def close_all() -> None:
    """
    Close all OpenCV windows.
    """
    cv2.destroyAllWindows()