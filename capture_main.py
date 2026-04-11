"""
capture_main.py

Interactive dataset capture utility for the Raspberry Pi vision system.

Features:
- Live camera preview
- Real-time calibrated frame display
- Keyboard-triggered image capture
- Optional saving of both calibrated and raw frames

Controls:
- Press 'c' to save the calibrated frame
- Press 'r' to toggle raw frame saving
- Press 'q' to quit
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from camera import Camera
from calibration import calibrate

CAPTURE_DIR = Path("capture")
WINDOW_NAME = "Dataset Capture (c=save, r=toggle raw, q=quit)"


def ensure_directory(path: Path) -> None:
    """
    Create the capture directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def generate_timestamped_name(prefix: str = "img", ext: str = "jpg") -> str:
    """
    Generate a timestamped filename.

    Example:
        cal_20260216_153012_123.jpg
    """
    timestamp = time.time()
    milliseconds = int((timestamp - int(timestamp)) * 1000)
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S_{milliseconds:03d}.{ext}")


def draw_hud(frame_bgr, raw_mode: bool, last_saved: str) -> None:
    """
    Draw a simple heads-up display on the preview frame.
    """
    lines = [
        f"Save RAW frames: {'ON' if raw_mode else 'OFF'} (press r)",
        "Press c to save | q to quit",
    ]

    if last_saved:
        lines.append(f"Last saved: {last_saved}")

    x, y = 10, 25
    for line in lines:
        # Shadow for readability
        cv2.putText(
            frame_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        # Foreground text
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


def main() -> None:
    """
    Run the interactive dataset capture loop.
    """
    ensure_directory(CAPTURE_DIR)

    camera = Camera()
    camera.start()

    save_raw_frames = False
    last_saved_filename = ""

    print("Dataset capture mode started.")
    print("  - Press 'c' to save the calibrated image to ./capture/")
    print("  - Press 'r' to toggle saving the raw frame as well")
    print("  - Press 'q' to quit")

    try:
        while True:
            # 1) Read raw frame (orientation-corrected in camera.py)
            raw_frame = camera.read_bgr()

            # 2) Apply calibration pipeline
            calibrated_frame, _stats = calibrate(raw_frame)

            # 3) Create preview with overlay
            preview_frame = calibrated_frame.copy()
            draw_hud(preview_frame, save_raw_frames, last_saved_filename)

            # 4) Show live calibrated preview
            cv2.imshow(WINDOW_NAME, preview_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                save_raw_frames = not save_raw_frames
                print(f"Raw frame saving: {'ON' if save_raw_frames else 'OFF'}")

            elif key == ord("c"):
                calibrated_name = generate_timestamped_name(prefix="cal", ext="jpg")
                calibrated_path = CAPTURE_DIR / calibrated_name
                cv2.imwrite(str(calibrated_path), calibrated_frame)

                print(f"[INFO] Saved calibrated frame: {calibrated_path}")
                last_saved_filename = calibrated_name

                if save_raw_frames:
                    raw_name = calibrated_name.replace("cal_", "raw_", 1)
                    raw_path = CAPTURE_DIR / raw_name
                    cv2.imwrite(str(raw_path), raw_frame)
                    print(f"[INFO] Saved raw frame:        {raw_path}")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Dataset capture mode stopped.")


if __name__ == "__main__":
    main()