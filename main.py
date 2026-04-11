"""
main.py

Classical computer vision runtime pipeline.

Pipeline:
    camera -> calibration -> detection -> visualization

This script represents an earlier stage of the project, where object detection
was performed using traditional image processing methods rather than a YOLO model.
It is useful for:
- baseline testing
- debugging preprocessing and masks
- demonstrating the project's development progression
"""

from __future__ import annotations

import config as cfg
from calibration import calibrate
from camera import Camera
from detector import detect
from visual import FPSCounter, close_all, draw_detections, draw_stats, show_windows


def main() -> None:
    """
    Run the classical detection pipeline.
    """
    camera = Camera()
    camera.start()

    fps_counter = FPSCounter()

    print("Classical detection pipeline started. Press 'q' to quit.")

    try:
        while True:
            # 1) Capture frame (orientation is already handled in camera.py)
            frame_bgr = camera.read_bgr()

            # 2) Apply preprocessing / calibration
            calibrated_frame, stats = calibrate(frame_bgr)

            # 3) Run classical detector
            detections, masks = detect(calibrated_frame)

            # 4) Draw detections
            draw_detections(
                calibrated_frame,
                detections,
                show_area=getattr(cfg, "SHOW_AREA_ON_LABEL", False),
            )

            # 5) Update FPS statistics
            stats["fps"] = fps_counter.tick()

            # 6) Draw diagnostic overlays
            draw_stats(calibrated_frame, stats)

            # 7) Show output windows
            show_masks = masks if getattr(cfg, "SHOW_MASKS", True) else None
            should_quit = show_windows(
                getattr(cfg, "WINDOW_MAIN", "Detection Output"),
                calibrated_frame,
                masks=show_masks,
            )

            if should_quit:
                break

    finally:
        camera.stop()
        close_all()
        print("Classical detection pipeline stopped.")


if __name__ == "__main__":
    main()