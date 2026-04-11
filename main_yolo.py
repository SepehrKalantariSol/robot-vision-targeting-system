#!/usr/bin/env python3
"""
main_yolo26_proto.py

Real-time target-driven object detection pipeline for Raspberry Pi.

This module:
- Captures live frames from a Pi Camera
- Optionally applies image calibration
- Runs YOLO inference on a lower-frame region of interest (ROI)
- Filters detections by the target class requested by an ESP32
- Selects the nearest valid object using bounding box area
- Converts object position into a navigation zone: left / middle / right
- Sends the result back to the ESP32 over a lightweight UART protocol

Protocol:
    ESP32 -> Raspberry Pi: T,s / T,b / T,g / T,r / T,n
    Raspberry Pi -> ESP32: D,l / D,m / D,r / D,n

Class mapping used by the default model:
    0: Black
    1: Green
    2: Red
    3: Silver
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

import calibration
from camera import Camera
from serial_protocol import SerialProtocol

DEFAULT_WEIGHTS_PATH = Path("weights/ball_yolo26_best.pt")

# Target command -> model class ID
TARGET_TO_CLASS_ID = {
    "b": 0,   # Black
    "g": 1,   # Green
    "r": 2,   # Red
    "s": 3,   # Silver
    "n": None,  # No active target
}


def get_bottom_roi(frame_bgr, top_crop_ratio: float) -> tuple[Any, int]:
    """
    Return the lower region of interest and its top y-offset.

    Args:
        frame_bgr: Input frame in BGR format.
        top_crop_ratio: Fraction of the top of the image to ignore.

    Returns:
        A tuple of:
        - cropped ROI image
        - y-offset of ROI in the original frame
    """
    height, _ = frame_bgr.shape[:2]
    y_offset = int(height * float(top_crop_ratio))
    y_offset = max(0, min(height - 1, y_offset))
    return frame_bgr[y_offset:, :, :], y_offset


def zone_from_x(center_x: float, frame_width: int, middle_band: float = 0.24) -> str:
    """
    Convert an x-coordinate into a navigation zone.

    Args:
        center_x: Horizontal center of the detected object.
        frame_width: Width of the ROI.
        middle_band: Fraction of the frame treated as the central zone.

    Returns:
        One of:
        - 'l' for left
        - 'm' for middle
        - 'r' for right
    """
    if frame_width <= 0:
        return "m"

    width = float(frame_width)
    frame_center = width / 2.0
    half_middle_band = (middle_band * width) / 2.0

    if (frame_center - half_middle_band) <= center_x <= (frame_center + half_middle_band):
        return "m"

    return "l" if center_x < frame_center else "r"


def draw_guides(frame, y_offset: int, roi_width: int, roi_height: int, middle_band: float) -> None:
    """
    Draw ROI and navigation guides on the frame.
    """
    frame_height, frame_width = frame.shape[:2]

    # ROI boundary
    cv2.rectangle(frame, (0, y_offset), (frame_width - 1, frame_height - 1), (255, 255, 255), 2)

    # Center line
    center_x = roi_width // 2
    cv2.line(frame, (center_x, y_offset), (center_x, y_offset + roi_height - 1), (255, 255, 255), 2)

    # Middle band boundaries
    half_band = int((middle_band * roi_width) / 2.0)
    left_boundary = center_x - half_band
    right_boundary = center_x + half_band

    cv2.line(frame, (left_boundary, y_offset), (left_boundary, y_offset + roi_height - 1), (120, 120, 255), 1)
    cv2.line(frame, (right_boundary, y_offset), (right_boundary, y_offset + roi_height - 1), (120, 120, 255), 1)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface parser.
    """
    parser = argparse.ArgumentParser(
        description="Run real-time YOLO target detection with Raspberry Pi <-> ESP32 UART communication."
    )

    # Model settings
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Path to YOLO model weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--device", default=None, help="Inference device (e.g. 'cpu', '0').")

    # Display and recording
    parser.add_argument("--show", action="store_true", help="Display the live output window.")
    parser.add_argument("--save", action="store_true", help="Save output video.")
    parser.add_argument("--out", type=Path, default=Path("runs/yolo26_proto.mp4"), help="Output video path.")

    # Calibration
    parser.add_argument("--no-calib", action="store_true", help="Disable frame calibration.")

    # ROI and directional zoning
    parser.add_argument(
        "--top-crop",
        type=float,
        default=0.25,
        help="Fraction of the top of the frame to ignore (0.25 = use bottom 75%%).",
    )
    parser.add_argument(
        "--middle-band",
        type=float,
        default=0.24,
        help="Fraction of the ROI width treated as the middle zone.",
    )

    # Serial communication
    parser.add_argument("--serial-port", default="/dev/ttyAMA0", help="UART device path.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument(
        "--no-esp",
        action="store_true",
        help="Disable UART transmission and use dry-run mode instead.",
    )

    return parser


def main() -> None:
    """
    Main runtime loop for target-driven detection and direction reporting.
    """
    args = build_arg_parser().parse_args()
    calibration_enabled = not args.no_calib

    model = YOLO(str(args.weights))

    camera = Camera()
    camera.start()

    protocol = SerialProtocol(
        port=args.serial_port,
        baud=args.baud,
        enabled=(not args.no_esp),
        min_send_interval=0.10,
    )

    video_writer = None
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = camera.read_bgr()

            if calibration_enabled:
                frame, _ = calibration.calibrate(frame)

            roi, y_offset = get_bottom_roi(frame, args.top_crop)
            roi_height, roi_width = roi.shape[:2]

            # Read the latest target request from the ESP32 (non-blocking)
            target_code = protocol.read_target()  # expected: 's', 'b', 'g', 'r', or 'n'
            target_class_id = TARGET_TO_CLASS_ID.get(target_code, None)

            direction = "n"
            status_text = f"TARGET={target_code}"

            if target_class_id is None:
                # No active target requested
                direction = "n"
                protocol.send_dir(direction, only_on_change=True)
                status_text += " | idle"
            else:
                # Run detection on ROI only
                results = model.predict(
                    source=roi,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )

                result = results[0] if results else None
                best_detection = None

                if result is not None and hasattr(result, "boxes") and result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf.item()) if hasattr(box, "conf") else 0.0
                        if confidence < args.conf:
                            continue

                        class_id = int(box.cls.item()) if hasattr(box, "cls") else -1
                        if class_id != target_class_id:
                            continue

                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        center_x = (x1 + x2) / 2.0

                        if best_detection is None or area > best_detection["area"]:
                            best_detection = {
                                "area": area,
                                "center_x": center_x,
                                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                                "confidence": confidence,
                            }

                if best_detection is None:
                    direction = "n"
                    status_text += " | no-detect"
                else:
                    direction = zone_from_x(best_detection["center_x"], roi_width, args.middle_band)
                    status_text += f" | dir={direction} area={int(best_detection['area'])}"

                    x1, y1, x2, y2 = best_detection["xyxy"]
                    cv2.rectangle(frame, (x1, y1 + y_offset), (x2, y2 + y_offset), (0, 255, 0), 2)

                protocol.send_dir(direction, only_on_change=True)

            draw_guides(frame, y_offset, roi_width, roi_height, args.middle_band)

            frame_count += 1
            elapsed = time.time() - start_time + 1e-6
            fps = frame_count / elapsed

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if args.save and video_writer is None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(args.out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (width, height),
                )
                print(f"[INFO] Saving output video to: {args.out} ({width}x{height})")

            if video_writer is not None:
                video_writer.write(frame)

            if args.show:
                cv2.imshow("YOLO Target Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        camera.stop()
        protocol.close()

        if video_writer is not None:
            video_writer.release()

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()