"""
esp_serial.py

Legacy serial communication module using compact numeric byte codes.

This module was an earlier communication approach used before the current
text-based UART protocol was introduced. It encodes object class and direction
into a single byte for transmission to the ESP32.

Encoding scheme:
    Black:   0-2
    Silver:  3-5
    Green:   6-8
    Red:     9-11
    None:    12

Zone mapping:
    left   -> 0
    right  -> 1
    middle -> 2

Example:
    silver + middle -> 3 + 2 = 5

This module is retained for documentation and development-history purposes.
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import serial  # type: ignore
except Exception:
    serial = None


COLOR_BASE = {
    "black": 0,
    "silver": 3,
    "green": 6,
    "red": 9,
}

ZONE_OFFSET = {
    "left": 0,
    "right": 1,
    "middle": 2,
}

NO_DETECT = 12


def zone_from_x(center_x: float, frame_width: int, middle_band: float = 0.24) -> str:
    """
    Convert an x-coordinate into a directional zone.

    Args:
        center_x: Horizontal center of the target.
        frame_width: Width of the reference frame or ROI.
        middle_band: Fraction of the frame treated as the middle region.

    Returns:
        One of: 'left', 'middle', or 'right'
    """
    if frame_width <= 0:
        return "middle"

    width = float(frame_width)
    frame_center = width / 2.0
    half_band = (middle_band * width) / 2.0

    if (frame_center - half_band) <= center_x <= (frame_center + half_band):
        return "middle"

    return "left" if center_x < frame_center else "right"


def encode_code(color: str | None, zone: str | None) -> int:
    """
    Encode a color + zone pair into a compact integer code.

    Returns:
        Encoded byte value, or NO_DETECT if invalid
    """
    if not color or not zone:
        return NO_DETECT

    color_key = color.lower().strip()
    zone_key = zone.lower().strip()

    if color_key not in COLOR_BASE or zone_key not in ZONE_OFFSET:
        return NO_DETECT

    return COLOR_BASE[color_key] + ZONE_OFFSET[zone_key]


def decode_code(code: int) -> str:
    """
    Convert an encoded byte value into a human-readable description.

    Returns:
        A descriptive label such as 'SILVER_MIDDLE' or 'NO_DETECT'
    """
    code = int(code)

    if code == NO_DETECT:
        return "NO_DETECT"

    if 0 <= code <= 2:
        color = "black"
        base = 0
    elif 3 <= code <= 5:
        color = "silver"
        base = 3
    elif 6 <= code <= 8:
        color = "green"
        base = 6
    elif 9 <= code <= 11:
        color = "red"
        base = 9
    else:
        return "UNKNOWN"

    zone_index = code - base
    zone = {0: "left", 1: "right", 2: "middle"}.get(zone_index, "unknown")
    return f"{color.upper()}_{zone.upper()}"


class ESPSender:
    """
    Legacy serial sender for single-byte communication with an ESP32.

    If the serial port cannot be opened, the class switches to dry-run mode,
    where outgoing messages are printed instead of transmitted.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = 115200,
        min_interval_s: float = 0.10,
        enabled: bool = True,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.min_interval_s = float(min_interval_s)
        self.enabled = bool(enabled)

        self._last_send_time = 0.0
        self._last_code: Optional[int] = None

        self.serial_conn: Optional[object] = None
        self.print_only = False

        if not self.enabled or serial is None:
            self.print_only = True
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(1.5)  # Allow time for ESP32 reset on serial open
        except Exception:
            self.serial_conn = None
            self.print_only = True

    def close(self) -> None:
        """
        Close the serial connection if open.
        """
        try:
            if self.serial_conn is not None:
                self.serial_conn.close()
        except Exception:
            pass

    def send(self, code: int, force: bool = False, only_on_change: bool = True) -> bool:
        """
        Send a single encoded byte to the ESP32.

        Args:
            code: Integer code to transmit
            force: If True, bypass rate limiting and repeated-send suppression
            only_on_change: If True, suppress repeated identical codes

        Returns:
            True if the code was sent or printed, False if skipped
        """
        code = int(code) & 0xFF
        now = time.time()

        if only_on_change and self._last_code == code and not force:
            return False

        if (now - self._last_send_time) < self.min_interval_s and not force:
            return False

        description = decode_code(code)

        if self.print_only or self.serial_conn is None:
            print(f"[ESP-SEND-DRY] code={code} ({description})")
        else:
            try:
                self.serial_conn.write(bytes([code]))
                self.serial_conn.flush()
            except Exception:
                self.print_only = True
                print(f"[ESP-SEND-DRY] code={code} ({description})")

        self._last_send_time = now
        self._last_code = code
        return True