"""
serial_protocol.py

Lightweight UART protocol for Raspberry Pi <-> ESP32 communication.

Protocol:
    ESP32 -> Raspberry Pi: T,<target>\\n
    Raspberry Pi -> ESP32: D,<direction>\\n

Targets:
    s = silver
    b = black
    g = green
    r = red
    n = none / no active search target

Directions:
    l = left
    m = middle
    r = right
    n = none / no valid detection

Features:
- Non-blocking serial reads
- Line-based message parsing
- Safe fallback to dry-run mode if serial is unavailable
- Optional rate limiting and send-on-change behavior
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import serial  # type: ignore
except Exception:
    serial = None


VALID_TARGETS = {"s", "b", "g", "r", "n"}
VALID_DIRECTIONS = {"l", "m", "r", "n"}


class SerialProtocol:
    """
    Two-way serial protocol wrapper for ESP32 <-> Raspberry Pi communication.

    If serial is disabled or the port cannot be opened, the class switches to
    dry-run mode, where outgoing messages are printed instead of transmitted.

    Behavior:
    - `read_target()` returns the latest valid target command received
    - `send_direction()` sends or prints a `D,<direction>` message
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = 115200,
        enabled: bool = True,
        min_send_interval: float = 0.10,
    ) -> None:
        self.enabled = bool(enabled)
        self.port = port
        self.baud = int(baud)
        self.min_send_interval = float(min_send_interval)

        self.serial_conn: Optional[object] = None
        self.print_only = False

        self.target = "n"
        self._rx_buffer = b""

        self._last_send_time = 0.0
        self._last_sent_message: Optional[str] = None

        if not self.enabled or serial is None:
            self.print_only = True
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0)
            time.sleep(1.5)  # Allow time for ESP32 reset on serial open
        except Exception:
            self.serial_conn = None
            self.print_only = True

    def close(self) -> None:
        """
        Close the serial connection if it is open.
        """
        try:
            if self.serial_conn is not None:
                self.serial_conn.close()
        except Exception:
            pass

    def _process_line(self, line: str) -> None:
        """
        Parse a single incoming protocol line.

        Expected format:
            T,<target>

        Invalid messages are ignored safely.
        """
        line = line.strip()
        if not line:
            return

        parts = line.split(",")
        if len(parts) != 2:
            return

        prefix = parts[0].strip()
        value = parts[1].strip().lower()

        if prefix != "T":
            return

        if value in VALID_TARGETS:
            self.target = value

    def read_target(self) -> str:
        """
        Read and process any available incoming serial data.

        This method is non-blocking:
        - reads all currently available bytes
        - processes complete newline-terminated messages
        - returns the most recent valid target command

        Returns:
            One of: 's', 'b', 'g', 'r', or 'n'
        """
        if self.print_only or self.serial_conn is None:
            return self.target

        try:
            incoming = self.serial_conn.read(256)
            if incoming:
                self._rx_buffer += incoming

                while b"\n" in self._rx_buffer:
                    raw_line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                    try:
                        decoded_line = raw_line.decode("ascii", errors="ignore")
                        self._process_line(decoded_line)
                    except Exception:
                        pass
        except Exception:
            pass

        return self.target

    def send_direction(self, direction: str, only_on_change: bool = True) -> bool:
        """
        Send a direction response to the ESP32.

        Message format:
            D,<direction>\\n

        Args:
            direction: One of 'l', 'm', 'r', or 'n'
            only_on_change: If True, skip repeated identical messages

        Returns:
            True if a message was sent or printed, False if skipped
        """
        direction = (direction or "n").strip().lower()
        if direction not in VALID_DIRECTIONS:
            direction = "n"

        message = f"D,{direction}\n"
        now = time.time()

        if (now - self._last_send_time) < self.min_send_interval:
            return False

        if only_on_change and self._last_sent_message == message:
            return False

        if self.print_only or self.serial_conn is None:
            print(f"[SERIAL-DRY] {message.strip()}")
        else:
            try:
                self.serial_conn.write(message.encode("ascii", errors="ignore"))
            except Exception:
                self.print_only = True
                print(f"[SERIAL-DRY] {message.strip()}")

        self._last_send_time = now
        self._last_sent_message = message
        return True

    def send_dir(self, direction: str, only_on_change: bool = True) -> bool:
        """
        Backward-compatible alias for send_direction().

        Keeps older runtime files working without modification.
        """
        return self.send_direction(direction, only_on_change=only_on_change)