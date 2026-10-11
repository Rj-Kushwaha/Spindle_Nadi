"""Read calibrated ESP32 telemetry sent as newline-delimited JSON over USB."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import math
from typing import Any, Callable

PHYSICAL_CHANNELS = {
    "temperature": {"temperature", "temperature_c", "temp_c"},
    "vibration": {"vibration", "vibration_mm_s"},
    "vibration_x": {"vibration_x", "vibration_x_mm_s"},
    "vibration_y": {"vibration_y", "vibration_y_mm_s"},
    "vibration_z": {"vibration_z", "vibration_z_mm_s"},
    "motor_current": {"motor_current", "current_a"},
    "rpm": {"rpm", "spindle_speed_rpm"},
}


def parse_esp32_line(line: bytes | str) -> dict[str, str | float]:
    """Parse one calibrated JSON telemetry line, rejecting invalid sensor data."""
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON line: {error}") from error
    if not isinstance(message, dict):
        raise ValueError("Telemetry line must be a JSON object")

    result: dict[str, str | float] = {
        "timestamp": _parse_timestamp(message.get("timestamp")),
        "machine_id": str(message.get("machine_id", message.get("device_id", "ESP32"))),
        "data_source": "ESP32 USB · physical telemetry",
    }
    for key, value in message.items():
        if key in {"timestamp", "machine_id", "device_id"}:
            continue
        canonical = next(
            (channel for channel, aliases in PHYSICAL_CHANNELS.items() if key in aliases),
            None,
        )
        if canonical is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"Sensor value {key!r} must be a finite number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sensor value {key!r} must be a finite number") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"Sensor value {key!r} must be a finite number")
        if canonical in result:
            raise ValueError(f"Multiple fields map to sensor channel {canonical!r}")
        result[canonical] = numeric_value

    if not set(PHYSICAL_CHANNELS).intersection(result):
        raise ValueError("Telemetry line must include at least one supported sensor channel")
    return result


def list_esp32_ports() -> list[tuple[str, str]]:
    """Return available serial device paths and descriptions."""
    list_ports = importlib.import_module("serial.tools.list_ports")
    return [(port.device, port.description) for port in list_ports.comports()]


class Esp32SerialReader:
    """Non-blocking reader for JSON Lines emitted by an ESP32 over USB serial."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not port.strip():
            raise ValueError("A serial port must be selected")
        if baudrate not in {9600, 19200, 38400, 57600, 115200, 230400}:
            raise ValueError("Unsupported baud rate")
        if serial_factory is None:
            serial_module = importlib.import_module("serial")
            serial_factory = serial_module.Serial
        self.connection = serial_factory(port, baudrate=baudrate, timeout=0)
        self._buffer = bytearray()
        self.closed = False

    def read_available(self) -> tuple[list[dict[str, str | float]], list[str]]:
        """Read complete lines currently available without waiting for the next sample."""
        if self.closed:
            return [], []
        waiting = int(self.connection.in_waiting)
        if waiting:
            self._buffer.extend(self.connection.read(waiting))
        if len(self._buffer) > 65_536 and b"\n" not in self._buffer:
            self._buffer.clear()
            return [], ["Serial line exceeded the 64 KiB maximum and was discarded."]

        records: list[dict[str, str | float]] = []
        errors: list[str] = []
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            if not line.strip():
                continue
            if len(line) > 65_536:
                errors.append("Serial line exceeded the 64 KiB maximum and was discarded.")
                continue
            try:
                records.append(parse_esp32_line(line.strip()))
            except ValueError as error:
                errors.append(str(error))
        return records, errors

    def close(self) -> None:
        """Close the device connection once."""
        if not self.closed:
            self.connection.close()
            self.closed = True


def _parse_timestamp(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be a valid ISO-8601 value") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")
