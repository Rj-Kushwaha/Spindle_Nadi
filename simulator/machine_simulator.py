"""A lightweight, reproducible simulator for spindle sensor readings."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Scenario(str, Enum):
    NORMAL = "NORMAL"
    DEGRADATION = "DEGRADATION"
    NEAR_FAILURE = "NEAR_FAILURE"


FAULT_MODES = (
    "bearing_wear",
    "cooling_degradation",
    "lubrication_loss",
    "tool_wear",
    "spindle_imbalance",
)


@dataclass(frozen=True)
class SensorReading:
    timestamp: str
    machine_id: str
    temperature: float
    vibration: float
    vibration_x: float
    vibration_y: float
    vibration_z: float
    motor_current: float
    rpm_setpoint: float
    rpm: float
    spindle_load_pct: float
    torque_nm: float
    bearing_temperature_c: float
    coolant_flow_lpm: float
    coolant_temperature_c: float
    lubrication_pressure_bar: float
    acoustic_noise_db: float
    feed_rate_mm_min: float
    tool_wear_pct_est: float
    power_kw_est: float
    operating_hours: float
    workload: float
    status: str
    scenario: str
    fault_mode: str
    degradation: float

    def to_dict(self) -> dict[str, str | float]:
        """Return a flat record suitable for CSV or later data processing."""
        return {
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "temperature": self.temperature,
            "vibration": self.vibration,
            "vibration_x": self.vibration_x,
            "vibration_y": self.vibration_y,
            "vibration_z": self.vibration_z,
            "motor_current": self.motor_current,
            "rpm_setpoint": self.rpm_setpoint,
            "rpm": self.rpm,
            "spindle_load_pct": self.spindle_load_pct,
            "torque_nm": self.torque_nm,
            "bearing_temperature_c": self.bearing_temperature_c,
            "coolant_flow_lpm": self.coolant_flow_lpm,
            "coolant_temperature_c": self.coolant_temperature_c,
            "lubrication_pressure_bar": self.lubrication_pressure_bar,
            "acoustic_noise_db": self.acoustic_noise_db,
            "feed_rate_mm_min": self.feed_rate_mm_min,
            "tool_wear_pct_est": self.tool_wear_pct_est,
            "power_kw_est": self.power_kw_est,
            "operating_hours": self.operating_hours,
            "workload": self.workload,
            "status": self.status,
            "scenario": self.scenario,
            "fault_mode": self.fault_mode,
            "degradation": self.degradation,
        }

    def format(self) -> str:
        """Format a reading for display in the interactive simulator."""
        return (
            f"{self.timestamp} | {self.machine_id} | {self.status} | "
            f"Temp {self.temperature:.1f} C | Vibration {self.vibration:.2f} mm/s | "
            f"Current {self.motor_current:.2f} A | RPM {self.rpm:.0f} | "
            f"Workload {self.workload:.0f}% | Operating hours {self.operating_hours:.1f} | "
            f"Degradation {self.degradation:.0%}"
        )


class MachineSimulator:
    """Generate noisy sensor readings for normal and degrading machine states."""

    WARNING_THRESHOLD = 0.50
    CRITICAL_THRESHOLD = 0.85
    DEGRADATION_PER_STEP = 0.06

    def __init__(
        self,
        machine_id: str = "CNC-01",
        seed: int | None = None,
        hours_per_step: float = 0.1,
        workload: float = 55.0,
        rpm_setpoint: float = 1800.0,
    ) -> None:
        if not machine_id.strip():
            raise ValueError("machine_id cannot be empty")
        if hours_per_step <= 0:
            raise ValueError("hours_per_step must be greater than zero")
        self.machine_id = machine_id
        self.hours_per_step = hours_per_step
        self._random = random.Random(seed)
        self._initial_workload = self._validate_workload(workload)
        self._initial_rpm_setpoint = self._validate_rpm(rpm_setpoint)
        self.reset()

    def set_normal(self) -> None:
        """Return the machine to healthy operation without resetting its clock."""
        self.scenario = Scenario.NORMAL
        self.degradation = 0.0

    def start_degradation(self, fault_mode: str = "bearing_wear") -> None:
        """Begin a gradual degradation sequence from the current machine state."""
        if fault_mode not in FAULT_MODES:
            raise ValueError(f"fault_mode must be one of: {', '.join(FAULT_MODES)}")
        if self.scenario == Scenario.NORMAL:
            self.degradation = 0.0
        self.fault_mode = fault_mode
        self.scenario = Scenario.DEGRADATION

    def set_near_failure(self) -> None:
        """Move directly to the near-failure scenario for simulator testing."""
        self.scenario = Scenario.NEAR_FAILURE
        if self.fault_mode == "NONE":
            self.fault_mode = FAULT_MODES[0]
        self.degradation = 1.0

    def apply_maintenance(self, recovery_fraction: float = 0.85) -> float:
        """Virtually reduce accumulated wear while preserving operating hours."""
        if not 0.0 <= recovery_fraction <= 1.0:
            raise ValueError("recovery_fraction must be between 0 and 1")
        self.degradation = round(self.degradation * (1.0 - recovery_fraction), 3)
        self.scenario = Scenario.NORMAL
        self.fault_mode = "NONE"
        return self.degradation

    def set_workload(self, workload: float) -> None:
        """Set workload percentage for subsequent readings."""
        self.workload = self._validate_workload(workload)

    def set_rpm_setpoint(self, rpm_setpoint: float) -> None:
        """Set the commanded spindle speed for subsequent readings."""
        self.rpm_setpoint = self._validate_rpm(rpm_setpoint)

    def reset(self) -> None:
        """Restore a healthy machine and reset elapsed operating hours."""
        self.scenario = Scenario.NORMAL
        self.fault_mode = "NONE"
        self.degradation = 0.0
        self.operating_hours = 0.0
        self.workload = self._initial_workload
        self.rpm_setpoint = self._initial_rpm_setpoint

    def step(self) -> SensorReading:
        """Advance simulation by one sample and return the generated sensor data."""
        if self.scenario == Scenario.DEGRADATION:
            self.degradation = min(1.0, self.degradation + self.DEGRADATION_PER_STEP)

        self.operating_hours += self.hours_per_step
        degradation = self.degradation
        workload = self.workload

        rpm_delta = self.rpm_setpoint - 1800.0
        temperature = 38.0 + workload * 0.20 + rpm_delta * 0.0012 + degradation * 22.0
        vibration = 0.75 + workload * 0.006 + rpm_delta * 0.00008 + degradation * 4.0
        motor_current = 4.0 + workload * 0.075 + rpm_delta * 0.0003 + degradation * 5.0
        rpm = self.rpm_setpoint * (1.0 - degradation * 0.10)
        rpm_noise = (4.0 / 1800.0 + degradation * 0.02) * self.rpm_setpoint
        fault = self.fault_mode

        spindle_load = min(100.0, workload + degradation * 15.0)
        torque = 0.25 * spindle_load + degradation * 4.0
        bearing_temperature = 32.0 + workload * 0.15 + rpm_delta * 0.0004 + degradation * 8.0
        coolant_flow = max(0.0, 8.0 - workload * 0.01)
        coolant_temperature = 22.0 + workload * 0.08
        lubrication_pressure = 3.5 - workload * 0.003
        acoustic_noise = 58.0 + workload * 0.08 + rpm_delta * 0.0001 + degradation * 5.0
        feed_rate = 250.0 + workload * 14.0
        tool_wear = min(100.0, degradation * 8.0)

        if fault == "bearing_wear":
            bearing_temperature += degradation * 30.0
            acoustic_noise += degradation * 22.0
        elif fault == "cooling_degradation":
            temperature += degradation * 18.0
            coolant_flow = max(0.0, coolant_flow - degradation * 7.0)
            coolant_temperature += degradation * 18.0
        elif fault == "lubrication_loss":
            bearing_temperature += degradation * 18.0
            lubrication_pressure = max(0.0, lubrication_pressure - degradation * 2.8)
            vibration += degradation * 1.2
        elif fault == "tool_wear":
            tool_wear = min(100.0, 100.0 * degradation)
            torque += degradation * 7.0
            feed_rate *= 1.0 - 0.3 * degradation
            acoustic_noise += degradation * 12.0
        elif fault == "spindle_imbalance":
            acoustic_noise += degradation * 18.0

        vibration_scale = vibration / 1.0
        vibration_x = vibration_scale * 0.577350 + (degradation * 3.5 if fault == "spindle_imbalance" else 0.0)
        vibration_y = vibration_scale * 0.577350 + (degradation * 1.2 if fault == "spindle_imbalance" else 0.0)
        vibration_z = vibration_scale * 0.577350
        if fault == "bearing_wear":
            vibration_x += degradation * 0.8
            vibration_z += degradation * 1.8

        power = torque * rpm / 9550.0

        status = self._status_for(degradation)
        return SensorReading(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            machine_id=self.machine_id,
            temperature=round(temperature + self._random.gauss(0.0, 0.7), 2),
            vibration=round(max(0.0, vibration + self._random.gauss(0.0, 0.08)), 3),
            vibration_x=round(max(0.0, vibration_x + self._random.gauss(0.0, 0.035)), 3),
            vibration_y=round(max(0.0, vibration_y + self._random.gauss(0.0, 0.035)), 3),
            vibration_z=round(max(0.0, vibration_z + self._random.gauss(0.0, 0.035)), 3),
            motor_current=round(max(0.0, motor_current + self._random.gauss(0.0, 0.18)), 2),
            rpm_setpoint=self.rpm_setpoint,
            rpm=round(max(0.0, rpm + self._random.gauss(0.0, rpm_noise)), 1),
            spindle_load_pct=round(min(100.0, spindle_load + self._random.gauss(0.0, 1.0)), 2),
            torque_nm=round(max(0.0, torque + self._random.gauss(0.0, 0.4)), 2),
            bearing_temperature_c=round(bearing_temperature + self._random.gauss(0.0, 0.6), 2),
            coolant_flow_lpm=round(max(0.0, coolant_flow + self._random.gauss(0.0, 0.12)), 2),
            coolant_temperature_c=round(coolant_temperature + self._random.gauss(0.0, 0.5), 2),
            lubrication_pressure_bar=round(max(0.0, lubrication_pressure + self._random.gauss(0.0, 0.04)), 2),
            acoustic_noise_db=round(acoustic_noise + self._random.gauss(0.0, 0.8), 2),
            feed_rate_mm_min=round(max(0.0, feed_rate + self._random.gauss(0.0, 2.0)), 2),
            tool_wear_pct_est=round(min(100.0, max(0.0, tool_wear + self._random.gauss(0.0, 1.0))), 2),
            power_kw_est=round(max(0.0, power + self._random.gauss(0.0, 0.08)), 2),
            operating_hours=round(self.operating_hours, 2),
            workload=workload,
            status=status,
            scenario=self.scenario.value,
            fault_mode=fault,
            degradation=round(degradation, 3),
        )

    @classmethod
    def _status_for(cls, degradation: float) -> str:
        if degradation >= cls.CRITICAL_THRESHOLD:
            return "CRITICAL"
        if degradation >= cls.WARNING_THRESHOLD:
            return "WARNING"
        return "NORMAL"

    @staticmethod
    def _validate_workload(workload: float) -> float:
        if not 0.0 <= workload <= 100.0:
            raise ValueError("workload must be between 0 and 100")
        return float(workload)

    @staticmethod
    def _validate_rpm(rpm_setpoint: float) -> float:
        if not 500.0 <= rpm_setpoint <= 30_000.0:
            raise ValueError("RPM setpoint must be between 500 and 30000")
        return float(rpm_setpoint)
