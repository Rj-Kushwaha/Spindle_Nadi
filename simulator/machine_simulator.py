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


@dataclass(frozen=True)
class SensorReading:
    timestamp: str
    machine_id: str
    temperature: float
    vibration: float
    motor_current: float
    rpm: float
    operating_hours: float
    workload: float
    status: str
    scenario: str
    degradation: float

    def to_dict(self) -> dict[str, str | float]:
        """Return a flat record suitable for CSV or later data processing."""
        return {
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "temperature": self.temperature,
            "vibration": self.vibration,
            "motor_current": self.motor_current,
            "rpm": self.rpm,
            "operating_hours": self.operating_hours,
            "workload": self.workload,
            "status": self.status,
            "scenario": self.scenario,
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

    WARNING_THRESHOLD = 0.40
    CRITICAL_THRESHOLD = 0.80
    DEGRADATION_PER_STEP = 0.06

    def __init__(
        self,
        machine_id: str = "CNC-01",
        seed: int | None = None,
        hours_per_step: float = 0.1,
        workload: float = 55.0,
    ) -> None:
        if not machine_id.strip():
            raise ValueError("machine_id cannot be empty")
        if hours_per_step <= 0:
            raise ValueError("hours_per_step must be greater than zero")
        self.machine_id = machine_id
        self.hours_per_step = hours_per_step
        self._random = random.Random(seed)
        self._initial_workload = self._validate_workload(workload)
        self.reset()

    def set_normal(self) -> None:
        """Return the machine to healthy operation without resetting its clock."""
        self.scenario = Scenario.NORMAL
        self.degradation = 0.0

    def start_degradation(self) -> None:
        """Begin a gradual degradation sequence from the current machine state."""
        if self.scenario == Scenario.NORMAL:
            self.degradation = 0.0
        self.scenario = Scenario.DEGRADATION

    def set_near_failure(self) -> None:
        """Move directly to the near-failure scenario for simulator testing."""
        self.scenario = Scenario.NEAR_FAILURE
        self.degradation = 1.0

    def apply_maintenance(self, recovery_fraction: float = 0.85) -> float:
        """Virtually reduce accumulated wear while preserving operating hours."""
        if not 0.0 <= recovery_fraction <= 1.0:
            raise ValueError("recovery_fraction must be between 0 and 1")
        self.degradation = round(self.degradation * (1.0 - recovery_fraction), 3)
        self.scenario = Scenario.NORMAL
        return self.degradation

    def set_workload(self, workload: float) -> None:
        """Set workload percentage for subsequent readings."""
        self.workload = self._validate_workload(workload)

    def reset(self) -> None:
        """Restore a healthy machine and reset elapsed operating hours."""
        self.scenario = Scenario.NORMAL
        self.degradation = 0.0
        self.operating_hours = 0.0
        self.workload = self._initial_workload

    def step(self) -> SensorReading:
        """Advance simulation by one sample and return the generated sensor data."""
        if self.scenario == Scenario.DEGRADATION:
            self.degradation = min(1.0, self.degradation + self.DEGRADATION_PER_STEP)

        self.operating_hours += self.hours_per_step
        degradation = self.degradation
        workload = self.workload

        temperature = 38.0 + workload * 0.20 + degradation * 22.0
        vibration = 0.75 + workload * 0.006 + degradation * 4.0
        motor_current = 4.0 + workload * 0.075 + degradation * 5.0
        rpm = 1800.0 - degradation * 180.0
        rpm_noise = 4.0 + degradation * 35.0

        status = self._status_for(degradation)
        return SensorReading(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            machine_id=self.machine_id,
            temperature=round(temperature + self._random.gauss(0.0, 0.7), 2),
            vibration=round(max(0.0, vibration + self._random.gauss(0.0, 0.08)), 3),
            motor_current=round(max(0.0, motor_current + self._random.gauss(0.0, 0.18)), 2),
            rpm=round(max(0.0, rpm + self._random.gauss(0.0, rpm_noise)), 1),
            operating_hours=round(self.operating_hours, 2),
            workload=workload,
            status=status,
            scenario=self.scenario.value,
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
