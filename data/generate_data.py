"""Generate reproducible labeled sensor data for the prototype models."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from simulator import MachineSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "synthetic_sensor_data.csv"
DEFAULT_WORKLOADS = (25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0)


def generate_dataset(
    output_path: str | Path = DEFAULT_OUTPUT,
    seed: int = 42,
    normal_samples: int = 12,
    machine_count: int = 8,
) -> pd.DataFrame:
    """Generate healthy, degrading, and near-failure time series for a fleet."""
    if normal_samples < 1:
        raise ValueError("normal_samples must be at least 1")
    if machine_count < 2:
        raise ValueError("machine_count must be at least 2 for group evaluation")

    rows: list[dict[str, str | float]] = []
    for machine_number in range(machine_count):
        workload = DEFAULT_WORKLOADS[machine_number % len(DEFAULT_WORKLOADS)]
        simulator = MachineSimulator(
            machine_id=f"CNC-{machine_number + 1:02d}",
            seed=seed + machine_number,
            workload=workload,
        )
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=machine_number)
        rows.extend(_sample_row(simulator, base_time) for _ in range(normal_samples))
        simulator.start_degradation()
        while simulator.degradation < 1.0:
            rows.append(_sample_row(simulator, base_time))
        simulator.set_near_failure()
        rows.extend(_sample_row(simulator, base_time) for _ in range(3))

    frame = pd.DataFrame(rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def _sample_row(simulator: MachineSimulator, base_time: datetime) -> dict[str, str | float]:
    reading = simulator.step()
    row = reading.to_dict()
    row["timestamp"] = (base_time + timedelta(hours=reading.operating_hours)).isoformat(timespec="seconds")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-samples", type=int, default=12)
    parser.add_argument("--machines", type=int, default=8)
    args = parser.parse_args()
    frame = generate_dataset(args.output, args.seed, args.normal_samples, args.machines)
    print(f"Saved {len(frame)} sensor readings for {frame['machine_id'].nunique()} machines to {args.output}")


if __name__ == "__main__":
    main()
