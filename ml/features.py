"""Workload-adjusted and temporal features for sensor data."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "temperature_excess",
    "vibration_excess",
    "current_excess",
    "rpm_drop",
    "temperature_mean_5",
    "vibration_mean_5",
    "current_mean_5",
    "temperature_slope_5",
    "vibration_slope_5",
    "current_slope_5",
    "rpm_std_5",
]

FEATURE_LABELS = {
    "temperature_excess": "Temperature above workload baseline",
    "vibration_excess": "Vibration above workload baseline",
    "current_excess": "Motor current above workload baseline",
    "rpm_drop": "RPM below nominal",
    "temperature_mean_5": "Recent temperature trend",
    "vibration_mean_5": "Recent vibration trend",
    "current_mean_5": "Recent motor-current trend",
    "temperature_slope_5": "Temperature rate of change",
    "vibration_slope_5": "Vibration rate of change",
    "current_slope_5": "Motor-current rate of change",
    "rpm_std_5": "Recent RPM instability",
}


def engineer_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of sensor data with workload-adjusted rolling features."""
    required = {
        "machine_id",
        "operating_hours",
        "temperature",
        "vibration",
        "motor_current",
        "rpm",
        "workload",
    }
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Sensor data is missing required columns: {', '.join(sorted(missing))}")
    if readings.empty:
        raise ValueError("Sensor data must contain at least one reading")

    frame = readings.copy()
    frame["_original_order"] = range(len(frame))
    frame = frame.sort_values(["machine_id", "operating_hours", "_original_order"])
    groups = frame.groupby("machine_id", sort=False)

    frame["temperature_excess"] = frame["temperature"] - (38.0 + 0.20 * frame["workload"])
    frame["vibration_excess"] = frame["vibration"] - (0.75 + 0.006 * frame["workload"])
    frame["current_excess"] = frame["motor_current"] - (4.0 + 0.075 * frame["workload"])
    frame["rpm_drop"] = 1800.0 - frame["rpm"]

    for source, destination in (
        ("temperature_excess", "temperature_mean_5"),
        ("vibration_excess", "vibration_mean_5"),
        ("current_excess", "current_mean_5"),
    ):
        frame[destination] = groups[source].transform(
            lambda values: values.rolling(window=5, min_periods=1).mean()
        )

    for source, destination in (
        ("temperature_excess", "temperature_slope_5"),
        ("vibration_excess", "vibration_slope_5"),
        ("current_excess", "current_slope_5"),
    ):
        frame[destination] = groups[source].transform(lambda values: values.diff(4) / 4.0).fillna(0.0)

    frame["rpm_std_5"] = groups["rpm"].transform(
        lambda values: values.rolling(window=5, min_periods=2).std()
    ).fillna(0.0)
    return frame.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)
