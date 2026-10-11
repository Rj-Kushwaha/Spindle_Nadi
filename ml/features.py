"""Workload-adjusted and temporal features for sensor data."""

from __future__ import annotations

import pandas as pd

REQUIRED_SENSOR_COLUMNS = {
    "machine_id",
    "operating_hours",
    "temperature",
    "vibration",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "motor_current",
    "rpm_setpoint",
    "rpm",
    "workload",
    "spindle_load_pct",
    "torque_nm",
    "bearing_temperature_c",
    "coolant_flow_lpm",
    "coolant_temperature_c",
    "lubrication_pressure_bar",
    "acoustic_noise_db",
    "feed_rate_mm_min",
    "tool_wear_pct_est",
    "power_kw_est",
}

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
    "vibration_x_excess",
    "vibration_y_excess",
    "vibration_z_excess",
    "spindle_load_excess",
    "torque_excess",
    "bearing_temperature_excess",
    "coolant_flow_drop",
    "coolant_temperature_excess",
    "lubrication_pressure_drop",
    "acoustic_noise_excess",
    "feed_rate_drop",
    "tool_wear_est",
    "power_excess",
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
    "vibration_x_excess": "X-axis vibration above baseline",
    "vibration_y_excess": "Y-axis vibration above baseline",
    "vibration_z_excess": "Z-axis vibration above baseline",
    "spindle_load_excess": "Spindle load above workload",
    "torque_excess": "Torque above workload baseline",
    "bearing_temperature_excess": "Bearing temperature above baseline",
    "coolant_flow_drop": "Coolant flow below baseline",
    "coolant_temperature_excess": "Coolant temperature above baseline",
    "lubrication_pressure_drop": "Lubrication pressure below baseline",
    "acoustic_noise_excess": "Acoustic level above baseline",
    "feed_rate_drop": "Feed rate below workload baseline",
    "tool_wear_est": "Estimated tool wear",
    "power_excess": "Estimated power above baseline",
}


def engineer_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of sensor data with workload-adjusted rolling features."""
    missing = REQUIRED_SENSOR_COLUMNS.difference(readings.columns)
    if missing:
        raise ValueError(f"Sensor data is missing required columns: {', '.join(sorted(missing))}")
    if readings.empty:
        raise ValueError("Sensor data must contain at least one reading")

    frame = readings.copy()
    frame["_original_order"] = range(len(frame))
    frame = frame.sort_values(["machine_id", "operating_hours", "_original_order"])
    groups = frame.groupby("machine_id", sort=False)

    rpm_delta = frame["rpm_setpoint"] - 1800.0
    frame["temperature_excess"] = frame["temperature"] - (
        38.0 + 0.20 * frame["workload"] + rpm_delta * 0.0012
    )
    frame["vibration_excess"] = frame["vibration"] - (
        0.75 + 0.006 * frame["workload"] + rpm_delta * 0.00008
    )
    frame["current_excess"] = frame["motor_current"] - (
        4.0 + 0.075 * frame["workload"] + rpm_delta * 0.0003
    )
    frame["rpm_drop"] = (frame["rpm_setpoint"] - frame["rpm"]) / frame["rpm_setpoint"] * 1800.0
    expected_vibration = 0.75 + 0.006 * frame["workload"] + rpm_delta * 0.00008
    for axis in ("x", "y", "z"):
        frame[f"vibration_{axis}_excess"] = (
            frame[f"vibration_{axis}"] - expected_vibration * 0.577350
        )
    frame["spindle_load_excess"] = frame["spindle_load_pct"] - frame["workload"]
    frame["torque_excess"] = frame["torque_nm"] - 0.25 * frame["workload"]
    frame["bearing_temperature_excess"] = (
        frame["bearing_temperature_c"] - (32.0 + 0.15 * frame["workload"] + rpm_delta * 0.0004)
    )
    frame["coolant_flow_drop"] = (
        (8.0 - 0.01 * frame["workload"]) - frame["coolant_flow_lpm"]
    )
    frame["coolant_temperature_excess"] = (
        frame["coolant_temperature_c"] - (22.0 + 0.08 * frame["workload"])
    )
    frame["lubrication_pressure_drop"] = (
        (3.5 - 0.003 * frame["workload"]) - frame["lubrication_pressure_bar"]
    )
    frame["acoustic_noise_excess"] = (
        frame["acoustic_noise_db"] - (58.0 + 0.08 * frame["workload"] + rpm_delta * 0.0001)
    )
    frame["feed_rate_drop"] = (
        (250.0 + 14.0 * frame["workload"]) - frame["feed_rate_mm_min"]
    )
    frame["tool_wear_est"] = frame["tool_wear_pct_est"]
    frame["power_excess"] = (
        frame["power_kw_est"] - (
            0.25 * frame["workload"] * frame["rpm_setpoint"] / 9550.0
        )
    )

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

    normalized_rpm = frame["rpm"] / frame["rpm_setpoint"] * 1800.0
    frame["rpm_std_5"] = normalized_rpm.groupby(frame["machine_id"], sort=False).transform(
        lambda values: values.rolling(window=5, min_periods=2).std()
    ).fillna(0.0)
    return frame.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)
