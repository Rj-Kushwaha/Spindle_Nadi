"""Connect simulators, model analysis, alerts, and local CSV history."""

from __future__ import annotations

import csv
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import pandas as pd

from data.generate_data import DEFAULT_OUTPUT, generate_dataset
from ml.features import REQUIRED_SENSOR_COLUMNS
from ml.pipeline import MachineAnalysis, SpindleNadiPipeline
from ml.prediction.risk_model import CRITICAL_THRESHOLD, WARNING_THRESHOLD
from simulator import MachineSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "generated" / "live_sensor_readings.csv"
MAINTENANCE_PATH = PROJECT_ROOT / "data" / "generated" / "maintenance_history.csv"
EVENT_TIMELINE_PATH = PROJECT_ROOT / "data" / "generated" / "event_timeline.csv"
NOTIFICATION_HISTORY_PATH = PROJECT_ROOT / "data" / "generated" / "notification_history.csv"

FIXED_LIMITS = {
    "temperature": 75.0,
    "vibration": 3.0,
    "motor_current": 12.5,
    "rpm": 1600.0,
    "bearing_temperature_c": 80.0,
    "coolant_flow_lpm": 4.0,
    "lubrication_pressure_bar": 1.0,
}

SENSOR_FIELDS = [
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "motor_current",
    "rpm_setpoint",
    "rpm",
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
    "operating_hours",
    "workload",
    "scenario",
    "fault_mode",
    "degradation",
    "risk",
    "status",
    "anomaly_score",
    "is_anomaly",
    "rul_hours",
    "health_score",
]
MAINTENANCE_FIELDS = [
    "timestamp",
    "machine_id",
    "status",
    "risk",
    "alert",
    "reason",
    "recommendation",
]
EVENT_FIELDS = [
    "timestamp",
    "machine_id",
    "event_type",
    "status_before",
    "risk_before",
    "status_after",
    "risk_after",
    "details",
]
NOTIFICATION_FIELDS = [
    "timestamp",
    "machine_id",
    "event_type",
    "status",
    "risk",
    "message",
    "dashboard_status",
    "email_status",
    "email_detail",
    "whatsapp_status",
    "whatsapp_detail",
]


class SpindleNadiService:
    """Stateful fleet service used by the dashboard and integration tests."""

    def __init__(
        self,
        live_data_path: str | Path = LIVE_DATA_PATH,
        maintenance_path: str | Path = MAINTENANCE_PATH,
        event_timeline_path: str | Path = EVENT_TIMELINE_PATH,
        training_path: str | Path = DEFAULT_OUTPUT,
        pipeline: SpindleNadiPipeline | None = None,
        notification_path: str | Path = NOTIFICATION_HISTORY_PATH,
        notifier: Any | None = None,
    ) -> None:
        self.live_data_path = Path(live_data_path)
        self.maintenance_path = Path(maintenance_path)
        self.event_timeline_path = Path(event_timeline_path)
        self.notification_path = Path(notification_path)
        self.notifier = notifier
        training_file = Path(training_path)
        if training_file.exists():
            training_data = pd.read_csv(training_file)
        else:
            training_data = generate_dataset(training_file)
        if REQUIRED_SENSOR_COLUMNS.difference(training_data.columns):
            training_data = generate_dataset(training_file)
        self.pipeline = pipeline or SpindleNadiPipeline(training_data)

        fleet = {
            "CNC-01": 55.0,
            "CNC-02": 70.0,
            "CNC-03": 40.0,
            "3D-Printer-01": 30.0,
        }
        self.simulators = {
            machine_id: MachineSimulator(machine_id=machine_id, seed=100 + index, workload=workload)
            for index, (machine_id, workload) in enumerate(fleet.items())
        }
        self.session_records: dict[str, list[dict[str, Any]]] = {
            machine_id: [] for machine_id in self.simulators
        }
        self._revision = {machine_id: 0 for machine_id in self.simulators}
        self._last_status = {machine_id: "NORMAL" for machine_id in self.simulators}
        self.maintenance_history = self._read_maintenance_history()
        self.event_timeline = self._read_event_timeline()
        self.notification_history = self._read_notification_history()
        for machine_id in self.simulators:
            self.advance(machine_id)

    @property
    def machine_ids(self) -> list[str]:
        return list(self.simulators)

    def latest(self, machine_id: str) -> dict[str, Any]:
        return self.session_records[machine_id][-1]

    def revision(self, machine_id: str) -> int:
        self._simulator(machine_id)
        return self._revision[machine_id]

    def records_for(self, machine_id: str) -> pd.DataFrame:
        return pd.DataFrame(self.session_records[machine_id])

    def all_session_records(self) -> pd.DataFrame:
        records = [record for machine_records in self.session_records.values() for record in machine_records]
        return pd.DataFrame(records)

    def fleet_summary(self) -> list[dict[str, Any]]:
        rank = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        records = [self.latest(machine_id) for machine_id in self.machine_ids]
        return sorted(
            records,
            key=lambda row: (
                -rank[row["status"]],
                -row["risk"],
                -row["anomaly_score"],
                row["rul_hours"],
            ),
        )

    def start_degradation(self, machine_id: str, fault_mode: str = "bearing_wear") -> None:
        self._simulator(machine_id).start_degradation(fault_mode)

    def set_near_failure(self, machine_id: str) -> None:
        self._simulator(machine_id).set_near_failure()

    def set_normal(self, machine_id: str) -> None:
        self._simulator(machine_id).set_normal()

    def set_workload(self, machine_id: str, workload: float) -> None:
        self._simulator(machine_id).set_workload(workload)

    def set_rpm_setpoint(self, machine_id: str, rpm_setpoint: float) -> None:
        self._simulator(machine_id).set_rpm_setpoint(rpm_setpoint)

    @staticmethod
    def sensor_health(record: dict[str, Any], max_age_seconds: float = 30.0) -> dict[str, Any]:
        """Check required synthetic channels for missing, invalid, implausible, or stale values."""
        ranges = {
            "temperature": (-20.0, 200.0),
            "vibration": (0.0, 100.0),
            "vibration_x": (0.0, 100.0),
            "vibration_y": (0.0, 100.0),
            "vibration_z": (0.0, 100.0),
            "motor_current": (0.0, 1000.0),
            "rpm": (0.0, 30_000.0),
            "rpm_setpoint": (500.0, 30_000.0),
            "spindle_load_pct": (0.0, 100.0),
            "torque_nm": (0.0, 100_000.0),
            "bearing_temperature_c": (-20.0, 200.0),
            "coolant_flow_lpm": (0.0, 1000.0),
            "coolant_temperature_c": (-20.0, 200.0),
            "lubrication_pressure_bar": (0.0, 1000.0),
            "acoustic_noise_db": (0.0, 200.0),
            "feed_rate_mm_min": (0.0, 100_000.0),
            "tool_wear_pct_est": (0.0, 100.0),
            "power_kw_est": (0.0, 100_000.0),
        }
        issues = []
        valid_count = 0
        for channel, (minimum, maximum) in ranges.items():
            value = record.get(channel)
            if value is None:
                issues.append({"sensor": channel, "issue": "missing"})
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                issues.append({"sensor": channel, "issue": "not numeric"})
                continue
            if not math.isfinite(numeric_value):
                issues.append({"sensor": channel, "issue": "not finite"})
                continue
            if not minimum <= numeric_value <= maximum:
                issues.append({"sensor": channel, "issue": "outside broad plausibility range"})
                continue
            valid_count += 1

        timestamp = record.get("timestamp")
        try:
            sample_time = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if sample_time.tzinfo is None:
                sample_time = sample_time.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (datetime.now(timezone.utc) - sample_time.astimezone(timezone.utc)).total_seconds())
            if age_seconds > max_age_seconds:
                issues.append({"sensor": "timestamp", "issue": f"stale ({age_seconds:.0f}s old)"})
        except (TypeError, ValueError):
            age_seconds = None
            issues.append({"sensor": "timestamp", "issue": "missing or invalid"})

        if valid_count == 0:
            status = "UNAVAILABLE"
        elif issues:
            status = "CHECK"
        else:
            status = "HEALTHY"
        return {
            "status": status,
            "quality_pct": round(valid_count / len(ranges) * 100.0),
            "valid_channels": valid_count,
            "total_channels": len(ranges),
            "age_seconds": age_seconds,
            "issues": issues,
        }

    def maintenance_report(self, machine_id: str) -> str:
        """Render a downloadable, evidence-grounded report for one machine."""
        record = self.latest(self._simulator(machine_id).machine_id)
        sensor_health = self.sensor_health(record)
        comparison = self.comparison(record)
        history = self.records_for(machine_id)
        risk_trend = "insufficient readings"
        if len(history) >= 2:
            midpoint = max(1, len(history) // 2)
            earlier = float(history.iloc[:midpoint]["risk"].mean())
            recent = float(history.iloc[midpoint:]["risk"].mean())
            risk_trend = f"{earlier:.0%} average earlier → {recent:.0%} average recent"
        explanations = "\n".join(
            f"- {item['feature']}: estimated contribution {float(item['contribution']):.3f}"
            for item in record["explanations"][:5]
        ) or "- No ranked factors available"
        alerts = [
            row for row in self.notification_history
            if row["machine_id"] == machine_id
        ][-10:]
        alert_lines = "\n".join(
            f"- {row['timestamp']}: {row['event_type']} — email {row['email_status']}, "
            f"WhatsApp {row['whatsapp_status']}"
            for row in alerts
        ) or "- No alerts recorded"
        recommendation = self.recommend_maintenance(record["explanations"][:3])
        issues = "\n".join(
            f"- {item['sensor']}: {item['issue']}"
            for item in sensor_health["issues"]
        ) or "- No channel-quality issues detected"
        maintenance = [
            row for row in self.maintenance_history
            if row["machine_id"] == machine_id
        ][-10:]
        maintenance_lines = "\n".join(
            f"- {row['timestamp']}: {row['alert']} — {row['reason']}; {row['recommendation']}"
            for row in maintenance
        ) or "- No maintenance events recorded"
        return (
            f"# Spindle Nadi maintenance report — {machine_id}\n\n"
            f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"- Data source: synthetic simulator\n"
            f"- Status: {record['status']}\n"
            f"- Estimated risk: {record['risk']:.0%}\n"
            f"- Prototype health score: {record['health_score']}/100\n"
            "- Health score definition: 100 × (1 − model-estimated risk); not measured equipment health\n"
            f"- Prototype RUL: {record['rul_hours']:.2f} simulated hours\n"
            f"- Operating time: {record['operating_hours']:.2f} simulated hours\n"
            f"- Scenario / fault mode: {record['scenario']} / {record['fault_mode']}\n"
            f"- Sensor quality: {sensor_health['status']} "
            f"({sensor_health['quality_pct']}%, {sensor_health['valid_channels']}/"
            f"{sensor_health['total_channels']} channels)\n\n"
            "## Risk trend\n\n"
            f"{risk_trend}\n\n"
            "## WHY — model evidence\n\n"
            f"{explanations}\n\n"
            "## Sensor quality checks\n\n"
            f"{issues}\n\n"
            "## AI vs fixed comparator\n\n"
            f"- Model: {record['status']} at {record['risk']:.0%} estimated risk\n"
            f"- Illustrative fixed alarm: {'ALARM' if comparison['fixed_alarm'] else 'no limit crossed'}\n"
            f"- Crossed limits: {', '.join(comparison['crossed_limits']) or 'none'}\n\n"
            "## Suggested prototype action\n\n"
            f"{recommendation}\n\n"
            "## Alert delivery history\n\n"
            f"{alert_lines}\n\n"
            "## Maintenance history\n\n"
            f"{maintenance_lines}\n\n"
            "## Limitations\n\n"
            "All readings and model estimates in this report are from a synthetic prototype. "
            "Health score, RUL, thresholds, and recommendations are not validated for real equipment "
            "and must not be used as safety instructions.\n"
        )

    def reset(self, machine_id: str) -> dict[str, Any]:
        simulator = self._simulator(machine_id)
        previous = self.latest(machine_id)
        simulator.reset()
        self.session_records[machine_id].clear()
        self._last_status[machine_id] = "NORMAL"
        recovered = self.advance(machine_id)
        self._record_event(
            machine_id,
            "RESET",
            previous,
            recovered,
            "Simulator reset to a healthy baseline.",
        )
        return recovered

    def advance(self, machine_id: str, samples: int = 1) -> dict[str, Any]:
        simulator = self._simulator(machine_id)
        if samples < 1 or samples > 500:
            raise ValueError("samples must be between 1 and 500")
        result: dict[str, Any] = {}
        for _ in range(samples):
            reading = simulator.step()
            self._revision[machine_id] += 1
            records = self.session_records[machine_id]
            records.append(reading.to_dict())
            analysis = self.pipeline.analyze(pd.DataFrame(records))
            result = self._enrich(reading.to_dict(), analysis)
            records[-1] = result
            self._append_csv(self.live_data_path, SENSOR_FIELDS, result)
            self._record_transition(result, analysis)
        return result

    def comparison(self, record: dict[str, Any]) -> dict[str, Any]:
        crossed = []
        if record["temperature"] >= FIXED_LIMITS["temperature"]:
            crossed.append("temperature limit")
        if record["vibration"] >= FIXED_LIMITS["vibration"]:
            crossed.append("vibration limit")
        if record["motor_current"] >= FIXED_LIMITS["motor_current"]:
            crossed.append("motor-current limit")
        if record["rpm"] <= FIXED_LIMITS["rpm"]:
            crossed.append("low-RPM limit")
        return {
            "fixed_alarm": bool(crossed),
            "crossed_limits": crossed,
            "limits": FIXED_LIMITS.copy(),
        }

    def compare_workload(
        self,
        machine_id: str,
        workload: float,
        samples: int = 10,
        rpm_setpoint: float | None = None,
    ) -> dict[str, Any]:
        """Compare paired future simulations without changing live machine state."""
        self._simulator(machine_id)
        MachineSimulator._validate_workload(workload)
        current_rpm = self.simulators[machine_id].rpm_setpoint
        alternate_rpm = current_rpm if rpm_setpoint is None else MachineSimulator._validate_rpm(rpm_setpoint)
        if samples < 1 or samples > 100:
            raise ValueError("samples must be between 1 and 100")

        history = self.records_for(machine_id)
        baseline_simulator = copy.deepcopy(self.simulators[machine_id])
        what_if_simulator = copy.deepcopy(self.simulators[machine_id])
        baseline_result = self._project(baseline_simulator, history, samples)
        what_if_simulator.set_workload(workload)
        what_if_simulator.set_rpm_setpoint(alternate_rpm)
        what_if_result = self._project(what_if_simulator, history, samples)

        return {
            "machine_id": machine_id,
            "current_workload": self.simulators[machine_id].workload,
            "what_if_workload": float(workload),
            "current_rpm_setpoint": current_rpm,
            "what_if_rpm_setpoint": alternate_rpm,
            "samples": samples,
            "revision": self._revision[machine_id],
            "baseline": baseline_result,
            "what_if": what_if_result,
            "risk_delta": round(what_if_result["risk"] - baseline_result["risk"], 4),
            "sensor_deltas": {
                sensor: round(
                    what_if_result[sensor] - baseline_result[sensor],
                    2,
                )
                for sensor in (
                    "temperature",
                    "vibration",
                    "motor_current",
                    "rpm",
                    "bearing_temperature_c",
                    "coolant_flow_lpm",
                    "lubrication_pressure_bar",
                    "acoustic_noise_db",
                )
            },
        }

    def advance_until_status(self, machine_id: str, target_status: str) -> dict[str, Any]:
        """Advance a live demonstration until a target risk state is reached."""
        targets = {"WARNING": WARNING_THRESHOLD, "CRITICAL": CRITICAL_THRESHOLD}
        if target_status not in targets:
            raise ValueError("target_status must be WARNING or CRITICAL")
        result = self.latest(machine_id)
        if result["risk"] >= targets[target_status]:
            return result
        simulator = self._simulator(machine_id)
        simulator.start_degradation()
        for _ in range(50):
            result = self.advance(machine_id)
            if result["risk"] >= targets[target_status]:
                return result
        raise RuntimeError(f"Simulation did not reach {target_status} within 50 samples")

    def perform_virtual_maintenance(
        self,
        machine_id: str,
        maintenance_type: str = "Recommended inspection and service",
        recovery_fraction: float = 0.85,
    ) -> dict[str, Any]:
        """Reduce simulated wear and verify the risk model for up to ten samples."""
        if not maintenance_type.strip():
            raise ValueError("maintenance_type cannot be empty")
        simulator = self._simulator(machine_id)
        before = self.latest(machine_id)
        remaining_wear = simulator.apply_maintenance(recovery_fraction)
        after = before
        verification_samples = 0
        for _ in range(10):
            after = self.advance(machine_id)
            verification_samples += 1
            if after["status"] == "NORMAL":
                break
        self._record_event(
            machine_id,
            "VIRTUAL_MAINTENANCE",
            before,
            after,
            (
                f"{maintenance_type}; wear reduced by {recovery_fraction:.0%}; "
                f"simulated residual wear {remaining_wear:.1%}."
            ),
        )
        history_row = {
            "timestamp": after["timestamp"],
            "machine_id": machine_id,
            "status": after["status"],
            "risk": after["risk"],
            "alert": "Virtual maintenance completed",
            "reason": maintenance_type,
            "recommendation": (
                f"Recovery verification: risk {before['risk']:.0%} → {after['risk']:.0%}; "
                f"status {before['status']} → {after['status']}."
            ),
        }
        self.maintenance_history.append(history_row)
        self._append_csv(self.maintenance_path, MAINTENANCE_FIELDS, history_row)
        if before["status"] != "NORMAL" and after["status"] == "NORMAL":
            self._record_alert(
                after,
                event_type="RECOVERED",
                message=(
                    f"Machine Recovered\n{machine_id} has returned to normal operating "
                    "conditions after maintenance."
                ),
                send_external=True,
            )
        return {
            "before": before,
            "after": after,
            "recovery_fraction": recovery_fraction,
            "recovered": after["status"] == "NORMAL",
            "risk_improved": after["risk"] < before["risk"],
            "verification_samples": verification_samples,
            "revision": self._revision[machine_id],
        }

    def maintenance_copilot(self, machine_id: str) -> dict[str, Any]:
        """Build a transparent, local recommendation from model output and trends."""
        record = self.latest(machine_id)
        history = self.records_for(machine_id)
        top_factors = record["explanations"][:3]
        reasons = [str(item["feature"]) for item in top_factors if float(item["contribution"]) > 0]
        if not reasons:
            reasons = ["No feature is above the learned healthy reference"]

        slopes: dict[str, float] = {}
        if len(history) >= 5:
            latest_features = self.pipeline.analyze(history).features.iloc[0]
            slopes = {
                name: float(latest_features[name])
                for name in ("temperature_slope_5", "vibration_slope_5", "current_slope_5")
            }

        if record["status"] == "CRITICAL":
            urgency = "Urgent: schedule an inspection before returning the simulated machine to service."
        elif record["status"] == "WARNING":
            urgency = "Plan a targeted inspection soon and continue monitoring the trend."
        else:
            urgency = "Continue routine monitoring; no simulated intervention is indicated now."

        return {
            "machine_id": machine_id,
            "status": record["status"],
            "risk": record["risk"],
            "urgency": urgency,
            "evidence": reasons,
            "trend_slopes": slopes,
            "recommendation": self.recommend_maintenance(top_factors),
            "rul_hours": record["rul_hours"],
            "disclaimer": (
                "Local rule-based maintenance guidance grounded in this prototype's model output; "
                "not an external LLM or real-equipment service instruction."
            ),
        }

    def answer_copilot_question(self, machine_id: str, question: str) -> str:
        """Answer common operator questions using this machine's computed evidence."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        copilot = self.maintenance_copilot(machine_id)
        normalized = question.lower()
        evidence = ", ".join(copilot["evidence"])
        if any(word in normalized for word in ("why", "reason", "cause")):
            return (
                f"{machine_id} is {copilot['status']} at {copilot['risk']:.0%} estimated risk. "
                f"Top model evidence: {evidence}."
            )
        if any(word in normalized for word in ("rul", "remaining", "last good", "when")):
            last_good = self.last_good_hour(machine_id)
            observed = (
                f"Last observed below-warning reading: {last_good['operating_hours']:.1f} simulated hours."
                if last_good
                else "A warning boundary has not yet been observed in this session."
            )
            return (
                f"Prototype RUL is {copilot['rul_hours']:.2f} simulated hours. {observed} "
                "This is simulator-based, not a real failure-time promise."
            )
        if any(
            word in normalized
            for word in (
                "inspect",
                "maintenance",
                "action",
                "fix",
                "solve",
                "recommend",
                "should i",
                "should we",
                "what should",
                "what can i do",
                "how can i",
                "next step",
                "do now",
                "help",
            )
        ):
            return f"{copilot['urgency']} Suggested action: {copilot['recommendation']}"
        if any(word in normalized for word in ("workload", "what if", "scenario")):
            return (
                "Use What-if analysis to compare paired baseline and alternate-workload projections. "
                "It changes no live machine state and reflects workload-adjusted sensor features."
            )
        return (
            f"{machine_id}: {copilot['status']} at {copilot['risk']:.0%} estimated risk; "
            f"anomaly score {self.latest(machine_id)['anomaly_score']:.0f}/100. "
            f"Evidence: {evidence}. {copilot['urgency']}"
        )

    def last_good_hour(self, machine_id: str) -> dict[str, Any] | None:
        """Return the last observed point before the model first entered warning."""
        records = self.session_records[machine_id]
        for index, record in enumerate(records):
            if record["risk"] >= WARNING_THRESHOLD:
                if index == 0:
                    return None
                previous = records[index - 1]
                return {
                    "timestamp": previous["timestamp"],
                    "operating_hours": previous["operating_hours"],
                    "first_warning_timestamp": record["timestamp"],
                }
        return None

    def evidence_timeline(self, machine_id: str) -> pd.DataFrame:
        """Combine live sensor evidence, threshold crossings, and service events."""
        if machine_id not in self.simulators:
            self._simulator(machine_id)
        events: list[dict[str, Any]] = []
        for record in self.session_records[machine_id]:
            fixed = self.comparison(record)
            events.append(
                {
                    "timestamp": record["timestamp"],
                    "operating_hours": record["operating_hours"],
                    "event": "Sensor reading",
                    "status": record["status"],
                    "risk": record["risk"],
                    "details": (
                        f"Vibration {record['vibration']:.2f}; "
                        f"temperature {record['temperature']:.1f}; "
                        f"fixed alarm {'yes' if fixed['fixed_alarm'] else 'no'}"
                    ),
                }
            )
        for event in self.event_timeline:
            if event["machine_id"] == machine_id:
                events.append(
                    {
                        "timestamp": event["timestamp"],
                        "operating_hours": None,
                        "event": event["event_type"].replace("_", " ").title(),
                        "status": f"{event['status_before']} → {event['status_after']}",
                        "risk": event["risk_after"],
                        "details": event["details"],
                    }
                )
        return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True) if events else pd.DataFrame()

    def observed_warning_lead_time(self, machine_id: str) -> float | None:
        """Measure risk-warning lead over fixed-limit alarm in this recorded trace."""
        records = self.session_records[machine_id]
        model_warning = next((row for row in records if row["risk"] >= WARNING_THRESHOLD), None)
        fixed_alarm = next((row for row in records if self.comparison(row)["fixed_alarm"]), None)
        if model_warning is None or fixed_alarm is None:
            return None
        return round(
            float(fixed_alarm["operating_hours"] - model_warning["operating_hours"]),
            2,
        )

    def _enrich(self, reading: dict[str, Any], analysis: MachineAnalysis) -> dict[str, Any]:
        risk = min(1.0, max(0.0, analysis.risk))
        return {
            **reading,
            "risk": risk,
            "health_score": round((1.0 - risk) * 100),
            "status": analysis.status,
            "anomaly_score": analysis.anomaly_score,
            "is_anomaly": analysis.is_anomaly,
            "rul_hours": analysis.rul_hours,
            "explanations": analysis.explanations,
        }

    def _record_transition(self, record: dict[str, Any], analysis: MachineAnalysis) -> None:
        previous = self._last_status[record["machine_id"]]
        self._last_status[record["machine_id"]] = record["status"]
        rank = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
        if rank[record["status"]] <= rank[previous]:
            return

        leading_reasons = analysis.explanations[:2]
        reason = " + ".join(str(item["feature"]).lower() for item in leading_reasons)
        recommendation = self.recommend_maintenance(leading_reasons)
        row = {
            "timestamp": record["timestamp"],
            "machine_id": record["machine_id"],
            "status": record["status"],
            "risk": record["risk"],
            "alert": "Maintenance recommended",
            "reason": reason or "Increasing simulated degradation",
            "recommendation": recommendation,
        }
        self.maintenance_history.append(row)
        self._append_csv(self.maintenance_path, MAINTENANCE_FIELDS, row)
        if record["status"] in {"WARNING", "CRITICAL"}:
            self._record_alert(
                record,
                event_type=record["status"],
                message=(
                    f"{record['status']}: {record['machine_id']} reached "
                    f"{record['risk']:.0%} estimated risk. {recommendation}"
                ),
                send_external=record["status"] == "CRITICAL",
            )

    @staticmethod
    def recommend_maintenance(explanations: list[dict[str, Any]]) -> str:
        features = " ".join(str(item["feature"]).lower() for item in explanations)
        if "lubrication pressure" in features:
            return "Inspect the lubrication reservoir, filter, lines, and pressure sensor."
        if "coolant" in features:
            return "Check coolant level, flow path, pump, and temperature before the next run."
        if "bearing" in features or "vibration" in features:
            return "Inspect spindle bearings, mounting, and alignment; confirm with a qualified technician."
        if "tool wear" in features or "feed rate" in features:
            return "Inspect the cutting tool and verify tool-wear measurements before adjusting the process."
        if "spindle load" in features or "torque" in features:
            return "Check the spindle load and cutting conditions against the approved operating recipe."
        if "vibration" in features:
            return "Inspect spindle and bearing condition; check for looseness or imbalance."
        if "motor current" in features:
            return "Inspect motor load, electrical connections, and drive condition."
        if "temperature" in features:
            return "Inspect cooling, lubrication, and temperature-sensitive components."
        if "rpm" in features:
            return "Inspect speed control and spindle drive stability."
        return "Inspect the machine and verify sensor readings before continued operation."

    def _read_maintenance_history(self) -> list[dict[str, Any]]:
        if not self.maintenance_path.exists():
            return []
        with self.maintenance_path.open(newline="", encoding="utf-8") as history_file:
            rows = list(csv.DictReader(history_file))
        for row in rows:
            row["risk"] = float(row["risk"])
        return rows

    def _record_alert(
        self,
        record: dict[str, Any],
        event_type: str,
        message: str,
        send_external: bool,
    ) -> dict[str, Any]:
        channels: dict[str, dict[str, str]] = {
            "email": {"status": "not_sent", "detail": "Only critical and recovered events are sent."},
            "whatsapp": {"status": "not_sent", "detail": "Only critical and recovered events are sent."},
        }
        if send_external:
            if self.notifier is None:
                channels = {
                    name: {"status": "not_configured", "detail": "Configure Streamlit notification secrets."}
                    for name in ("email", "whatsapp")
                }
            else:
                try:
                    channels = self.notifier.send(
                        {
                            "machine_id": str(record["machine_id"]),
                            "status": event_type,
                            "risk": float(record["risk"]),
                            "message": message,
                        }
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    channels = {
                        name: {
                            "status": "failed",
                            "detail": f"Notification dispatch failed ({type(error).__name__}).",
                        }
                        for name in ("email", "whatsapp")
                    }

        row = {
            "timestamp": str(record["timestamp"]),
            "machine_id": str(record["machine_id"]),
            "event_type": event_type,
            "status": str(record["status"]),
            "risk": float(record["risk"]),
            "message": message,
            "dashboard_status": "shown",
            "email_status": channels.get("email", {}).get("status", "not_sent"),
            "email_detail": channels.get("email", {}).get("detail", ""),
            "whatsapp_status": channels.get("whatsapp", {}).get("status", "not_sent"),
            "whatsapp_detail": channels.get("whatsapp", {}).get("detail", ""),
        }
        self.notification_history.append(row)
        self._append_csv(self.notification_path, NOTIFICATION_FIELDS, row)
        return row

    def send_test_notification(self, machine_id: str) -> dict[str, Any]:
        """Send an explicitly requested test message without changing machine state."""
        record = self.latest(self._simulator(machine_id).machine_id)
        return self._record_alert(
            record,
            event_type="TEST",
            message=f"Spindle Nadi test notification for {machine_id}. No machine state was changed.",
            send_external=True,
        )

    def _read_notification_history(self) -> list[dict[str, Any]]:
        if not self.notification_path.exists():
            return []
        with self.notification_path.open(newline="", encoding="utf-8") as notification_file:
            rows = list(csv.DictReader(notification_file))
        for row in rows:
            row["risk"] = float(row["risk"])
        return rows

    def _read_event_timeline(self) -> list[dict[str, Any]]:
        if not self.event_timeline_path.exists():
            return []
        with self.event_timeline_path.open(newline="", encoding="utf-8") as event_file:
            rows = list(csv.DictReader(event_file))
        for row in rows:
            row["risk_before"] = float(row["risk_before"])
            row["risk_after"] = float(row["risk_after"])
        return rows

    def _project(
        self,
        simulator: MachineSimulator,
        history: pd.DataFrame,
        samples: int,
    ) -> dict[str, Any]:
        projected = history.copy()
        latest: dict[str, Any] = {}
        for _ in range(samples):
            latest = simulator.step().to_dict()
            projected = pd.concat([projected, pd.DataFrame([latest])], ignore_index=True)
        analysis = self.pipeline.analyze(projected)
        risk = min(1.0, max(0.0, analysis.risk))
        return {
            **latest,
            "risk": risk,
            "health_score": round((1.0 - risk) * 100),
            "status": analysis.status,
            "anomaly_score": analysis.anomaly_score,
            "is_anomaly": analysis.is_anomaly,
            "rul_hours": analysis.rul_hours,
        }

    def _record_event(
        self,
        machine_id: str,
        event_type: str,
        before: dict[str, Any],
        after: dict[str, Any],
        details: str,
    ) -> None:
        row = {
            "timestamp": after["timestamp"],
            "machine_id": machine_id,
            "event_type": event_type,
            "status_before": before["status"],
            "risk_before": before["risk"],
            "status_after": after["status"],
            "risk_after": after["risk"],
            "details": details,
        }
        self.event_timeline.append(row)
        self._append_csv(self.event_timeline_path, EVENT_FIELDS, row)

    @staticmethod
    def _append_csv(path: Path, fields: list[str], record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def _simulator(self, machine_id: str) -> MachineSimulator:
        try:
            return self.simulators[machine_id]
        except KeyError as error:
            raise ValueError(f"Unknown machine ID: {machine_id}") from error
