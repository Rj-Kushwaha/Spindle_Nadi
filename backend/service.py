"""Connect simulators, model analysis, alerts, and local CSV history."""

from __future__ import annotations

import csv
import copy
from pathlib import Path
from typing import Any

import pandas as pd

from data.generate_data import DEFAULT_OUTPUT, generate_dataset
from ml.pipeline import MachineAnalysis, SpindleNadiPipeline
from simulator import MachineSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "generated" / "live_sensor_readings.csv"
MAINTENANCE_PATH = PROJECT_ROOT / "data" / "generated" / "maintenance_history.csv"
EVENT_TIMELINE_PATH = PROJECT_ROOT / "data" / "generated" / "event_timeline.csv"

FIXED_LIMITS = {
    "temperature": 75.0,
    "vibration": 3.0,
    "motor_current": 12.5,
    "rpm": 1600.0,
}

SENSOR_FIELDS = [
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "motor_current",
    "rpm",
    "operating_hours",
    "workload",
    "scenario",
    "degradation",
    "risk",
    "status",
    "anomaly_score",
    "is_anomaly",
    "rul_hours",
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


class SpindleNadiService:
    """Stateful fleet service used by the dashboard and integration tests."""

    def __init__(
        self,
        live_data_path: str | Path = LIVE_DATA_PATH,
        maintenance_path: str | Path = MAINTENANCE_PATH,
        event_timeline_path: str | Path = EVENT_TIMELINE_PATH,
        training_path: str | Path = DEFAULT_OUTPUT,
        pipeline: SpindleNadiPipeline | None = None,
    ) -> None:
        self.live_data_path = Path(live_data_path)
        self.maintenance_path = Path(maintenance_path)
        self.event_timeline_path = Path(event_timeline_path)
        training_file = Path(training_path)
        if training_file.exists():
            training_data = pd.read_csv(training_file)
        else:
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

    def start_degradation(self, machine_id: str) -> None:
        self._simulator(machine_id).start_degradation()

    def set_near_failure(self, machine_id: str) -> None:
        self._simulator(machine_id).set_near_failure()

    def set_normal(self, machine_id: str) -> None:
        self._simulator(machine_id).set_normal()

    def set_workload(self, machine_id: str, workload: float) -> None:
        self._simulator(machine_id).set_workload(workload)

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
    ) -> dict[str, Any]:
        """Compare paired future simulations without changing live machine state."""
        self._simulator(machine_id)
        MachineSimulator._validate_workload(workload)
        if samples < 1 or samples > 100:
            raise ValueError("samples must be between 1 and 100")

        history = self.records_for(machine_id)
        baseline_simulator = copy.deepcopy(self.simulators[machine_id])
        what_if_simulator = copy.deepcopy(self.simulators[machine_id])
        baseline_result = self._project(baseline_simulator, history, samples)
        what_if_simulator.set_workload(workload)
        what_if_result = self._project(what_if_simulator, history, samples)

        return {
            "machine_id": machine_id,
            "current_workload": self.simulators[machine_id].workload,
            "what_if_workload": float(workload),
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
                for sensor in ("temperature", "vibration", "motor_current", "rpm")
            },
        }

    def advance_until_status(self, machine_id: str, target_status: str) -> dict[str, Any]:
        """Advance a live demonstration until a target risk state is reached."""
        targets = {"WARNING": 0.40, "CRITICAL": 0.80}
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
        """Apply simulated wear recovery and record before/after verification."""
        if not maintenance_type.strip():
            raise ValueError("maintenance_type cannot be empty")
        simulator = self._simulator(machine_id)
        before = self.latest(machine_id)
        remaining_wear = simulator.apply_maintenance(recovery_fraction)
        after = self.advance(machine_id)
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
        return {
            "before": before,
            "after": after,
            "recovery_fraction": recovery_fraction,
            "recovered": after["risk"] < before["risk"],
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
        if any(word in normalized for word in ("inspect", "maintenance", "action", "fix")):
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
            if record["risk"] >= 0.40:
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
        model_warning = next((row for row in records if row["risk"] >= 0.40), None)
        fixed_alarm = next((row for row in records if self.comparison(row)["fixed_alarm"]), None)
        if model_warning is None or fixed_alarm is None:
            return None
        return round(
            float(fixed_alarm["operating_hours"] - model_warning["operating_hours"]),
            2,
        )

    def _enrich(self, reading: dict[str, Any], analysis: MachineAnalysis) -> dict[str, Any]:
        return {
            **reading,
            "risk": analysis.risk,
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

    @staticmethod
    def recommend_maintenance(explanations: list[dict[str, Any]]) -> str:
        features = " ".join(str(item["feature"]).lower() for item in explanations)
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
        return {
            **latest,
            "risk": analysis.risk,
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
