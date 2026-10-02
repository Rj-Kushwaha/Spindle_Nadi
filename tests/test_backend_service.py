import tempfile
import unittest
from pathlib import Path

from backend.service import SpindleNadiService
from data.generate_data import generate_dataset


class SpindleNadiServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        training = root / "training.csv"
        generate_dataset(training)
        self.service = SpindleNadiService(
            live_data_path=root / "live.csv",
            maintenance_path=root / "maintenance.csv",
            event_timeline_path=root / "events.csv",
            training_path=training,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_fleet_starts_with_four_healthy_machines(self) -> None:
        fleet = self.service.fleet_summary()

        self.assertEqual(len(fleet), 4)
        self.assertTrue(all(machine["status"] == "NORMAL" for machine in fleet))

    def test_degradation_generates_risk_transition_and_durable_alert(self) -> None:
        self.service.start_degradation("CNC-01")
        for _ in range(16):
            record = self.service.advance("CNC-01")

        self.assertEqual(record["status"], "CRITICAL")
        self.assertGreaterEqual(record["risk"], 0.8)
        self.assertTrue(self.service.maintenance_history)
        self.assertTrue(Path(self.service.maintenance_path).exists())
        self.assertIn("recommendation", self.service.maintenance_history[-1])
        self.assertTrue(Path(self.service.live_data_path).exists())

    def test_persisted_alert_risk_is_loaded_as_numeric_data(self) -> None:
        self.service.start_degradation("CNC-01")
        self.service.advance("CNC-01", 16)
        reloaded = SpindleNadiService(
            live_data_path=self.service.live_data_path,
            maintenance_path=self.service.maintenance_path,
            event_timeline_path=self.service.event_timeline_path,
            training_path=Path(self.temporary_directory.name) / "training.csv",
            pipeline=self.service.pipeline,
        )

        self.assertIsInstance(reloaded.maintenance_history[0]["risk"], float)

    def test_reset_clears_session_trend_and_returns_to_normal(self) -> None:
        self.service.start_degradation("CNC-01")
        self.service.advance("CNC-01", 8)

        record = self.service.reset("CNC-01")

        self.assertEqual(record["status"], "NORMAL")
        self.assertEqual(record["operating_hours"], 0.1)
        self.assertEqual(len(self.service.records_for("CNC-01")), 1)

    def test_workload_validation_and_fixed_limit_comparator(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            self.service.set_workload("CNC-01", 101)

        comparison = self.service.comparison(self.service.latest("CNC-01"))

        self.assertFalse(comparison["fixed_alarm"])
        self.assertIn("vibration", comparison["limits"])

    def test_unknown_machine_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown machine ID"):
            self.service.advance("NO-SUCH-MACHINE")

    def test_what_if_projects_workload_without_mutating_live_machine(self) -> None:
        simulator = self.service.simulators["CNC-01"]
        before = self.service.latest("CNC-01").copy()
        workload_before = simulator.workload
        hours_before = simulator.operating_hours

        projection = self.service.compare_workload("CNC-01", workload=90, samples=10)

        self.assertGreater(projection["what_if"]["temperature"], projection["baseline"]["temperature"])
        self.assertAlmostEqual(projection["what_if"]["risk"], projection["baseline"]["risk"], delta=0.03)
        self.assertEqual(simulator.workload, workload_before)
        self.assertEqual(simulator.operating_hours, hours_before)
        self.assertEqual(self.service.latest("CNC-01"), before)

    def test_virtual_maintenance_records_recovery_and_verification(self) -> None:
        self.service.start_degradation("CNC-01")
        self.service.advance("CNC-01", 16)
        before_hours = self.service.simulators["CNC-01"].operating_hours

        outcome = self.service.perform_virtual_maintenance("CNC-01", "Bearing service")

        self.assertLess(outcome["after"]["risk"], outcome["before"]["risk"])
        self.assertTrue(outcome["recovered"])
        self.assertEqual(
            self.service.simulators["CNC-01"].operating_hours,
            before_hours + self.service.simulators["CNC-01"].hours_per_step,
        )
        self.assertEqual(self.service.event_timeline[-1]["event_type"], "VIRTUAL_MAINTENANCE")
        self.assertIn("Recovery verification", self.service.maintenance_history[-1]["recommendation"])

    def test_copilot_explains_model_output_and_gives_recommendation(self) -> None:
        copilot = self.service.maintenance_copilot("CNC-01")

        self.assertEqual(copilot["machine_id"], "CNC-01")
        self.assertTrue(copilot["urgency"])
        self.assertTrue(copilot["evidence"])
        self.assertTrue(copilot["recommendation"])
        self.assertIn("not an external LLM", copilot["disclaimer"])

    def test_priority_board_puts_critical_machine_first(self) -> None:
        self.service.set_near_failure("CNC-01")
        self.service.advance("CNC-01", 3)

        prioritized = self.service.fleet_summary()

        self.assertEqual(prioritized[0]["machine_id"], "CNC-01")
        self.assertEqual(prioritized[0]["status"], "CRITICAL")

    def test_timeline_includes_virtual_maintenance_event(self) -> None:
        self.service.start_degradation("CNC-01")
        self.service.advance("CNC-01", 16)
        self.service.perform_virtual_maintenance("CNC-01", "Bearing service")

        timeline = self.service.evidence_timeline("CNC-01")

        self.assertIn("Virtual Maintenance", set(timeline["event"]))
        self.assertIn("Sensor reading", set(timeline["event"]))

    def test_warning_lead_time_is_measured_only_after_both_events_occur(self) -> None:
        self.assertIsNone(self.service.observed_warning_lead_time("CNC-01"))
        self.service.start_degradation("CNC-01")
        self.service.advance("CNC-01", 30)

        lead_time = self.service.observed_warning_lead_time("CNC-01")

        self.assertIsNotNone(lead_time)

    def test_demo_shortcut_advances_from_warning_to_critical(self) -> None:
        warning = self.service.advance_until_status("CNC-01", "WARNING")
        critical = self.service.advance_until_status("CNC-01", "CRITICAL")

        self.assertGreaterEqual(warning["risk"], 0.4)
        self.assertGreaterEqual(critical["risk"], 0.8)
        self.assertGreater(critical["operating_hours"], warning["operating_hours"])

    def test_copilot_answers_are_grounded_in_current_machine_state(self) -> None:
        answer = self.service.answer_copilot_question("CNC-01", "Why is this machine at risk?")

        self.assertIn("CNC-01", answer)
        self.assertIn("estimated risk", answer)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.service.answer_copilot_question("CNC-01", " ")


if __name__ == "__main__":
    unittest.main()
