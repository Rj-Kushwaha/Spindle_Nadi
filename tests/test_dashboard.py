import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from backend.service import SpindleNadiService
from data.generate_data import generate_dataset
from ml.pipeline import SpindleNadiPipeline


class DashboardInteractionTests(unittest.TestCase):
    def test_what_if_copilot_and_virtual_repair_work_through_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = generate_dataset(root / "training.csv")
            pipeline = SpindleNadiPipeline(training)
            service = SpindleNadiService(
                live_data_path=root / "live.csv",
                maintenance_path=root / "maintenance.csv",
                event_timeline_path=root / "events.csv",
                training_path=root / "training.csv",
                pipeline=pipeline,
            )
            original_workload = service.simulators["CNC-01"].workload

            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.session_state["service"] = service
            app.session_state["selected_machine"] = "CNC-01"
            app.run(timeout=60)

            self.assertFalse(app.exception)
            self.assertIn("What-if", [tab.label for tab in app.tabs])
            self.assertIn("Digital twin", [tab.label for tab in app.tabs])
            self.assertIn("Evidence timeline", [tab.label for tab in app.tabs])

            next(button for button in app.button if button.label == "Run paired what-if").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertIn("what_if_result_CNC-01", app.session_state)
            self.assertEqual(service.simulators["CNC-01"].workload, original_workload)

            app.text_input(key="copilot_question_CNC-01").set_value("Why is this machine at risk?").run(timeout=60)
            next(button for button in app.button if button.label == "Ask copilot").click().run(timeout=60)
            self.assertFalse(app.exception)
            self.assertIn("CNC-01", app.session_state["copilot_answer_CNC-01"])

            service.start_degradation("CNC-01")
            service.advance("CNC-01", 16)
            app.run(timeout=60)
            next(
                button for button in app.button
                if button.label == "Perform virtual maintenance"
            ).click().run(timeout=60)
            self.assertFalse(app.exception)
            result = app.session_state["maintenance_outcome_CNC-01"]
            self.assertLess(result["after"]["risk"], result["before"]["risk"])
            self.assertEqual(service.event_timeline[-1]["event_type"], "VIRTUAL_MAINTENANCE")


if __name__ == "__main__":
    unittest.main()
