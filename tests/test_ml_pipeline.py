import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.generate_data import generate_dataset
from ml.features import FEATURE_COLUMNS, engineer_features
from ml.pipeline import SpindleNadiPipeline
from ml.prediction.risk_model import status_for_risk
from simulator import MachineSimulator


class FeatureEngineeringTests(unittest.TestCase):
    def test_features_include_workload_adjustments_and_temporal_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = generate_dataset(Path(directory) / "training.csv")
            second = generate_dataset(Path(directory) / "training-copy.csv")

        features = engineer_features(first)

        self.assertTrue(set(FEATURE_COLUMNS).issubset(features.columns))
        self.assertTrue(features[FEATURE_COLUMNS].notna().all().all())
        self.assertEqual(len(features), len(first))
        self.assertTrue((features["temperature_slope_5"] > 0).any())
        self.assertTrue(first.equals(second))
        self.assertEqual(
            set(first["scenario"]),
            {"NORMAL", "DEGRADATION", "NEAR_FAILURE"},
        )
        self.assertEqual(
            set(first["fault_mode"]) - {"NONE"},
            {"bearing_wear", "cooling_degradation", "lubrication_loss", "tool_wear", "spindle_imbalance"},
        )
        self.assertGreater(first["rpm_setpoint"].nunique(), 1)

    def test_healthy_operating_point_adjustments_normalize_rpm_changes(self) -> None:
        low_speed = MachineSimulator(seed=14, workload=60, rpm_setpoint=1800).step().to_dict()
        high_speed = MachineSimulator(seed=14, workload=60, rpm_setpoint=9000).step().to_dict()
        features = engineer_features(pd.DataFrame([low_speed, high_speed]))

        for feature in ("temperature_excess", "vibration_excess", "current_excess", "rpm_drop"):
            self.assertAlmostEqual(features.loc[0, feature], features.loc[1, feature], delta=0.05)

    def test_invalid_sensor_frame_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            engineer_features(pd.DataFrame({"temperature": [50.0]}))


class SpindleNadiPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.training_data = generate_dataset(Path(cls.temporary_directory.name) / "training.csv")
        cls.pipeline = SpindleNadiPipeline(cls.training_data)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_training_reports_group_held_out_measures(self) -> None:
        metrics = self.pipeline.metrics

        self.assertGreater(metrics["test_machines"], 0)
        self.assertGreater(metrics["test_samples"], 0)
        self.assertIn("mae", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)

    def test_model_risk_threshold_boundaries(self) -> None:
        self.assertEqual(status_for_risk(0.4999), "NORMAL")
        self.assertEqual(status_for_risk(0.50), "WARNING")
        self.assertEqual(status_for_risk(0.6499), "WARNING")
        self.assertEqual(status_for_risk(0.8499), "WARNING")
        self.assertEqual(status_for_risk(0.85), "CRITICAL")

    def test_model_risk_and_explanations_rise_with_degradation(self) -> None:
        simulator = MachineSimulator(seed=5)
        history: list[dict[str, str | float]] = []
        healthy = [simulator.step().to_dict() for _ in range(5)]
        history.extend(healthy)
        healthy_result = self.pipeline.analyze(pd.DataFrame(history))

        simulator.start_degradation()
        history.extend(simulator.step().to_dict() for _ in range(15))
        degraded_result = self.pipeline.analyze(pd.DataFrame(history))

        self.assertGreater(degraded_result.risk, healthy_result.risk)
        self.assertEqual(degraded_result.status, "CRITICAL")
        self.assertGreaterEqual(degraded_result.rul_hours, 0.0)
        self.assertEqual(len(degraded_result.explanations), 5)
        self.assertGreater(degraded_result.anomaly_score, healthy_result.anomaly_score)
        self.assertTrue(
            {"bearing_temperature_excess", "coolant_flow_drop", "lubrication_pressure_drop"}
            .issubset(FEATURE_COLUMNS)
        )


if __name__ == "__main__":
    unittest.main()
