import unittest

from simulator import MachineSimulator, Scenario


class MachineSimulatorTests(unittest.TestCase):
    def test_normal_machine_stays_healthy_with_sensor_noise(self) -> None:
        simulator = MachineSimulator(seed=7)

        readings = [simulator.step() for _ in range(5)]

        self.assertTrue(all(reading.status == "NORMAL" for reading in readings))
        self.assertTrue(all(reading.scenario == Scenario.NORMAL.value for reading in readings))
        self.assertGreater(len({reading.temperature for reading in readings}), 1)

    def test_gradual_degradation_progresses_through_warning_to_critical(self) -> None:
        simulator = MachineSimulator(seed=7)
        simulator.start_degradation()

        statuses = [simulator.step().status for _ in range(15)]

        self.assertIn("NORMAL", statuses)
        self.assertIn("WARNING", statuses)
        self.assertEqual(statuses[-1], "CRITICAL")
        self.assertEqual(statuses, sorted(statuses, key=["NORMAL", "WARNING", "CRITICAL"].index))
        self.assertEqual(statuses.index("WARNING"), 6)
        self.assertEqual(statuses.index("CRITICAL"), 13)

    def test_near_failure_starts_critical(self) -> None:
        simulator = MachineSimulator(seed=7)
        simulator.set_near_failure()

        reading = simulator.step()

        self.assertEqual(reading.status, "CRITICAL")
        self.assertEqual(reading.scenario, Scenario.NEAR_FAILURE.value)

    def test_reset_returns_to_healthy_state_and_zeroes_operating_hours(self) -> None:
        simulator = MachineSimulator(seed=7)
        simulator.start_degradation()
        simulator.step()
        simulator.reset()

        reading = simulator.step()

        self.assertEqual(reading.status, "NORMAL")
        self.assertEqual(reading.scenario, Scenario.NORMAL.value)
        self.assertEqual(reading.degradation, 0.0)
        self.assertEqual(reading.operating_hours, 0.1)

    def test_workload_changes_expected_sensor_levels(self) -> None:
        low_load = MachineSimulator(seed=7, workload=20).step()
        high_load = MachineSimulator(seed=7, workload=90).step()

        self.assertGreater(high_load.temperature, low_load.temperature)
        self.assertGreater(high_load.vibration, low_load.vibration)
        self.assertGreater(high_load.motor_current, low_load.motor_current)

    def test_invalid_workload_is_rejected(self) -> None:
        simulator = MachineSimulator(seed=7)

        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            simulator.set_workload(101)

    def test_reading_is_flattened_for_data_export(self) -> None:
        reading = MachineSimulator(seed=7).step()

        self.assertEqual(reading.to_dict()["machine_id"], "CNC-01")
        self.assertIn("timestamp", reading.to_dict())
        self.assertEqual(reading.to_dict()["status"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
