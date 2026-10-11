import unittest

from simulator import FAULT_MODES, MachineSimulator, Scenario


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
        self.assertEqual(statuses.index("WARNING"), 8)
        self.assertEqual(statuses.index("CRITICAL"), 14)

    def test_simulator_state_threshold_boundaries(self) -> None:
        self.assertEqual(MachineSimulator._status_for(0.499), "NORMAL")
        self.assertEqual(MachineSimulator._status_for(0.50), "WARNING")
        self.assertEqual(MachineSimulator._status_for(0.649), "WARNING")
        self.assertEqual(MachineSimulator._status_for(0.849), "WARNING")
        self.assertEqual(MachineSimulator._status_for(0.85), "CRITICAL")

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

    def test_rpm_setpoint_changes_generated_speed_and_resets_to_initial_value(self) -> None:
        simulator = MachineSimulator(seed=7, rpm_setpoint=6000)
        first = simulator.step()
        simulator.set_rpm_setpoint(9000)
        second = simulator.step()

        self.assertEqual(first.rpm_setpoint, 6000)
        self.assertEqual(second.rpm_setpoint, 9000)
        self.assertGreater(second.rpm, first.rpm)
        self.assertGreater(second.temperature, first.temperature)
        simulator.reset()
        self.assertEqual(simulator.step().rpm_setpoint, 6000)

    def test_invalid_rpm_setpoint_is_rejected(self) -> None:
        simulator = MachineSimulator(seed=7)

        with self.assertRaisesRegex(ValueError, "between 500 and 30000"):
            simulator.set_rpm_setpoint(30_001)

    def test_reading_is_flattened_for_data_export(self) -> None:
        reading = MachineSimulator(seed=7).step()

        self.assertEqual(reading.to_dict()["machine_id"], "CNC-01")
        self.assertIn("timestamp", reading.to_dict())
        self.assertEqual(reading.to_dict()["status"], "NORMAL")
        self.assertIn("vibration_x", reading.to_dict())
        self.assertIn("lubrication_pressure_bar", reading.to_dict())
        self.assertEqual(reading.fault_mode, "NONE")

    def test_simulated_fault_modes_have_distinct_sensor_signatures(self) -> None:
        baselines = {
            mode: MachineSimulator(seed=7).step().to_dict()
            for mode in FAULT_MODES
        }
        outcomes = {}
        for mode in FAULT_MODES:
            simulator = MachineSimulator(seed=7)
            simulator.start_degradation(mode)
            outcomes[mode] = [simulator.step().to_dict() for _ in range(15)][-1]

        self.assertGreater(
            outcomes["bearing_wear"]["bearing_temperature_c"],
            baselines["bearing_wear"]["bearing_temperature_c"],
        )
        self.assertLess(
            outcomes["cooling_degradation"]["coolant_flow_lpm"],
            baselines["cooling_degradation"]["coolant_flow_lpm"],
        )
        self.assertLess(
            outcomes["lubrication_loss"]["lubrication_pressure_bar"],
            baselines["lubrication_loss"]["lubrication_pressure_bar"],
        )
        self.assertGreater(outcomes["tool_wear"]["tool_wear_pct_est"], 85.0)
        self.assertGreater(outcomes["spindle_imbalance"]["vibration_x"], 4.0)

    def test_invalid_fault_mode_is_rejected(self) -> None:
        simulator = MachineSimulator(seed=7)

        with self.assertRaisesRegex(ValueError, "fault_mode must be one of"):
            simulator.start_degradation("unknown_fault")


if __name__ == "__main__":
    unittest.main()
