import unittest
from io import BytesIO
import zipfile

from backend.esp32_serial import Esp32SerialReader, parse_esp32_line
import pandas as pd
from data.recorded_replay import (
    list_excel_sheets,
    list_supported_archive_members,
    parse_recorded_csv,
    parse_recorded_file,
    sensor_columns,
)


class RecordedReplayTests(unittest.TestCase):
    def test_parses_and_sorts_supported_timestamped_sensor_columns(self) -> None:
        contents = (
            b"time,machine,bearing_temp_c,coolant_flow_lpm\n"
            b"2025-01-02T00:00:00Z,CNC-B,55,6.2\n"
            b"2025-01-01T00:00:00Z,CNC-B,51,7.1\n"
        )

        readings = parse_recorded_csv(contents)

        self.assertEqual(len(readings), 2)
        self.assertEqual(readings.iloc[0]["machine_id"], "CNC-B")
        self.assertAlmostEqual(readings.iloc[0]["bearing_temperature_c"], 51.0)
        self.assertLess(readings.iloc[0]["timestamp"], readings.iloc[1]["timestamp"])

    def test_missing_timestamp_and_unknown_sensor_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "recognized numeric sensor"):
            parse_recorded_csv(b"operator_note\nok\n")

    def test_parses_timestamp_free_benchmark_in_source_row_order(self) -> None:
        contents = (
            b"Temperature_C,Current_A,RPM\n"
            b"53.2,4.1,1800\n"
            b"54.0,4.3,1795\n"
        )

        readings = parse_recorded_csv(contents)

        self.assertEqual(list(readings["sample_index"]), [1, 2])
        self.assertNotIn("timestamp", readings.columns)
        self.assertEqual(list(readings["temperature"]), [53.2, 54.0])
        self.assertEqual(list(readings["motor_current"]), [4.1, 4.3])
        self.assertEqual(list(readings["rpm"]), [1800, 1795])

    def test_date_alias_is_normalized_to_timestamp(self) -> None:
        readings = parse_recorded_csv(
            b"Date,temperature_c\n2025-01-02,55\n2025-01-01,51\n"
        )

        self.assertIn("timestamp", readings.columns)
        self.assertLess(readings.iloc[0]["timestamp"], readings.iloc[1]["timestamp"])

    def test_numeric_time_column_is_used_as_cycle_time_not_misread_as_date(self) -> None:
        readings = parse_recorded_csv(b"time,sensor_a\n1,12\n2,15\n")

        self.assertNotIn("timestamp", readings.columns)
        self.assertEqual(list(readings["replay_time"]), [1, 2])
        self.assertEqual(readings["time_source"].iloc[0], "cycle")

    def test_timestamp_free_csv_gets_explicit_synthetic_one_second_axis(self) -> None:
        readings = parse_recorded_csv(b"measurement_a,measurement_b\n2,5\n3,6\n")

        self.assertEqual(list(readings["replay_time_seconds"]), [0.0, 1.0])
        self.assertIn("synthetic", readings["time_source"].iloc[0])

    def test_headerless_csv_keeps_unknown_schema_generic(self) -> None:
        row = ",".join(str(number) for number in range(1, 27))
        readings = parse_recorded_file(f"{row}\n{row}\n".encode(), "Damage_Propagation_Modeling.csv")

        self.assertIn("measurement_1", readings.columns)
        self.assertIn("measurement_26", readings.columns)
        self.assertNotIn("engine_id", readings.columns)
        self.assertNotIn("sensor_1", readings.columns)

    def test_parses_headerless_cmapss_txt_without_mapping_sensor_semantics(self) -> None:
        rows = [
            " ".join(str(number) for number in range(1, 27)),
            " ".join(str(number) for number in range(1, 27)),
        ]
        readings = parse_recorded_file("\n".join(rows).encode(), "train_FD001.txt")

        self.assertEqual(list(readings["machine_id"]), ["1", "1"])
        self.assertEqual(list(readings["cycle"]), [2, 2])
        self.assertEqual(list(readings["replay_time"]), [2, 2])
        self.assertIn("sensor_1", readings.columns)
        self.assertIn("sensor_21", readings.columns)
        self.assertNotIn("temperature", readings.columns)
        self.assertIn("sensor_1", sensor_columns(readings))

    def test_parses_excel_workbook_and_lists_worksheets(self) -> None:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame({"cycle": [1, 2], "sensor_1": [3.0, 4.0]}).to_excel(
                writer, sheet_name="FD001", index=False
            )
            pd.DataFrame({"cycle": [1], "sensor_2": [8.0]}).to_excel(
                writer, sheet_name="FD002", index=False
            )
        contents = output.getvalue()

        self.assertEqual(list_excel_sheets(contents), ["FD001", "FD002"])
        readings = parse_recorded_file(contents, "benchmark.xlsx", sheet_name="FD002")
        self.assertEqual(list(readings["replay_time"]), [1])
        self.assertIn("sensor_2", readings.columns)

    def test_zip_data_member_can_be_selected_without_extraction(self) -> None:
        output = BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("nested/FD001.csv", b"cycle,sensor_1\n1,0.5\n2,0.7\n")
            archive.writestr("notes.pdf", b"%PDF-not-parsed")
        contents = output.getvalue()

        members = list_supported_archive_members(contents)
        self.assertEqual(members, ["nested/FD001.csv"])
        readings = parse_recorded_file(contents, "benchmark.zip", member_name=members[0])
        self.assertEqual(list(readings["replay_time"]), [1, 2])

    def test_invalid_timestamp_is_not_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid date/time"):
            parse_recorded_csv(b"timestamp,rpm\nnot-a-date,1500\n")


class Esp32SerialTests(unittest.TestCase):
    def test_parses_calibrated_serial_measurements_and_normalizes_metadata(self) -> None:
        record = parse_esp32_line(
            '{"timestamp":"2026-10-03T08:00:00Z","device_id":"ESP32-demo",'
            '"temperature_c":31.2,"vibration_x_mm_s":1.8,"current_a":1.2}'
        )

        self.assertEqual(record["machine_id"], "ESP32-demo")
        self.assertEqual(record["temperature"], 31.2)
        self.assertEqual(record["vibration_x"], 1.8)
        self.assertEqual(record["motor_current"], 1.2)
        self.assertEqual(record["data_source"], "ESP32 USB · physical telemetry")

    def test_rejects_bad_payloads_and_raw_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid JSON line"):
            parse_esp32_line("{")
        with self.assertRaisesRegex(ValueError, "at least one supported"):
            parse_esp32_line('{"operator_note":"hello"}')
        with self.assertRaisesRegex(ValueError, "finite number"):
            parse_esp32_line('{"rpm":NaN}')

    def test_serial_reader_buffers_partial_lines_and_reports_invalid_lines(self) -> None:
        class FakeSerial:
            def __init__(self) -> None:
                self.chunks = [
                    b'{"temperature_c":30',
                    b'.5}\nnot-json\n{"rpm":1200}\n',
                ]
                self.closed = False

            @property
            def in_waiting(self) -> int:
                return len(self.chunks[0]) if self.chunks else 0

            def read(self, size: int) -> bytes:
                chunk = self.chunks.pop(0)
                if size != len(chunk):
                    raise AssertionError("Reader did not request the available serial bytes")
                return chunk

            def close(self) -> None:
                self.closed = True

        fake = FakeSerial()

        def serial_factory(port: str, baudrate: int, timeout: float) -> FakeSerial:
            self.assertEqual(port, "/dev/test")
            self.assertEqual(baudrate, 115200)
            self.assertEqual(timeout, 0)
            return fake

        reader = Esp32SerialReader(
            "/dev/test",
            serial_factory=serial_factory,
        )
        self.assertEqual(reader.read_available(), ([], []))
        records, errors = reader.read_available()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["temperature"], 30.5)
        self.assertEqual(records[1]["rpm"], 1200.0)
        self.assertEqual(len(errors), 1)
        reader.close()
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
