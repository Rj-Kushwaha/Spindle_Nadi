import unittest

from frontend.alert_messages import spoken_status_message


class SpokenStatusMessageTests(unittest.TestCase):
    def test_status_messages_are_bilingual_and_include_machine_and_risk(self) -> None:
        message = spoken_status_message("CRITICAL", "CNC-01", 0.91)

        self.assertIn("CNC-01", message)
        self.assertIn("CRITICAL", message)
        self.assertIn("91%", message)
        self.assertIn("Urgent maintenance attention is needed.", message)
        self.assertIn("मशीन की तुरंत जाँच आवश्यक है।", message)

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported machine status"):
            spoken_status_message("UNKNOWN", "CNC-01", 0.2)


if __name__ == "__main__":
    unittest.main()
