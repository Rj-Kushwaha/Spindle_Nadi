import unittest
from contextlib import AbstractContextManager
from urllib.parse import parse_qs

from backend.notifications import AlertNotifier


EVENT = {
    "machine_id": "CNC-01",
    "status": "CRITICAL",
    "risk": 0.92,
    "message": "CRITICAL: CNC-01 reached 92% estimated risk.",
}


class NotificationTests(unittest.TestCase):
    def test_unconfigured_channels_report_that_delivery_was_not_attempted(self) -> None:
        notifier = AlertNotifier()

        result = notifier.send(EVENT)

        self.assertEqual(result["email"]["status"], "not_configured")
        self.assertEqual(result["whatsapp"]["status"], "not_configured")

    def test_email_uses_tls_and_sends_message_to_configured_recipient(self) -> None:
        sent_messages = []

        class FakeSMTP(AbstractContextManager):
            def __init__(self, host, port, timeout):
                self.details = (host, port, timeout)

            def starttls(self):
                self.tls_started = True

            def login(self, username, password):
                self.credentials = (username, password)

            def send_message(self, message):
                sent_messages.append(message)

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                return False

        notifier = AlertNotifier(
            email={
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
                "sender": "alerts@example.com",
                "recipient": "operator@example.com",
                "username": "alerts",
                "password": "test-only",
                "use_starttls": True,
            },
            smtp_factory=FakeSMTP,
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["email"]["status"], "sent")
        self.assertEqual(sent_messages[0]["To"], "operator@example.com")
        self.assertIn("92%", sent_messages[0].get_content())
        self.assertEqual(result["whatsapp"]["status"], "not_configured")

    def test_email_supports_up_to_ten_recipients_without_exposing_addresses(self) -> None:
        delivered_to = []

        class FakeSMTP(AbstractContextManager):
            def __init__(self, host, port, timeout):
                del host, port, timeout

            def starttls(self):
                return None

            def send_message(self, message):
                delivered_to.append(message["To"])

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                return False

        recipients = [f"operator{index}@example.com" for index in range(10)]
        notifier = AlertNotifier(
            email={
                "smtp_host": "smtp.example.com",
                "sender": "alerts@example.com",
                "recipients": recipients,
                "use_starttls": True,
            },
            smtp_factory=FakeSMTP,
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["email"]["status"], "sent")
        self.assertEqual(result["email"]["detail"], "Email sent to 10/10 recipients")
        self.assertEqual(delivered_to, recipients)

    def test_email_recipient_limit_is_enforced_before_delivery(self) -> None:
        notifier = AlertNotifier(
            email={
                "smtp_host": "smtp.example.com",
                "sender": "alerts@example.com",
                "recipients": [f"operator{index}@example.com" for index in range(11)],
            }
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["email"]["status"], "failed")
        self.assertIn("no more than 10", result["email"]["detail"])

    def test_whatsapp_posts_only_to_twilio_with_e164_recipient(self) -> None:
        requests = []

        class FakeResponse(AbstractContextManager):
            status = 201

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                return False

        def fake_open(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        notifier = AlertNotifier(
            whatsapp={
                "account_sid": "AC" + "a" * 32,
                "auth_token": "test-token",
                "from_number": "+14155238886",
                "to_number": "+15551234567",
            },
            http_open=fake_open,
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["whatsapp"]["status"], "sent")
        request, timeout = requests[0]
        self.assertEqual(timeout, 10)
        self.assertEqual(request.full_url.split("/")[2], "api.twilio.com")
        form = parse_qs(request.data.decode("utf-8"))
        self.assertEqual(form["To"], ["whatsapp:+15551234567"])
        self.assertEqual(form["From"], ["whatsapp:+14155238886"])
        self.assertEqual(result["email"]["status"], "not_configured")

    def test_whatsapp_sends_one_message_per_configured_recipient(self) -> None:
        requests = []

        class FakeResponse(AbstractContextManager):
            status = 201

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                return False

        def fake_open(request, timeout):
            self.assertEqual(timeout, 10)
            requests.append(request)
            return FakeResponse()

        recipients = [f"+155512345{index:02d}" for index in range(10)]
        notifier = AlertNotifier(
            whatsapp={
                "account_sid": "AC" + "a" * 32,
                "auth_token": "test-token",
                "from_number": "+14155238886",
                "to_numbers": recipients,
            },
            http_open=fake_open,
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["whatsapp"]["status"], "sent")
        self.assertEqual(result["whatsapp"]["detail"], "WhatsApp sent to 10/10 recipients")
        self.assertEqual(len(requests), 10)
        self.assertEqual(
            [parse_qs(request.data.decode("utf-8"))["To"][0] for request in requests],
            [f"whatsapp:{recipient}" for recipient in recipients],
        )

    def test_invalid_whatsapp_number_is_rejected_before_network_request(self) -> None:
        notifier = AlertNotifier(
            whatsapp={
                "account_sid": "AC" + "a" * 32,
                "auth_token": "test-token",
                "from_number": "+14155238886",
                "to_number": "not-a-phone-number",
            }
        )

        result = notifier.send(EVENT)

        self.assertEqual(result["whatsapp"]["status"], "failed")
        self.assertIn("E.164", result["whatsapp"]["detail"])


if __name__ == "__main__":
    unittest.main()
