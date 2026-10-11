"""Optional email and WhatsApp delivery for machine status alerts."""

from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import parseaddr
import re
import smtplib
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

AlertEvent = dict[str, str | float]
ChannelResult = dict[str, str]
SendMessage = Callable[[AlertEvent], ChannelResult]
_E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")


class AlertNotifier:
    """Send critical and recovery messages through configured channels."""

    def __init__(
        self,
        email: dict[str, Any] | None = None,
        whatsapp: dict[str, Any] | None = None,
        smtp_factory: Callable[..., Any] = smtplib.SMTP,
        smtp_ssl_factory: Callable[..., Any] = smtplib.SMTP_SSL,
        http_open: Callable[..., Any] = urlopen,
    ) -> None:
        self.email = email or {}
        self.whatsapp = whatsapp or {}
        self.smtp_factory = smtp_factory
        self.smtp_ssl_factory = smtp_ssl_factory
        self.http_open = http_open

    def send(self, event: AlertEvent) -> dict[str, ChannelResult]:
        """Attempt all configured channels and return explicit channel results."""
        return {
            "email": self._send_email(event),
            "whatsapp": self._send_whatsapp(event),
        }

    def configuration_status(self) -> dict[str, bool]:
        """Report whether the minimum settings for each channel are present."""
        return {
            "email": all(str(self.email.get(name, "")).strip() for name in ("smtp_host", "sender"))
            and bool(_recipient_values(self.email, "recipients", "recipient")),
            "whatsapp": all(
                str(self.whatsapp.get(name, "")).strip()
                for name in ("account_sid", "auth_token", "from_number")
            )
            and bool(_recipient_values(self.whatsapp, "to_numbers", "to_number")),
        }

    def _send_email(self, event: AlertEvent) -> ChannelResult:
        recipients = _recipient_values(self.email, "recipients", "recipient")
        missing = [name for name in ("smtp_host", "sender") if not str(self.email.get(name, "")).strip()]
        if not recipients:
            missing.append("recipients")
        if missing:
            return {
                "status": "not_configured",
                "detail": f"Add email setting(s): {', '.join(missing)}",
            }
        sender = str(self.email["sender"]).strip()
        if len(recipients) > 10:
            return {"status": "failed", "detail": "Configure no more than 10 email recipients"}
        if not _is_email_address(sender) or any(not _is_email_address(address) for address in recipients):
            return {"status": "failed", "detail": "Email sender or one or more recipients are invalid"}

        message = EmailMessage()
        message["Subject"] = f"Spindle Nadi {event['status']}: {event['machine_id']}"
        message["From"] = sender
        message.set_content(str(event["message"]))

        host = str(self.email["smtp_host"]).strip()
        sent_count = 0
        failures = []
        try:
            port = int(self.email.get("smtp_port", 587))
            if not 1 <= port <= 65535:
                return {"status": "failed", "detail": "SMTP port must be between 1 and 65535"}
            timeout = 10
            username = str(self.email.get("username", "")).strip()
            password = str(self.email.get("password", ""))
            use_ssl = _as_bool(self.email.get("use_ssl", False))
            use_starttls = _as_bool(self.email.get("use_starttls", not use_ssl))
        except (ValueError, TypeError):
            return {"status": "failed", "detail": "SMTP port must be a number between 1 and 65535"}
        for recipient in recipients:
            recipient_message = EmailMessage()
            recipient_message["Subject"] = message["Subject"]
            recipient_message["From"] = sender
            recipient_message["To"] = recipient
            recipient_message.set_content(str(event["message"]))
            factory = self.smtp_ssl_factory if use_ssl else self.smtp_factory
            try:
                with factory(host, port, timeout=timeout) as client:
                    if not use_ssl and use_starttls:
                        client.starttls()
                    if username:
                        client.login(username, password)
                    client.send_message(recipient_message)
                sent_count += 1
            except (OSError, smtplib.SMTPException, ValueError) as error:
                failures.append(type(error).__name__)
        status = _delivery_status(sent_count, len(recipients))
        detail = f"Email sent to {sent_count}/{len(recipients)} recipients"
        if failures:
            detail += f"; delivery error ({failures[0]})"
        return {"status": status, "detail": detail}

    def _send_whatsapp(self, event: AlertEvent) -> ChannelResult:
        required = ("account_sid", "auth_token", "from_number")
        recipients = _recipient_values(self.whatsapp, "to_numbers", "to_number")
        missing = [name for name in required if not str(self.whatsapp.get(name, "")).strip()]
        if not recipients:
            missing.append("to_numbers")
        if missing:
            return {
                "status": "not_configured",
                "detail": f"Add WhatsApp setting(s): {', '.join(missing)}",
            }

        sender = _whatsapp_address(str(self.whatsapp["from_number"]))
        normalized_recipients = [_whatsapp_address(recipient) for recipient in recipients]
        if sender is None or any(recipient is None for recipient in normalized_recipients):
            return {
                "status": "failed",
                "detail": "WhatsApp sender and all recipients must be E.164 numbers such as +15551234567",
            }
        if len(normalized_recipients) > 10:
            return {"status": "failed", "detail": "Configure no more than 10 WhatsApp recipients"}

        account_sid = str(self.whatsapp["account_sid"]).strip()
        auth_token = str(self.whatsapp["auth_token"])
        if not re.fullmatch(r"AC[0-9a-fA-F]{32}", account_sid):
            return {"status": "failed", "detail": "Twilio account SID format is invalid"}

        sent_count = 0
        failures = []
        for recipient in normalized_recipients:
            body = urlencode(
                {
                    "From": f"whatsapp:{sender}",
                    "To": f"whatsapp:{recipient}",
                    "Body": str(event["message"]),
                }
            ).encode("utf-8")
            credentials = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
            request = Request(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
                data=body,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )
            try:
                with self.http_open(request, timeout=10) as response:
                    if not 200 <= response.status < 300:
                        failures.append(f"HTTP {response.status}")
                    else:
                        sent_count += 1
            except HTTPError as error:
                failures.append(f"HTTP {error.code}")
            except (URLError, OSError, ValueError) as error:
                failures.append(type(error).__name__)
        status = _delivery_status(sent_count, len(normalized_recipients))
        detail = f"WhatsApp sent to {sent_count}/{len(normalized_recipients)} recipients"
        if failures:
            detail += f"; delivery error ({failures[0]})"
        return {"status": status, "detail": detail}


def notifier_from_streamlit_secrets(secrets: Any) -> AlertNotifier:
    """Build a notifier from the optional [notifications.email/whatsapp] sections."""
    if hasattr(secrets, "to_dict"):
        secrets = secrets.to_dict()
    notifications = secrets.get("notifications", {})
    if not isinstance(notifications, dict):
        raise ValueError("Streamlit secrets [notifications] must be a table")
    email = notifications.get("email", {})
    whatsapp = notifications.get("whatsapp", {})
    if not isinstance(email, dict) or not isinstance(whatsapp, dict):
        raise ValueError("Streamlit secrets email and whatsapp settings must be tables")
    return AlertNotifier(email=email, whatsapp=whatsapp)


def _is_email_address(value: str) -> bool:
    _, address = parseaddr(value)
    return bool(address and "@" in address and not any(character.isspace() for character in address))


def _recipient_values(settings: dict[str, Any], list_key: str, legacy_key: str) -> list[str]:
    raw_values = settings.get(list_key, settings.get(legacy_key, []))
    if isinstance(raw_values, str):
        values = raw_values.split(",")
    elif isinstance(raw_values, (list, tuple)):
        values = raw_values
    elif raw_values in (None, ""):
        values = []
    else:
        return [""]
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _delivery_status(sent_count: int, total_count: int) -> str:
    if sent_count == total_count:
        return "sent"
    if sent_count:
        return "partially_sent"
    return "failed"


def _whatsapp_address(value: str) -> str | None:
    normalized = value.strip()
    if normalized.startswith("whatsapp:"):
        normalized = normalized.removeprefix("whatsapp:")
    return normalized if _E164_PATTERN.fullmatch(normalized) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
