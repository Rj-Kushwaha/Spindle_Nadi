# Email, WhatsApp, and dashboard alerts

## Alert behavior

- **NORMAL:** no message.
- **WARNING:** dashboard warning and a row in Alert history; no email or WhatsApp.
- **CRITICAL:** dashboard alert, then email and WhatsApp attempts on configured
  channels.
- **RECOVERED:** after virtual maintenance, the app advances up to ten
  post-maintenance samples. Only if the model state returns to NORMAL does the
  dashboard record recovery and attempt configured channels. Partial
  improvement without NORMAL is not announced as recovery.
- **TEST:** a message is sent only after an operator explicitly clicks **Send
  test notification**. Provider charges may apply.

Every event is written to the local
`data/generated/notification_history.csv` file and shown in **Alerts & messages**.
Each provider result is recorded as `sent`, `partially_sent`, `failed`, or
`not_configured`; a provider failure does not erase the dashboard alert. Local
CSV history is not durable on Streamlit Community Cloud restarts.

## Local Streamlit setup

Create `.streamlit/secrets.toml` locally (this path is ignored by Git) and add
the settings for the channel(s) you want. Do not paste credentials into source
code, chat, screenshots, or a public repository.

```toml
[notifications.email]
smtp_host = "smtp.example.com"
smtp_port = 587
sender = "spindle-nadi@example.com"
recipients = [
  "operator@example.com",
  "maintenance@example.com",
]
username = "smtp-account"
password = "use-a-provider-app-password"
use_ssl = false
use_starttls = true

[notifications.whatsapp]
account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token = "your-twilio-auth-token"
from_number = "+14155238886"
to_numbers = [
  "+15551234567",
  "+15557654321",
]
```

The values above are placeholders, not working credentials. Configure only
channels for which you have a legitimate sender account and recipient consent.
Configure up to 10 email recipients with the `recipients` TOML array and up to
10 WhatsApp recipients with the `to_numbers` array. Email addresses and phone
numbers must be valid; WhatsApp recipients must use E.164 format with a leading
`+`. Each recipient receives a separate message, and alert history reports the
number of successful deliveries. A `partially_sent` result means at least one
recipient succeeded and one or more failed. The older single-recipient
`recipient` and `to_number` settings remain supported.
For SMTP, ask your email provider for the correct host, port, and TLS mode; use
an app password where supported rather than your normal account password. SMTP
credentials are optional if the server allows sending without authentication.
For a Gmail sender with two-step verification enabled, the usual settings are
`smtp.gmail.com`, port `587`, STARTTLS enabled, the Gmail address as both
`sender` and `username`, and a Google App Password as `password`. Do not use
the normal Google account password. Provider requirements can change, so check
the provider's current SMTP/app-password guidance.

WhatsApp delivery uses the Twilio Messages API. Configure a Twilio WhatsApp
sender (the sandbox is suitable only for opted-in development recipients),
store the Account SID and Auth Token as secrets, and use E.164 phone numbers
with a leading `+`. Twilio may require the recipient to opt in and may charge
for messages. Check Twilio's current sandbox, template, and production rules
for your account before a demo. This prototype sends plain-text status messages
only; it does not initiate phone calls or send control commands.
For a sandbox demo, follow Twilio's WhatsApp sandbox enrollment instructions
for the recipient number, then use the sandbox's displayed WhatsApp sender
number and your enrolled recipient in the settings.

Restart the local Streamlit app after editing secrets, open **Alerts & messages**,
and confirm the desired channel is marked **Configured**. Use **Send test
notification** to deliberately send a real test to the configured recipients.
The test event and both delivery results are added to Alert history. If a
channel is not configured, the dashboard still records every relevant
transition and reports that delivery was not configured.

## Streamlit Community Cloud

In the app's **Settings → Secrets**, add the same TOML structure shown above.
Do not upload a secrets file to GitHub. Redeploy/restart the app, confirm the
channel configuration status, and only then choose **Send test notification**.

## Limits and operational safety

The default demo runs on a developer's computer. Its process must be running
and able to reach the SMTP provider or Twilio for delivery. Streamlit reruns
and session lifetime are not a reliable industrial alarm service; this
prototype has no queue, retry scheduler, delivery guarantee, user
acknowledgment, or on-call escalation. Local CSV history can be lost on
deployment restart. Use an approved durable notification service, database,
and monitoring process before any real operational reliance.

Warnings are intentionally dashboard-only. The app sends external notifications
for a rising CRITICAL state and for a verified virtual-maintenance recovery to
NORMAL; it does not send a recovery message merely because the simulator was
reset. All simulated status/risk values are demonstration outputs, not
real-machine safety signals.
