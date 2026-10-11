# ESP32 USB telemetry setup

Spindle Nadi can display physical ESP32 measurements over USB serial. It does
not control the board or machine, combine physical values with the synthetic
model, or report an AI failure risk for this stream.

## Device message contract

Send one UTF-8 JSON object per line, terminated with `\n`, at the same baud
rate selected in the dashboard (default `115200`). Send calibrated engineering
units, not uncalibrated ADC counts:

```json
{"timestamp":"2026-10-03T08:00:00Z","device_id":"ESP32-demo","temperature_c":31.2,"vibration_x_mm_s":1.8,"current_a":1.2}
```

Supported channels and accepted JSON keys:

| Dashboard channel | Accepted key(s) | Unit |
|---|---|---|
| Temperature | `temperature`, `temperature_c`, `temp_c` | °C |
| Overall vibration | `vibration`, `vibration_mm_s` | mm/s |
| Vibration axes | `vibration_x`, `vibration_x_mm_s` (and y/z) | mm/s |
| Motor current | `motor_current`, `current_a` | A |
| Spindle speed | `rpm`, `spindle_speed_rpm` | RPM |

`timestamp` is optional (UTC receive time is used when omitted). Use
`machine_id` or `device_id` to identify the demo device. Unknown keys are
ignored. Messages with malformed JSON, non-finite sensor values, or no
recognized channels are skipped and reported in the dashboard.

## Connect

1. Connect the ESP32 by USB and allow macOS to enumerate its serial device.
2. Install project dependencies with `python -m pip install -r requirements.txt`.
3. Start the local app with `streamlit run app.py`.
4. Open **ESP32 physical data**, select the ESP32 serial port and its firmware
   baud rate, then click **Connect ESP32**.
5. Send newline-terminated JSON messages. Confirm the received values and
   download the readings CSV if needed. Click **Disconnect ESP32** before
   unplugging or restarting the board.

The application opens the selected serial device exclusively while connected;
close Arduino Serial Monitor or any other serial terminal first. USB device
names and sensor electrical interfaces vary by board and operating system.
Use only correctly rated, low-voltage sensor modules and wiring; do not connect
prototype wiring to industrial machine power, controller outputs, or actuators.

This receiver proves a physical serial data path, not sensor accuracy. Calibrate
each sensor for its units/range and document its placement before interpreting
readings. Physical telemetry remains separate from the synthetic training
distribution. Training and validating a physical-data risk model requires
representative, labeled history and is not implemented by this serial viewer.
