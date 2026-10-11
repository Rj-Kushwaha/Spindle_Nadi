# Spindle Nadi

**Feel the Pulse. Predict the Failure.**

Spindle Nadi is a runnable, simulation-first machine-health prototype. It
generates multi-axis vibration, spindle load, torque, bearing temperature,
coolant, lubrication, acoustic, feed-rate, tool-wear-estimate, and power-estimate
signals; learns from labeled simulated behavior; detects anomalies and estimates
failure risk; explains risk drivers; estimates prototype remaining useful life;
and records maintenance alerts.

> This is a demonstration built on synthetic data. Its risk, anomaly, and RUL
> outputs are not validated for real equipment and must not be used as safety
> controls or a substitute for qualified maintenance.

## Run it

Requirements: Python 3.11 or newer and a working internet connection for the
initial package installation.

```sh
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local dashboard URL (usually `http://localhost:8501`).
The first launch generates the training dataset if it is missing, trains the
models, and starts with a healthy four-machine fleet.

## Try the end-to-end demo

1. Select `CNC-01` in the sidebar and review the healthy fleet and live readings.
2. Click **Start gradual degradation**. The first sample is produced immediately.
3. In the sidebar, choose a fault mode and click **Start gradual degradation**.
   The mode injects a distinct simulated signature into relevant channels.
4. In **Live & trends**, enable **Auto-stream synthetic readings** for a new
   sample every second. Each displayed sample advances 0.1 simulated hours.
5. Click **Run 10 samples** to make the simulated trend easy to see. The status
   progresses through NORMAL, WARNING, and CRITICAL as model-estimated risk rises.
6. Click **Demo → warning** or **Demo → critical** for a quick judge-ready
   demonstration, or use **Evidence timeline** to inspect each observed sample.
7. Open **Digital twin** for the sensor-linked component schematic and **Why &
   RUL** for the model explanations and prototype estimate.
8. Open **What-if** to project paired baseline and alternative workload/RPM
   settings without changing live machine state.
9. Open **Maintenance copilot** for grounded recommendations, ask an operator
   question, and perform a virtual repair to verify the simulated recovery.
10. Review **Maintenance history** and **Evidence** for recorded actions,
   held-out synthetic metrics, and the illustrative fixed-limit comparison.
11. In **Recorded data replay**, upload a CSV with recognized sensor headers.
   CSV, whitespace-delimited TXT, XLSX, and ZIP archives are supported.
   Timestamped files sort by timestamp; C-MAPSS-style cycle files use their
   cycle order; other timestamp-free files get a clearly labeled synthetic
   one-second-per-row replay axis. PDFs can be attached as separate reference
   documents but are never parsed as sensor data.
12. In **ESP32 physical data**, connect an ESP32 USB serial stream sending
   newline-delimited calibrated JSON sensor readings.
13. Use **Reset machine** to clear the selected machine's current trend and
   restore healthy operation.
14. Open **Alerts & messages** to review dashboard alerts, delivery results,
   and recovery messages. Configure optional SMTP/Twilio channels as described
   in [`docs/notifications.md`](docs/notifications.md).

The other sidebar controls set workload, advance one sample, jump to the
near-failure test state, or return the selected machine to normal. Use the CSV
download buttons to export current session readings and maintenance history.

## Generate data and run tests

Create a new deterministic training dataset (at least five machines are
required so each scripted fault mode is represented):

```sh
python -m data.generate_data --machines 10 --normal-samples 12
```

Run the full test suite:

```sh
python -m unittest discover -s tests -v
```

An optional terminal-only simulator is available with:

```sh
python -m simulator.cli
```

## What the prototype does

- **Data:** [`data/generated/synthetic_sensor_data.csv`](data/generated/synthetic_sensor_data.csv)
  contains reproducible, labeled readings across simulated machines, workloads,
  and five injected fault modes. Regenerate it with the command above.
- **Simulator:** healthy noise, gradual degradation, near-failure, workload
  changes, operating time, reset, and distinct bearing, cooling, lubrication,
  tool-wear, and spindle-imbalance signatures.
- **Signals:** tri-axial vibration, spindle load, torque, bearing temperature,
  coolant flow/temperature, lubrication pressure, acoustic level, feed rate,
  and estimated tool wear and power. Tool wear and power are explicitly marked
  as estimates, not direct measurements.
- **Features:** workload-adjusted sensor deviations, rolling means, five-sample
  trends, operating-RPM-normalized deviations and instability, and
  multi-sensor fault signatures.
- **Health and sensor quality:** the 0–100 prototype health score is
  `100 × (1 − estimated risk)`, not a measured equipment-health value.
  Sensor-quality checks flag missing, nonnumeric, non-finite, broadly
  implausible, or stale simulated channels; their ranges are illustrative,
  not manufacturer alarm limits.
- **Anomaly detection:** Isolation Forest calibrated on healthy synthetic
  operation.
- **Failure risk:** Random Forest regression predicts the simulator's
  degradation fraction. NORMAL is below 50%, WARNING starts at 50%, and
  CRITICAL starts at 85%. These are prototype thresholds.
- **WHY:** each local contribution estimates how much predicted risk changes
  when an individual feature is restored to the healthy training reference.
- **RUL:** estimated by extrapolating the current estimated degradation at the
  simulator's own fixed degradation rate. It is not a real failure-time
  prediction.
- **Maintenance:** alerts are recorded when a machine first transitions into a
  higher model-estimated health state. Virtual maintenance reduces the
  simulator's accumulated wear, preserves operating time, records before/after
  risk and status, and checks up to ten follow-up samples for a return to NORMAL.
- **Alert engine:** NORMAL sends no message; WARNING is a dashboard alert;
  CRITICAL attempts configured email/WhatsApp delivery; verified post-service
  recovery is recorded and also attempts configured channels. Provider results
  are preserved in notification history, including explicit failure or
  not-configured statuses.
- **What-if:** paired, cloned simulations compare alternate workload and RPM
  setpoints over the same horizon without mutating the live machine. Sensor
  operating-point adjustments may keep risk similar even when expected raw
  readings rise.
- **Maintenance copilot:** local deterministic guidance summarizes the model's
  top evidence, trend and RUL in a per-machine chat; it is not an external LLM
  or a source of professional equipment instructions.
- **Spoken status alerts:** optionally announce NORMAL, WARNING, CRITICAL, and
  recovery transitions with distinct browser tones and bilingual English/Hindi
  speech. Audio is opt-in, runs in the browser, and may require the operator to
  click **Play announcement** if autoplay is blocked.
- **Digital twin-style view:** a conceptual sensor-linked machine schematic
  maps model contributors to component markers. It is not CAD or a physical
  digital twin.
- **Fleet priority and evidence timeline:** fleet ordering uses status, risk,
  anomaly score, and RUL; timeline events combine live sensor evidence,
  transitions, reset, and virtual maintenance. Lead time is shown only when
  both warning and fixed-limit events occurred in the observed trace.
- **Dashboard:** fleet overview, one-second synthetic streaming, expanded sensor
  trends, read-only CSV replay, what-if projections,
  component view, explanations, RUL, copilot, virtual maintenance, evidence
  timeline, sensor-quality checks, downloadable maintenance reports, maintenance
  history, evaluation metrics, and data exports.

Details and limitations are in [`docs/architecture.md`](docs/architecture.md),
[`docs/model_notes.md`](docs/model_notes.md), and
[`docs/demo.md`](docs/demo.md). The nine-slide presentation is
[`assets/Spindle_Nadi_Demo.pptx`](assets/Spindle_Nadi_Demo.pptx); regenerate it
using [`docs/presentation.md`](docs/presentation.md).

## Project structure

```text
data/                 Labeled demo data and generated session logs
simulator/            Machine signal simulator and terminal controls
ml/                   Features, anomaly model, risk model, RUL, explanations
models/               Reserved for exported model artifacts
backend/              Fleet state, persistence, alerts, what-if and repair simulation
frontend/             Streamlit dashboard and digital-twin-style view
tests/                Simulator, ML, and backend tests
docs/                 Architecture, model limits, demo script
assets/               Presentation and project media
app.py                Dashboard entry point
requirements.txt      Runtime dependencies
```

## GitHub

For Streamlit Community Cloud deployment, follow
[`docs/deployment.md`](docs/dep./run.shloyment.md). The app entry point is `app.py`,
and the runtime dependencies are in `requirements.txt`.

Recorded CSV replay accepts a timestamp column (`timestamp`, `time`, or
`datetime`) plus one or more recognized sensor headers; it displays only the
uploaded values and does not produce an AI risk score. For genuinely live
machine data, connect a read-only adapter to the controller or data gateway
(for example, the site's approved OPC UA, MTConnect, or MQTT interface), map
its documented tags and units, and validate access with the machine owner.
That integration needs the specific machine, controller protocol, tag list,
credentials/permissions, and network details; none are configured here. Do
not put credentials in source control or allow a demo to issue actuator
commands.
For the physical ESP32 USB message contract, supported sensors, and safe
connection steps, see [`docs/esp32_setup.md`](docs/esp32_setup.md).
Notifications are dashboard-only by default. Email and WhatsApp require
provider credentials configured as Streamlit secrets; critical and verified
recovery transitions trigger messages only when those channels are configured.

Do not commit local `.venv/`, Streamlit secrets, or generated live/history logs.
