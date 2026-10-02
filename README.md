# Spindle Nadi

**Feel the Pulse. Predict the Failure.**

Spindle Nadi is a complete, runnable machine-health prototype. It simulates
temperature, vibration, motor current, RPM, workload, and operating time; learns
from labeled simulated behavior; detects anomalies and estimates failure risk;
explains risk drivers; estimates prototype remaining useful life; and records
maintenance alerts.

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
3. Click **Run 10 samples** to make the simulated trend easy to see. The status
   progresses through NORMAL, WARNING, and CRITICAL as model-estimated risk rises.
4. Click **Demo → warning** or **Demo → critical** for a quick judge-ready
   demonstration, or use **Evidence timeline** to inspect each observed sample.
5. Open **Digital twin** for the sensor-linked component schematic and **Why &
   RUL** for the model explanations and prototype estimate.
6. Open **What-if** to project paired baseline and alternative workloads
   without changing live machine state.
7. Open **Maintenance copilot** for grounded recommendations, ask an operator
   question, and perform a virtual repair to verify the simulated recovery.
8. Review **Maintenance history** and **Evidence** for recorded actions,
   held-out synthetic metrics, and the illustrative fixed-limit comparison.
9. Use **Reset machine** to clear the selected machine's current trend and
   restore healthy operation.

The other sidebar controls set workload, advance one sample, jump to the
near-failure test state, or return the selected machine to normal. Use the CSV
download buttons to export current session readings and maintenance history.

## Generate data and run tests

Create a new deterministic training dataset:

```sh
python -m data.generate_data --machines 8 --normal-samples 12
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
  contains reproducible, labeled readings across multiple simulated machines
  and workloads. Regenerate it with the command above.
- **Simulator:** healthy noise, gradual degradation, near-failure, workload
  changes, operating time, and reset.
- **Features:** workload-adjusted sensor deviations, rolling means, five-sample
  trends, and RPM instability.
- **Anomaly detection:** Isolation Forest calibrated on healthy synthetic
  operation.
- **Failure risk:** Random Forest regression predicts the simulator's
  degradation fraction. NORMAL is below 40%, WARNING starts at 40%, and
  CRITICAL starts at 80%. These are prototype thresholds.
- **WHY:** each local contribution estimates how much predicted risk changes
  when an individual feature is restored to the healthy training reference.
- **RUL:** estimated by extrapolating the current estimated degradation at the
  simulator's own fixed degradation rate. It is not a real failure-time
  prediction.
- **Maintenance:** alerts are recorded when a machine first transitions into a
  higher model-estimated health state. Virtual maintenance reduces the
  simulator's accumulated wear, preserves operating time, records before/after
  risk and status, and verifies whether estimated risk improved.
- **What-if:** paired, cloned simulations compare the current workload against
  a different workload over the same horizon without mutating the live machine.
  Workload-adjusted features may keep risk similar even when raw readings rise.
- **Maintenance copilot:** local deterministic guidance summarizes the model's
  top evidence, trend and RUL; it is not an external LLM or a source of
  professional equipment instructions.
- **Digital twin-style view:** a conceptual sensor-linked machine schematic
  maps model contributors to component markers. It is not CAD or a physical
  digital twin.
- **Fleet priority and evidence timeline:** fleet ordering uses status, risk,
  anomaly score, and RUL; timeline events combine live sensor evidence,
  transitions, reset, and virtual maintenance. Lead time is shown only when
  both warning and fixed-limit events occurred in the observed trace.
- **Dashboard:** fleet overview, live readings, trends, what-if projections,
  component view, explanations, RUL, copilot, virtual maintenance, evidence
  timeline, maintenance history, evaluation metrics, and data exports.

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
[`docs/deployment.md`](docs/deployment.md). The app entry point is `app.py`,
and the runtime dependencies are in `requirements.txt`.

This workspace is not currently connected to a GitHub repository. Do not
commit local `.venv/`, Streamlit secrets, or generated live/history logs.
