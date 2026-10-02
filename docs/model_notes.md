# Model behavior and limitations

## Synthetic dataset

The checked-in dataset is generated with a fixed seed and contains eight
simulated machines at different workloads. Each machine has twelve healthy
samples followed by gradual degradation until its simulated degradation
reaches 1.0. Sensor noise and operating time are included. Labels come from the
simulator, not independently observed machine failures.

## Feature engineering

For each sensor reading, workload-adjusted excess is calculated relative to
the simulator's healthy equations:

- Temperature: `temperature - (38 + 0.20 × workload)`
- Vibration: `vibration - (0.75 + 0.006 × workload)`
- Motor current: `current - (4 + 0.075 × workload)`
- RPM drop: `1800 - rpm`

Recent five-reading means, five-reading slopes, and rolling RPM standard
deviation add context. A machine's history is kept separate when calculating
rolling features.

## Models and evaluation

Isolation Forest is fitted only to healthy training features. The UI shows a
lower-tail score scaled against the healthy training-score distribution and
whether Isolation Forest classifies the reading as an outlier.

Random Forest regression estimates the simulator degradation fraction from
sensor features. Evaluation holds out entire machine IDs (rather than randomly
splitting neighboring time-series samples), then reports MAE and warning-class
precision, recall, F1, and confusion matrix. The dashboard calculates these
from the generated data at model startup. They measure performance on this
synthetic simulator distribution only; they do not represent real-world
accuracy.

The health-state thresholds are risk below 0.40 = NORMAL, risk at least 0.40 =
WARNING, and risk at least 0.80 = CRITICAL. They are prototype operating
settings, not engineering alarm limits.

## Explanations and RUL

The explanation is an estimate: for each input feature, the model prediction
is repeated with that one feature replaced by its healthy training median.
Positive reductions are shown as contributions. This is a small, model-based
counterfactual, not a causal explanation.

RUL extrapolates the model's degradation estimate using the simulator schedule
(0.06 degradation per 0.1 simulated operating hour). It is a demonstration of
the interface and calculation only. It is not a probability distribution,
failure guarantee, or validated estimate of a physical machine's lifetime.

The fixed-limit comparator uses illustrative temperature, vibration, current,
and RPM thresholds displayed in the Evidence tab. It must not be used as a
safety limit. The application computes warning lead time only from a machine
trace where both events (model warning and fixed-limit crossing) have already
been observed. It does not extrapolate lead time or claim that the model leads
the limit in advance.

## What-if, service simulation, and digital twin-style view

Workload what-if results compare two cloned simulator instances advanced with
paired random state, one at the current load and one at an alternative load.
The live simulator is unchanged. Load-adjusted features intentionally allow
raw readings to increase without treating expected load response alone as
degradation.

Virtual maintenance reduces the simulator's scalar degradation by the
operator-selected recovery fraction, preserves operating hours, records before
and after model estimates, and reports recovery only if estimated risk
decreases. The virtual scalar repair is illustrative, not a component-level
physical repair model.

The digital-twin-style view is a conceptual schematic that colors component
markers using local feature contributions. It has no CAD geometry, calibrated
physics, or actuator integration. The maintenance copilot uses deterministic
templates grounded in the local model output; it does not call an LLM or issue
validated industrial instructions.

## Before real deployment

Use validated sensor instrumentation and historical runs with verified
maintenance/failure labels; assess sensor drift, missing readings, workload
and machine differences; define safety thresholds with equipment experts; test
out-of-distribution and false-alarm behavior; and independently validate any
maintenance decision process. This prototype does none of those validations.
