# Architecture

```text
Synthetic simulator ──> labeled training CSV ──> workload-aware features
       │                                             │
       │                                             ├── Isolation Forest
       │                                             └── Random Forest regression
       │                                                        │
       └── live sensor readings ──> feature pipeline ──> risk + anomaly + RUL
                                                              │
                                      explanation <───────────┤
                                      maintenance alerts <────┤
                                      what-if clones <────────┤
                                      virtual service <───────┤
                                      copilot + priority <────┤
                                      CSV event history <─────┤
                                      Streamlit dashboard <───┘
```

## Main components

- `simulator/machine_simulator.py`: deterministic-noise sensor generator with
  NORMAL, DEGRADATION, and NEAR_FAILURE scenarios.
- `data/generate_data.py`: creates labeled sensor traces at varied workloads.
- `ml/features.py`: subtracts the simulator's workload-related healthy baseline
  and derives rolling means, slopes, and RPM instability.
- `ml/anomaly/detector.py`: fits Isolation Forest on healthy training features.
- `ml/prediction/risk_model.py`: trains a Random Forest regressor against the
  synthetic degradation fraction and evaluates held-out machines.
- `ml/explainability/contributions.py`: compares a prediction with feature-wise
  counterfactuals restored to their healthy reference values.
- `ml/rul/estimator.py`: extrapolates the remaining synthetic degradation.
- `backend/service.py`: manages four machine simulators, model analysis,
  workload counterfactuals, local copilot answers, machine priority,
  maintenance transitions and virtual recovery verification, and CSV
  persistence.
- `frontend/dashboard.py`: exposes fleet status, controls, sensor trends,
  what-if projections, a component-linked schematic, explanations, RUL,
  maintenance copilot and virtual repairs, evidence timeline, evaluation, and
  downloads.

## State and files

Training data is saved in `data/generated/synthetic_sensor_data.csv`. Running
the dashboard appends sensor readings to `live_sensor_readings.csv` and
maintenance state transitions to `maintenance_history.csv` and virtual repair/
reset verification to `event_timeline.csv` in the same directory. Dashboard
machine state is held in the current Streamlit session;
the local CSV logs survive restarts. A new dashboard session starts fresh live
simulators; raw historical sensor logs remain downloadable data but are not
replayed into the live simulator automatically.

`models/` is reserved for exported models. This prototype retrains from the
versioned synthetic dataset at startup and does not load opaque prebuilt model
artifacts.
