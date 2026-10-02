"""Train and connect the prototype anomaly and failure-risk models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.anomaly.detector import HealthyBaselineDetector
from ml.explainability.contributions import explain_risk
from ml.features import FEATURE_COLUMNS, engineer_features
from ml.prediction.risk_model import (
    CRITICAL_THRESHOLD,
    WARNING_THRESHOLD,
    FailureRiskModel,
)
from ml.rul.estimator import estimate_rul_hours


@dataclass(frozen=True)
class MachineAnalysis:
    risk: float
    status: str
    anomaly_score: float
    is_anomaly: bool
    rul_hours: float
    explanations: list[dict[str, str | float]]
    features: pd.DataFrame


class SpindleNadiPipeline:
    """Fit models on labeled simulated data and analyze machine readings."""

    def __init__(self, training_readings: pd.DataFrame) -> None:
        if training_readings.empty:
            raise ValueError("Training data must not be empty")
        self.training_readings = training_readings.copy()
        self.training_features = engineer_features(self.training_readings)
        self.anomaly_detector = HealthyBaselineDetector()
        healthy = self.training_readings["degradation"].astype(float).le(0.0)
        self.anomaly_detector.fit(self.training_features.loc[healthy, FEATURE_COLUMNS])
        self.risk_model = FailureRiskModel()
        self.risk_model.fit(self.training_features, self.training_readings)

    @property
    def metrics(self) -> dict[str, float | list[list[int]]]:
        return self.risk_model.metrics.copy()

    def analyze(self, readings: pd.DataFrame) -> MachineAnalysis:
        if readings.empty:
            raise ValueError("At least one reading is required for analysis")
        features = engineer_features(readings)
        latest = features.iloc[[-1]]
        risk = self.risk_model.predict(latest)
        anomaly_score, is_anomaly = self.anomaly_detector.score(latest)
        if risk >= CRITICAL_THRESHOLD:
            status = "CRITICAL"
        elif risk >= WARNING_THRESHOLD:
            status = "WARNING"
        else:
            status = "NORMAL"
        return MachineAnalysis(
            risk=risk,
            status=status,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            rul_hours=estimate_rul_hours(risk),
            explanations=explain_risk(self.risk_model, latest),
            features=latest[FEATURE_COLUMNS].copy(),
        )
