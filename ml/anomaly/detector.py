"""Isolation Forest detector calibrated on simulated healthy operation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.features import FEATURE_COLUMNS


class HealthyBaselineDetector:
    """Detect readings that differ from the healthy training distribution."""

    def __init__(self) -> None:
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42,
        )
        self._healthy_scores = np.array([], dtype=float)

    def fit(self, healthy_features: pd.DataFrame) -> None:
        if len(healthy_features) < 10:
            raise ValueError("At least 10 healthy samples are required to train anomaly detection")
        values = healthy_features[FEATURE_COLUMNS]
        self.model.fit(values)
        self._healthy_scores = np.sort(self.model.decision_function(values))

    def score(self, features: pd.DataFrame) -> tuple[float, bool]:
        if self._healthy_scores.size == 0:
            raise RuntimeError("Anomaly detector has not been fitted")
        values = features[FEATURE_COLUMNS]
        decision_score = float(self.model.decision_function(values)[0])
        rank = int(np.searchsorted(self._healthy_scores, decision_score, side="left"))
        healthy_percentile = rank / len(self._healthy_scores)
        score = 100.0 * (0.05 - healthy_percentile) / 0.05
        is_anomaly = int(self.model.predict(values)[0]) == -1
        return round(float(np.clip(score, 0.0, 100.0)), 1), is_anomaly
