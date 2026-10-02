"""Failure-progression regression and group-held-out evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from ml.features import FEATURE_COLUMNS

WARNING_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.80


class FailureRiskModel:
    """Estimate simulated degradation from engineered sensor features."""

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=1,
        )
        self.healthy_reference: pd.Series | None = None
        self.metrics: dict[str, float | list[list[int]]] = {}

    def fit(self, features: pd.DataFrame, readings: pd.DataFrame) -> None:
        if "degradation" not in readings or "machine_id" not in readings:
            raise ValueError("Training readings need degradation labels and machine IDs")
        if len(features) != len(readings):
            raise ValueError("Features and training readings must have matching row counts")

        x = features[FEATURE_COLUMNS]
        y = readings["degradation"].astype(float).clip(0.0, 1.0)
        groups = readings["machine_id"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_indices, test_indices = next(splitter.split(x, y, groups))

        evaluation_model = self._new_model()
        evaluation_model.fit(x.iloc[train_indices], y.iloc[train_indices])
        test_predictions = np.clip(evaluation_model.predict(x.iloc[test_indices]), 0.0, 1.0)
        actual = y.iloc[test_indices].to_numpy()
        predicted_critical = test_predictions >= WARNING_THRESHOLD
        actual_critical = actual >= WARNING_THRESHOLD
        matrix = confusion_matrix(actual_critical, predicted_critical, labels=[False, True])
        self.metrics = {
            "mae": float(mean_absolute_error(actual, test_predictions)),
            "precision": float(precision_score(actual_critical, predicted_critical, zero_division=0)),
            "recall": float(recall_score(actual_critical, predicted_critical, zero_division=0)),
            "f1": float(f1_score(actual_critical, predicted_critical, zero_division=0)),
            "confusion_matrix": matrix.tolist(),
            "test_samples": float(len(test_indices)),
            "test_machines": float(groups.iloc[test_indices].nunique()),
        }

        self.model.fit(x, y)
        healthy = readings["degradation"].astype(float).le(0.0)
        if not healthy.any():
            raise ValueError("Training readings must include healthy samples")
        self.healthy_reference = x.loc[healthy].median()

    def predict(self, features: pd.DataFrame) -> float:
        if self.healthy_reference is None:
            raise RuntimeError("Failure risk model has not been fitted")
        prediction = float(self.model.predict(features[FEATURE_COLUMNS])[0])
        return round(float(np.clip(prediction, 0.0, 1.0)), 4)

    @staticmethod
    def _new_model() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=1,
        )
