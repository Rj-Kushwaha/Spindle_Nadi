"""Simple model-based, per-reading feature contribution estimates."""

from __future__ import annotations

import pandas as pd

from ml.features import FEATURE_COLUMNS, FEATURE_LABELS
from ml.prediction.risk_model import FailureRiskModel


def explain_risk(
    model: FailureRiskModel,
    features: pd.DataFrame,
    limit: int = 5,
) -> list[dict[str, str | float]]:
    """Estimate local risk contributions by replacing features with healthy values."""
    if model.healthy_reference is None:
        raise RuntimeError("Failure risk model has not been fitted")
    current = features[FEATURE_COLUMNS].iloc[[0]].copy()
    baseline_risk = model.predict(current)
    healthy = model.healthy_reference
    explanations: list[dict[str, str | float]] = []

    for column in FEATURE_COLUMNS:
        counterfactual = current.copy()
        counterfactual.loc[:, column] = healthy[column]
        contribution = max(0.0, baseline_risk - model.predict(counterfactual))
        explanations.append(
            {
                "feature": FEATURE_LABELS[column],
                "contribution": round(contribution, 4),
                "value": round(float(current.iloc[0][column]), 3),
            }
        )

    explanations.sort(key=lambda item: float(item["contribution"]), reverse=True)
    return explanations[:limit]
