"""A transparent RUL estimate tied to the synthetic degradation schedule."""

DEGRADATION_PER_SAMPLE = 0.06
SIMULATED_HOURS_PER_SAMPLE = 0.1


def estimate_rul_hours(risk: float) -> float:
    """Estimate hours to simulated critical condition; this is not a guarantee."""
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be between 0 and 1")
    degradation_per_hour = DEGRADATION_PER_SAMPLE / SIMULATED_HOURS_PER_SAMPLE
    return round(max(0.0, (1.0 - risk) / degradation_per_hour), 2)
