# demand_scaling.py
import math
from typing import Callable, Dict, List, Optional


def constant(_, *, c: float) -> float:
    return c


def linear(t: float, *, a: float, b: float) -> float:
    return a * t + b


def exponential(t: float, *, a: float, b: float) -> float:
    return a * math.exp(b * t)


SCALERS: Dict[str, Callable] = {
    "constant": constant,
    "linear": linear,
    "exponential": exponential,
}


def build_demand_scale(
    scaler: Optional[str],
    params: Dict[str, float],
    time_horizon: int,
    default_scale: float,
) -> List[float]:
    """
    Build a time-series of demand scale factors using the chosen scaler and params.
    Falls back to a constant series at default_scale if scaler is None.
    """
    scaler_name = scaler or "constant"
    if scaler_name not in SCALERS:
        raise ValueError(f"Unknown demand scaler '{scaler_name}'. Valid: {list(SCALERS)}")
    scaler_fn = SCALERS[scaler_name]

    # Ensure constant has a default parameter if missing
    if scaler_name == "constant" and "c" not in params:
        params = {**params, "c": default_scale}

    return [float(scaler_fn(t, **params)) for t in range(time_horizon)]
