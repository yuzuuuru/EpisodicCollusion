# demand_function.py
"""
Time-varying demand scaling factor utilities.
Provides functions to build demand scale arrays for use in market environments.
"""
import math
from typing import Callable, Dict, List, Optional, Union
import numpy as np


def constant(t: float, *, c: float) -> float:
    """Constant demand scale: λ(t) = c"""
    return c


def linear(t: float, *, a: float, b: float) -> float:
    """Linear demand scale: λ(t) = a*t + b"""
    return a * t + b


def exponential(t: float, *, a: float, b: float) -> float:
    """Exponential demand scale: λ(t) = a * exp(b*t)"""
    return a * math.exp(b * t)


SCALERS: Dict[str, Callable] = {
    "constant": constant,
    "linear": linear,
    "exponential": exponential,
}


def compute_linear_params(start: float, end: float, time_horizon: int) -> Dict[str, float]:
    """
    Compute linear scaler parameters (a, b) from start and end values.
    λ(0) = b = start
    λ(T-1) = a*(T-1) + b = end
    => a = (end - start) / (T - 1)
    """
    if time_horizon <= 1:
        return {"a": 0.0, "b": start}
    a = (end - start) / (time_horizon - 1)
    b = start
    return {"a": a, "b": b}


def compute_exponential_params(start: float, end: float, time_horizon: int) -> Dict[str, float]:
    """
    Compute exponential scaler parameters (a, b) from start and end values.
    λ(0) = a * exp(0) = a = start
    λ(T-1) = a * exp(b*(T-1)) = end
    => b = ln(end/start) / (T - 1)
    """
    if time_horizon <= 1:
        return {"a": start, "b": 0.0}
    if start <= 0 or end <= 0:
        raise ValueError("Start and end values must be positive for exponential scaler")
    a = start
    b = math.log(end / start) / (time_horizon - 1)
    return {"a": a, "b": b}


def build_demand_scale(
    scaler: Optional[str],
    params: Optional[Dict[str, float]],
    time_horizon: int,
    default_scale: float,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[float]:
    """
    Build a time-series of demand scale factors using the chosen scaler and params.
    
    Args:
        scaler: Scaler type ("constant", "linear", "exponential") or None for constant.
        params: Dictionary of scaler parameters (e.g., {"a": 50, "b": 500}).
                If None and start/end are provided, params will be computed automatically.
        time_horizon: Number of time steps.
        default_scale: Default scale factor (used for constant scaler or as fallback).
        start: Starting value for demand scale (optional, for auto-computing params).
        end: Ending value for demand scale (optional, for auto-computing params).
    
    Returns:
        List of demand scale factors for each time step [λ(0), λ(1), ..., λ(T-1)].
    
    Examples:
        # Constant (backward compatible)
        build_demand_scale(None, None, 20, 1000)  # -> [1000, 1000, ..., 1000]
        
        # Linear with explicit params
        build_demand_scale("linear", {"a": 50, "b": 500}, 20, 1000)
        
        # Linear with start/end (auto-compute params)
        build_demand_scale("linear", None, 20, 1000, start=500, end=1500)
        
        # Exponential with start/end
        build_demand_scale("exponential", None, 20, 1000, start=500, end=1500)
    """
    scaler_name = scaler or "constant"
    
    if scaler_name not in SCALERS:
        raise ValueError(f"Unknown demand scaler '{scaler_name}'. Valid: {list(SCALERS)}")
    
    # Auto-compute params from start/end if provided
    if params is None and start is not None and end is not None:
        if scaler_name == "linear":
            params = compute_linear_params(start, end, time_horizon)
        elif scaler_name == "exponential":
            params = compute_exponential_params(start, end, time_horizon)
        elif scaler_name == "constant":
            # For constant, just use start value
            params = {"c": start}
    
    # Handle missing params
    if params is None:
        params = {}
    
    # Ensure constant has a default parameter if missing
    if scaler_name == "constant" and "c" not in params:
        params = {**params, "c": default_scale}
    
    scaler_fn = SCALERS[scaler_name]
    
    return [float(scaler_fn(t, **params)) for t in range(time_horizon)]


def build_demand_scale_array(
    scaler: Optional[str],
    params: Optional[Dict[str, float]],
    time_horizon: int,
    default_scale: float,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> np.ndarray:
    """
    Same as build_demand_scale but returns a numpy array.
    Useful for JAX compatibility.
    """
    return np.array(build_demand_scale(scaler, params, time_horizon, default_scale, start, end))
