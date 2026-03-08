"""Inventory observation discretization schemes.

Provides a flexible way to define how continuous inventory values
are mapped to discrete states. Instead of only uniform equal-width
buckets, users can specify arbitrary boundary arrays.

The core idea:
  Given sorted boundaries [b0, b1, ..., b_{k-1}], an inventory value
  is mapped to state j = searchsorted(boundaries, inv).
  This produces k+1 discrete states: [0, b0), [b0, b1), ..., [b_{k-1}, ∞).

Built-in schemes:
  - "continuous"  : No discretization (pass-through)
  - "uniform-N"   : N equal-width buckets (e.g. "uniform-16")
  - "stockout"    : 2 states: {0} vs {1..max} (boundary at 0.5)
  - "low_stock-K" : K+1 states: 0,1,...,K-1 individually, K+ lumped (K boundaries)
  - custom dict   : {"boundaries": [b0, b1, ...], "description": "..."}

All boundary arrays are expressed in terms of the raw integer inventory
(0..inv_max). They are resolved at experiment start time and saved in
args.pkl for full reproducibility.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_scheme(
    spec: Union[str, int, Dict[str, Any]],
    inv_max: float,
) -> Tuple[np.ndarray, int, str]:
    """Convert a scheme specification into (boundaries, num_states, description).

    Parameters
    ----------
    spec : str | int | dict
        - int or "continuous" / -1 : no discretization  → ([], 0, ...)
        - "uniform-N"             : N equal-width buckets
        - int >= 1                : shorthand for uniform-N
        - "stockout"              : 2 states – out-of-stock vs in-stock
        - "low_stock-K"           : K+1 states – 0..K-1 fine, K+ lumped
        - dict with "boundaries"  : user-defined boundaries list
    inv_max : float
        Maximum inventory value (e.g. 8800).

    Returns
    -------
    boundaries : np.ndarray, float64, sorted, shape (num_boundaries,)
    num_states : int  (len(boundaries) + 1, or 0 for continuous)
    description : str (human-readable mapping summary)
    """
    if isinstance(spec, (int, float, np.integer, np.floating)):
        n = int(spec)
        if n <= 0:
            return np.array([], dtype=np.float64), 0, "continuous (no discretization)"
        if n == 1:
            return np.array([], dtype=np.float64), 1, "1 state: all inventory → 0 (full hiding)"
        return _uniform(n, inv_max)

    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("continuous", "-1"):
            return np.array([], dtype=np.float64), 0, "continuous (no discretization)"
        if s.startswith("uniform-"):
            n = int(s.split("-", 1)[1])
            return _uniform(n, inv_max)
        if s == "stockout":
            return _stockout(inv_max)
        if s.startswith("low_stock-"):
            k = int(s.split("-", 1)[1])
            return _low_stock(k, inv_max)
        raise ValueError(f"Unknown scheme string: {spec!r}")

    if isinstance(spec, dict):
        boundaries = np.array(spec["boundaries"], dtype=np.float64)
        boundaries = np.sort(boundaries)
        num_states = len(boundaries) + 1
        desc = spec.get("description", f"custom {num_states}-state scheme")
        return boundaries, num_states, desc

    raise TypeError(f"Cannot resolve scheme spec of type {type(spec)}: {spec!r}")


def pad_boundaries_list(
    all_boundaries: List[np.ndarray],
    pad_value: float,
) -> np.ndarray:
    """Pad a list of boundary arrays to the same length for vmap/gridsearch.

    Parameters
    ----------
    all_boundaries : list of 1-D arrays (one per gridsearch config)
    pad_value : float
        Value used for padding (should be > inv_max so searchsorted
        never matches the pad entries). Use inv_max + 1.

    Returns
    -------
    padded : np.ndarray, shape (num_configs, max_len), float64
    """
    if not all_boundaries:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(b) for b in all_boundaries)
    if max_len == 0:
        max_len = 1  # at least 1 column so array isn't degenerate
    padded = np.full((len(all_boundaries), max_len), pad_value, dtype=np.float64)
    for i, b in enumerate(all_boundaries):
        if len(b) > 0:
            padded[i, : len(b)] = b
    return padded


def describe_mapping(
    boundaries: np.ndarray,
    num_states: int,
    inv_max: float,
) -> str:
    """Generate a human-readable description of the inventory mapping.

    Returns a multi-line string showing which inventory ranges map to
    which discrete state index.
    """
    if num_states == 0:
        return "  continuous: inv → inv/inv_max (no discretization)"
    if num_states == 1:
        return "  state 0: all inv values → 0"

    lines = []
    boundaries = np.asarray(boundaries, dtype=np.float64)
    # Only use actual (non-padding) boundaries
    real = boundaries[boundaries < inv_max + 0.5]  # filter out padding
    for j in range(len(real) + 1):
        lo = 0 if j == 0 else int(np.ceil(real[j - 1]))
        hi = int(inv_max) if j == len(real) else int(np.floor(real[j] - 1e-9))
        count = hi - lo + 1
        lines.append(f"  state {j}: inv [{lo}..{hi}] ({count} values)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in scheme implementations
# ---------------------------------------------------------------------------

def _uniform(n: int, inv_max: float) -> Tuple[np.ndarray, int, str]:
    """N equal-width buckets using boundaries at (inv_max+1)*k/n for k=1..n-1."""
    if n < 2:
        return np.array([], dtype=np.float64), 1, "1 state: all → 0"
    width = (inv_max + 1.0) / n
    boundaries = np.array([width * k for k in range(1, n)], dtype=np.float64)
    bucket_size = int(round(width))
    desc = f"uniform-{n}: {n} buckets of ~{bucket_size} values each"
    return boundaries, n, desc


def _stockout(inv_max: float) -> Tuple[np.ndarray, int, str]:
    """2 states: out-of-stock (inv==0) vs in-stock (inv>=1)."""
    boundaries = np.array([0.5], dtype=np.float64)
    desc = "stockout: state 0 = {0} (out of stock), state 1 = {1.." + f"{int(inv_max)}" + "}"
    return boundaries, 2, desc


def _low_stock(k: int, inv_max: float) -> Tuple[np.ndarray, int, str]:
    """K+1 states: 0,1,...,K-1 are individual levels; K+ is lumped.

    Boundaries at 0.5, 1.5, ..., (K-0.5).
    """
    if k < 1:
        raise ValueError(f"low_stock-K requires K >= 1, got {k}")
    boundaries = np.array([i + 0.5 for i in range(k)], dtype=np.float64)
    num_states = k + 1
    desc = (f"low_stock-{k}: states 0..{k-1} map to inv 0..{k-1} individually, "
            f"state {k} = inv {k}..{int(inv_max)}")
    return boundaries, num_states, desc
