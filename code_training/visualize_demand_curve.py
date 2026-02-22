"""
Exponential demand curves with discrete-time sum constraint.

We consider:
  λ(t) = a * exp(b*t),  t = 0,1,...,T-1

Given:
  - a = λ(0) (fixed, chosen by you)
  - total = sum_{t=0..T-1} λ(t) (fixed)

We solve for b such that:
  sum_{t=0..T-1} a * exp(b*t) = total

Then we can plot any subset of curves by selecting 'a' values.

No external dependencies beyond numpy/matplotlib.
"""

import math
from typing import Iterable, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def geom_sum(q: float, n: int) -> float:
    """Sum_{k=0..n-1} q^k."""
    if abs(q - 1.0) < 1e-14:
        return float(n)
    return (q**n - 1.0) / (q - 1.0)


def solve_q_for_geom_sum(target: float, n: int, *, iters: int = 300) -> float:
    """
    Solve geom_sum(q, n) = target for q > 0 via bisection.

    Properties:
      - For q in (0, 1): geom_sum(q,n) increases from 1 to n
      - For q in (1, inf): geom_sum(q,n) increases from n to inf

    So:
      - If target < n -> q in (0,1)
      - If target == n -> q = 1
      - If target > n -> q > 1
    """
    if abs(target - n) < 1e-12:
        return 1.0

    if target < n:
        # q in (0,1). Feasible target range is (1, n).
        if target <= 1.0:
            return 0.0  # extremely steep decay (edge)
        lo, hi = 0.0, 1.0
    else:
        # q > 1
        lo, hi = 1.0, 2.0
        while geom_sum(hi, n) < target:
            hi *= 2.0
            if hi > 1e10:
                raise RuntimeError("Failed to bracket solution for q>1; check parameters.")

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        val = geom_sum(mid, n)
        if val < target:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def solve_b_given_a_total(a: float, total: float, T: int) -> float:
    """
    Given a = λ(0), solve b such that sum_{t=0..T-1} a*exp(b*t) = total.
    """
    if a <= 0:
        raise ValueError("a must be positive.")
    target = total / a  # need geom_sum(q,T)=total/a with q=exp(b)
    q = solve_q_for_geom_sum(target, T)
    if q == 0.0:
        return -float("inf")
    return math.log(q)


def make_curve(a: float, b: float, T: int) -> np.ndarray:
    t = np.arange(T)
    return a * np.exp(b * t)


def compute_params_and_curve(a: float, total: float, T: int) -> Tuple[float, float, float, np.ndarray]:
    """
    Returns (b, lambda_end, sum_check, curve).
    """
    b = solve_b_given_a_total(a, total, T)
    curve = make_curve(a, b, T)
    return b, float(curve[-1]), float(curve.sum()), curve


def plot_exponential_curves(
    a_values: Iterable[float],
    *,
    total: float = 20000.0,
    T: int = 20,
    show_points: bool = True,
    show_legend: bool = True,
    title: str | None = None,
) -> Dict[float, Dict[str, float]]:
    """
    Plot curves for selected a_values.

    Returns a dict with per-a parameters:
      {a: {"b":..., "lambda_end":..., "sum":...}}
    """
    a_list: List[float] = list(a_values)
    if not a_list:
        raise ValueError("a_values is empty.")

    t = np.arange(T)

    results: Dict[float, Dict[str, float]] = {}

    plt.figure()
    for a in a_list:
        b, lam_end, s, curve = compute_params_and_curve(a, total, T)
        results[float(a)] = {"b": b, "lambda_end": lam_end, "sum": s}

        if show_points:
            plt.plot(t, curve, marker="o", markersize=3, label=f"a=λ(0)={a:g}")
        else:
            plt.plot(t, curve, label=f"a=λ(0)={a:g}")

    plt.xlabel("t")
    plt.ylabel("λ(t)")
    if title is None:
        title = f"Exponential demand scales (sum_{{t=0..{T-1}}} λ(t) = {total:g})"
    plt.title(title)

    if show_legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visualize_demand_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


if __name__ == "__main__":
    # ---- Choose which curves to show by editing this list ----
    # Examples:
    #   a_values = [100, 200, 400, 800, 1200, 1400, 1600]
    #   a_values = [800, 1000, 1200]  # compare mild growth/flat/mild decay
    #   a_values = [100, 200]         # strong growth cases only
    a_values = [100, 200, 400, 800,1000, 1200, 1600]

    results = plot_exponential_curves(
        a_values,
        total=20000.0,
        T=20,
        show_points=False,   # True to show markers on each integer t
        show_legend=True,
    )

    # Print parameters for each shown curve
    print("Computed parameters (given sum constraint):")
    for a in a_values:
        r = results[float(a)]
        print(f"a={a:>6g}  b={r['b']:+.6f}  λ(19)={r['lambda_end']:.6f}  sum={r['sum']:.6f}")
