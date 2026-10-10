"""Aggregate metrics over problem sets."""

from __future__ import annotations

import numpy as np


def _safe_log10(values: list[float], floor: float = 1e-16) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def geometric_mean(values: list[float], floor: float = 1e-16) -> float:
    """Geometric mean, floored to avoid log(0).  Used for residual magnitudes."""
    return float(10.0 ** float(np.mean(_safe_log10(values, floor))))


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(len(arr))}


def iters_to_tol(values: list[int], cap: int) -> dict[str, float]:
    """Statistics for 'iterations until certificate <= tol' (cap = not reached)."""
    arr = np.asarray(values, dtype=float)
    reached = arr < cap
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "reached": int(reached.sum()),
        "frac_reached": float(reached.mean()) if len(arr) else 0.0,
    }
