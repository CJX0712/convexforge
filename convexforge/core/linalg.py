"""Deterministic numeric guardrails shared by the solvers."""

from __future__ import annotations

import numpy as np

from .errors import NumericalError

_JITTER = 1e-12


def _jitter_scale(mat: np.ndarray) -> float:
    """Scale-aware jitter for regularising a (near-)singular matrix."""
    scale = float(np.max(np.abs(mat))) if mat.size else 1.0
    return _JITTER * max(scale, 1.0)


def solve_spd(mat: np.ndarray, rhs: np.ndarray, *, min_eig: float = 0.0) -> np.ndarray:
    """Solve ``mat z = rhs`` for symmetric PSD ``mat``.

    Tries Cholesky, then a scale-aware jitter, then least squares. Raises
    :class:`NumericalError` if every path produces a non-finite result.
    """
    mat = 0.5 * (mat + mat.T)
    n = mat.shape[0]
    attempt = mat.copy()
    for _ in range(6):
        try:
            sol = np.linalg.solve(attempt, rhs)
            if np.all(np.isfinite(sol)):
                return sol
        except np.linalg.LinAlgError:
            pass
        attempt = attempt + (_jitter_scale(attempt) + min_eig) * np.eye(n)
        break
    sol = np.linalg.lstsq(attempt, rhs, rcond=None)[0]
    if not np.all(np.isfinite(sol)):
        raise NumericalError("SPD solve produced non-finite output")
    return sol


#: Iterates beyond this magnitude are treated as divergent (keeps x'x from
#: overflowing float64 and stops a singular KKT solve poisoning the run).
_SANITY_LIMIT = 1e12


def iterate_sane(x: np.ndarray, limit: float = _SANITY_LIMIT) -> bool:
    """True iff ``x`` is finite and not absurdly large.

    Consensus ADMM on a rank-deficient constraint matrix can diverge to ~1e150;
    without this guard ``np.linalg.norm`` would overflow to ``inf``.
    """
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)) or arr.size == 0:
        return False
    return float(np.max(np.abs(arr))) <= limit


def assert_psd(Q: np.ndarray, tol: float = 1e-8) -> None:
    """Assert ``Q`` is symmetric PSD."""
    sym = float(np.max(np.abs(Q - Q.T)))
    if sym > tol:
        raise NumericalError(f"Q not symmetric (max asym {sym:.3e})", sym=sym)
    eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (Q + Q.T)))) if Q.size else 0.0
    if eig_min < -tol:
        raise NumericalError(f"Q not PSD (min eig {eig_min:.3e})", eig_min=eig_min)
