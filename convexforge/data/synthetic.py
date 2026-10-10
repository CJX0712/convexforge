"""Synthetic convex-QP generator with **exact known optima**.

Construction (all instances share it):
  1. draw ``Q`` PSD and ``A`` per ``regime``;
  2. draw a primal optimum ``x* >= 0`` with a random support;
  3. draw ``y*`` freely and build ``s* >= 0`` with strict complementarity
     (``s*_i = 0`` wherever ``x*_i > 0``);
  4. set ``b = A x*`` and ``c = A'y* + s* - Q x*``.

Then the KKT system holds *exactly* at ``(x*, y*, s*)``:
``Qx* + c - A'y* - s* = 0``, ``Ax* = b``, ``x* >= 0``, ``s* >= 0``, ``x* ⊙ s* = 0``,
so ``f*`` is the true optimum. This turns every benchmark row into a
ground-truth-verifiable statement instead of a self-reported one.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import NumericalError
from ..core.types import Problem

REGIMES = ("lp", "qp_pd", "qp_singular", "degen")


def _make_Q(regime: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Regime-specific Hessian (always symmetric PSD)."""
    if regime == "lp":
        return np.zeros((n, n))
    if regime == "qp_pd":
        R = rng.standard_normal((n, n))
        Q = (R.T @ R) / n + 0.1 * np.eye(n)
        return Q
    if regime == "qp_singular":
        r = max(1, n // 3)
        B = rng.standard_normal((r, n))
        return (B.T @ B) / r
    if regime == "degen":
        R = rng.standard_normal((n, n))
        return (R.T @ R) / n + 0.02 * np.eye(n)
    raise NumericalError(f"unknown regime {regime!r}", regime=regime)


def _make_A(regime: str, rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    """Regime-specific constraint matrix.

    ``degen`` duplicates rows so ``rank(A) < m``: the IPM normal matrix
    ``A diag(x/s) A'`` is then singular, which is exactly the setting where a
    Newton-based interior-point method loses its advantage and first-order
    splitting methods stay robust.
    """
    A = rng.standard_normal((m, n))
    if regime == "degen":
        k = max(1, m // 3)
        A[m - k :, :] = A[:k, :]
    return A


def _problem_seed(base: int, regime: str, index: int) -> int:
    return int(base) * 100000 + REGIMES.index(regime) * 1000 + int(index)


def make_problem(
    regime: str,
    index: int,
    base_seed: int,
    n_vars: int,
    n_cons: int,
    *,
    support_density: float = 0.6,
) -> Problem:
    """Build one problem whose optimum is known by construction."""
    if regime not in REGIMES:
        raise NumericalError(f"unknown regime {regime!r}", regime=regime)
    seed = _problem_seed(base_seed, regime, index)
    rng = np.random.default_rng(seed % (2**32 - 1))

    n, m = int(n_vars), int(n_cons)
    Q = _make_Q(regime, rng, n)
    A = _make_A(regime, rng, m, n)

    # Primal optimum: mix of active (x*>0) and inactive (x*=0) coordinates.
    support = rng.random(n) < support_density
    if not support.any():  # keep at least one active coordinate
        support[int(rng.integers(n))] = True
    if support.all():  # keep at least one inactive coordinate
        support[int(rng.integers(n))] = False
    x_star = np.zeros(n)
    x_star[support] = rng.uniform(0.1, 2.0, size=int(support.sum()))

    y_star = rng.standard_normal(m)
    s_star = np.zeros(n)
    inactive = ~support
    s_star[inactive] = rng.uniform(0.1, 1.0, size=int(inactive.sum()))

    b = A @ x_star
    c = A.T @ y_star + s_star - (Q @ x_star)
    f_star = float(0.5 * x_star @ (Q @ x_star) + c @ x_star)

    return Problem(
        Q=Q,
        c=c,
        A=A,
        b=b,
        x_star=x_star,
        f_star=f_star,
        regime=regime,
        index=int(index),
        seed=int(seed),
        meta={
            "n": n,
            "m": m,
            "rank_deficient_A": regime == "degen",
            "y_star": y_star,
            "s_star": s_star,
        },
    )


def ground_truth_kkt(prob: Problem) -> dict[str, float]:
    """Verify the constructed optimum really satisfies KKT (residuals ~ 0)."""
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b
    x, y, s = prob.x_star, prob.meta["y_star"], prob.meta["s_star"]
    return {
        "primal": float(np.max(np.abs(A @ x - b))),
        "stationarity": float(np.max(np.abs(Q @ x + c - A.T @ y - s))),
        "complementarity": float(np.max(np.abs(x * s))),
        "nonneg_x": float(np.max(np.maximum(-x, 0.0))),
        "nonneg_s": float(np.max(np.maximum(-s, 0.0))),
    }


def build_problem_set(
    regime: str,
    base_seed: int,
    n_vars: int,
    n_cons: int,
    n_train: int,
    n_test: int,
) -> tuple[list[Problem], list[Problem]]:
    """Disjoint TRAIN / TEST problem blocks (leak-safe routing fit)."""
    train = [make_problem(regime, i, base_seed, n_vars, n_cons) for i in range(n_train)]
    test = [make_problem(regime, n_train + i, base_seed, n_vars, n_cons) for i in range(n_test)]
    return train, test
