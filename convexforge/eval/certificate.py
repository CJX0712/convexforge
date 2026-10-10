"""Unified KKT certificate — one fair scoring rule for *every* solver.

Canonical QP::

    min 1/2 x'Qx + c'x    s.t.  A x = b,  x >= 0      (Q symmetric PSD)

Lagrangian ``L = 1/2 x'Qx + c'x + y'(b - Ax) - s'x``, ``s >= 0`` gives the KKT
system

    Qx + c - A'y - s = 0        (stationarity)
    A x = b                     (primal feasibility)
    x >= 0,  s >= 0             (nonnegativity)
    x_i s_i = 0                 (complementarity)

Given only ``(x, y)`` from a solver, the *best possible* choice of ``s`` (the one
minimising the stationarity residual subject to ``s >= 0``) is ``s = max(0, r_d)``
with ``r_d = Qx + c - A'y``. Using that same rule everywhere makes the score
identically fair for all engines, and leaves the residuals honest:

    stationarity residual  ->  ||min(0, r_d)||_inf     (dual infeasibility)
    primal residual        ->  ||Ax - b||_inf
    complementarity        ->  ||x ⊙ s||_inf
    nonnegativity          ->  ||max(0, -x)||_inf
"""

from __future__ import annotations

import numpy as np

from ..core.types import Problem


def dual_residual(prob: Problem, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``r_d = Qx + c - A'y``."""
    return prob.Q @ x + prob.c - prob.A.T @ y


def certificate(prob: Problem, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the (scale-normalised) certificate components and their max."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b

    r_p_raw = A @ x - b
    r_d = dual_residual(prob, x, y)
    s = np.maximum(r_d, 0.0)
    comp_raw = x * s
    neg_raw = np.maximum(-x, 0.0)

    # Scale-free normalisations (documented, identical for every engine).
    sb = max(1.0, float(np.max(np.abs(b))))
    sd = max(
        1.0,
        float(np.max(np.abs(c))) + float(np.max(np.abs(Q))) * max(1.0, float(np.max(np.abs(x)))),
    )
    sxinf = max(1.0, float(np.max(np.abs(x))))
    ssinf = max(1.0, float(np.max(np.abs(s))))
    sc = max(1.0, sxinf * ssinf)

    r_primal = float(np.max(np.abs(r_p_raw))) / sb
    r_dual = float(np.max(np.maximum(-r_d, 0.0))) / sd
    r_comp = float(np.max(np.abs(comp_raw))) / sc
    r_nonneg = float(np.max(neg_raw)) / sxinf

    return {
        "r_primal": r_primal,
        "r_dual": r_dual,
        "r_comp": r_comp,
        "r_nonneg": r_nonneg,
        "cert": max(r_primal, r_dual, r_comp, r_nonneg),
    }


def suboptimality(prob: Problem, x: np.ndarray) -> float:
    """``f(x) - f*`` (exact: every instance carries its true optimum)."""
    return prob.objective(np.asarray(x, dtype=float)) - prob.f_star


def duality_gap_lp(prob: Problem, x: np.ndarray, y: np.ndarray) -> float:
    """``c'x - b'y`` — weak duality demands this be >= 0 whenever x is feasible.

    Only meaningful for LP instances (``Q = 0``), where the dual objective has the
    closed form ``b'y``.
    """
    return float(prob.c @ x) - float(prob.b @ y)
