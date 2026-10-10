"""Mathematical invariant gates — the correctness firewall.

Every gate here is a statement that is *false* when an implementation is wrong,
so they genuinely bite (two real bugs in this codebase were caught this way).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import CertificateError, NumericalError
from ..core.linalg import assert_psd
from ..core.types import Problem
from ..data.synthetic import ground_truth_kkt
from .certificate import certificate, duality_gap_lp, suboptimality

__all__ = [
    "assert_finite",
    "assert_psd_problem",
    "assert_ground_truth_kkt",
    "assert_no_solution_beats_optimum",
    "assert_weak_duality_lp",
    "assert_monotone",
    "assert_improvement",
    "assert_deterministic",
    "assert_no_leak",
]


def assert_finite(*arrays: np.ndarray, name: str = "value") -> None:
    """All entries must be finite (no NaN / Inf anywhere)."""
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise NumericalError(
                f"{name} contains non-finite entries", count=int((~np.isfinite(arr)).sum())
            )


def assert_psd_problem(prob: Problem, tol: float = 1e-8) -> None:
    """The generated Hessian must be symmetric PSD."""
    assert_psd(prob.Q, tol=tol)


def assert_ground_truth_kkt(prob: Problem, tol: float = 1e-8) -> None:
    """The *constructed* optimum must satisfy KKT to machine level."""
    res = ground_truth_kkt(prob)
    worst = max(res.values())
    if worst > tol:
        raise CertificateError(
            f"generator ground truth violates KKT (max {worst:.3e} > {tol:.1e})", **res
        )


def assert_no_solution_beats_optimum(prob: Problem, x: np.ndarray, tol: float = 1e-6) -> None:
    """``f(x) >= f* - ||y*|| ||r_p|| - s*' max(0,-x) - tol``.

    Both slack terms are *required*, and getting this wrong is subtle: a slightly
    infeasible iterate can legitimately undershoot ``f*``, so a naive
    ``f(x) >= f*`` check fires on perfectly correct solvers. The iterate ``x`` is
    feasible for the perturbed problem::

        min f(x)  s.t.  A x = b + r_p,   x >= -eps,    eps = max(0, -x)

    and perturbing the equality RHS by ``r_p`` costs at most ``||y*|| ||r_p||``
    while relaxing the nonnegativity by ``eps`` costs at most ``s*' eps``.

    The inequality term is not optional: consensus ADMM only projects ``z``, so
    its ``x`` can sit slightly negative (observed 1.8e-4) and land 3.8e-4 *below*
    ``f*`` while being equality-feasible to 3e-14.
    """
    x = np.asarray(x, dtype=float)
    gap = suboptimality(prob, x)
    y_star = np.asarray(prob.meta["y_star"], dtype=float)
    s_star = np.asarray(prob.meta["s_star"], dtype=float)
    r_p = prob.A @ x - prob.b
    eps = np.maximum(-x, 0.0)
    slack = float(np.linalg.norm(y_star)) * float(np.linalg.norm(r_p)) + float(s_star @ eps)
    scale = max(1.0, abs(prob.f_star))
    if gap < -(slack + tol * scale):
        raise CertificateError(
            "solution beats the true optimum beyond the feasibility bound",
            gap=gap,
            bound=-(slack + tol * scale),
            primal_residual=float(np.linalg.norm(r_p)),
            nonneg_violation=float(np.max(eps)),
        )


def assert_weak_duality_lp(prob: Problem, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> None:
    """For LP instances at a *mutually feasible* pair: ``c'x - b'y >= -tol``.

    Weak duality requires **both** sides to be feasible, and checking only one is
    a real trap:

    * primal feasibility is ``Ax = b`` **and** ``x >= 0`` — consensus ADMM's ``x``
      can sit slightly negative (only ``z`` is projected);
    * dual feasibility requires the multiplier pair to satisfy ``A'y + s = c``
      with ``s >= 0``. Since the certificate always takes ``s = max(0, c - A'y)``,
      that holds iff ``min(c - A'y) >= 0``, i.e. iff the ``r_dual`` certificate
      component is zero.

    So the inequality is asserted only when primal *and* dual feasibility both
    hold; otherwise the comparison is meaningless and the gate stands down.
    """
    if float(np.max(np.abs(prob.Q))) > 0:
        return  # only meaningful for LPs (closed-form dual objective)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    r_p = float(np.max(np.abs(prob.A @ x - prob.b)))
    r_nonneg = float(np.max(np.maximum(-x, 0.0)))
    r_dual = prob.c - prob.A.T @ y
    r_dual_viol = float(np.max(np.maximum(-r_dual, 0.0)))
    if r_p > tol or r_nonneg > tol or r_dual_viol > tol:
        return  # not a mutually feasible pair => weak duality does not bind

    gap = duality_gap_lp(prob, x, y)
    scale = max(1.0, abs(float(prob.c @ x)), abs(float(prob.b @ y)))
    if gap < -tol * scale:
        raise CertificateError(
            f"weak duality violated (c'x - b'y = {gap:.3e})",
            gap=gap,
            primal_residual=r_p,
            nonneg_violation=r_nonneg,
            dual_violation=r_dual_viol,
        )


def assert_monotone(seq: list[float], tol: float = 1e-9, *, name: str = "sequence") -> None:
    """Assert a sequence is non-increasing (a *rise* is the violation)."""
    if len(seq) < 2:
        return
    diffs = np.diff(np.asarray(seq, dtype=float))
    worst = float(np.max(diffs))
    if worst > tol:
        raise NumericalError(
            f"{name} increased by {worst:.3e} (must be non-increasing)", worst_increase=worst
        )


def assert_improvement(
    start: float, final: float, factor: float, *, name: str = "residual"
) -> None:
    """Assert ``final`` improved on ``start`` by at least ``factor``."""
    start = max(float(start), 1e-300)
    if not (final <= start / factor):
        raise CertificateError(
            f"{name} did not improve {factor}x", start=start, final=final, factor=factor
        )


def assert_deterministic(a: dict[str, float], b: dict[str, float], tol: float = 1e-9) -> None:
    """Same seed must reproduce identical core scalars."""
    keys = set(a) & set(b)
    worst_key, worst = "", 0.0
    for k in keys:
        diff = abs(float(a[k]) - float(b[k]))
        if diff > worst:
            worst, worst_key = diff, k
    if worst > tol:
        raise NumericalError(
            f"determinism violated on {worst_key!r} (|Δ| = {worst:.3e})", key=worst_key, diff=worst
        )


def assert_no_leak(train_idx: list[int], test_idx: list[int]) -> None:
    """TRAIN and TEST problem indices must be disjoint."""
    overlap = sorted({int(i) for i in train_idx} & {int(i) for i in test_idx})
    if overlap:
        raise CertificateError("train/test index overlap (data leak)", overlap=overlap[:8])


def kkt_scale(prob: Problem) -> float:
    """Documented normalisation scale used by the certificate."""
    return max(
        1.0,
        float(np.max(np.abs(prob.b))),
        float(np.max(np.abs(prob.c))),
    )


def certified(prob: Problem, x: np.ndarray, y: np.ndarray, tol: float) -> bool:
    """True iff the iterate carries a KKT certificate tighter than ``tol``."""
    return certificate(prob, x, y)["cert"] <= tol
