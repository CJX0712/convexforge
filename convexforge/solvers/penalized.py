"""Naive baseline: quadratic-penalty projected gradient descent.

    min_x  Phi(x) = 1/2 x'Qx + c'x + (rho/2)||Ax - b||^2   s.t. x >= 0

The constraint ``Ax = b`` is only *penalised*, never enforced, so this method
converges slowly and plateaus well above solver-grade accuracy — exactly what a
honest naive baseline should do. Its multiplier estimate follows from comparing
its stationarity ``Qx + c + rho A'(Ax-b) - s = 0`` with the true one::

    y = -rho (A x - b)

The penalised objective is provably non-increasing under the step ``1/L`` with
``L = ||Q||_2 + rho ||A||_2^2`` (hard monotonicity invariant).
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import iterate_sane
from ..core.types import Problem, SolveResult
from ..eval.certificate import certificate

NAME = "pen"


def _penalized_objective(prob: Problem, x: np.ndarray, rho: float) -> float:
    resid = prob.A @ x - prob.b
    return 0.5 * x @ (prob.Q @ x) + prob.c @ x + 0.5 * rho * float(resid @ resid)


def solve(
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    rho: float = 1.0,
    adaptive: bool = False,
) -> SolveResult:
    """Run the naive penalty method for ``max_iters`` projected gradient steps."""
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b
    n = prob.n
    x = np.ones(n)

    lip = float(np.linalg.norm(Q, 2)) + rho * float(np.linalg.norm(A, 2) ** 2)
    step = 1.0 / max(lip, 1e-12)

    best_x, best_y, best_cert = x.copy(), np.zeros(prob.m), np.inf
    pen_trace: list[float] = []

    for _ in range(int(max_iters)):
        grad = Q @ x + c + rho * (A.T @ (A @ x - b))
        x = np.maximum(x - step * grad, 0.0)
        if not iterate_sane(x):
            break
        pen_trace.append(_penalized_objective(prob, x, rho))

        y = -rho * (A @ x - b)
        cert = certificate(prob, x, y)["cert"]
        if cert < best_cert:
            best_cert, best_x, best_y = cert, x.copy(), y.copy()
        if cert <= tol:
            break

    comp = certificate(prob, best_x, best_y)
    return SolveResult(
        method=NAME,
        x=best_x,
        y=best_y,
        iters=int(len(pen_trace)),
        cert=best_cert,
        obj=prob.objective(best_x),
        r_primal=comp["r_primal"],
        r_dual=comp["r_dual"],
        r_comp=comp["r_comp"],
        meta={"penalty_trace": pen_trace, "rho": rho},
    )
