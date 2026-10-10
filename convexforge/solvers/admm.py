"""Alternating Direction Method of Multipliers (Boyd et al., 2011).

Consensus splitting of

    min 1/2 x'Qx + c'x   s.t.  A x = b,  x >= 0

into ``x`` carrying ``{Ax = b}`` and ``z`` carrying ``{z >= 0}``, tied by ``x = z``::

    x+ = argmin 1/2 x'Qx + c'x + (rho/2)||x - z + u||^2   s.t. Ax = b
    z+ = max(0, x+ + u)
    u  = u + (x+ - z+)

The x-step is a saddle-point solve of the constant-shape KKT system
``[[Q + rho I, A'], [A, 0]]``; the multiplier ``nu`` returned by that solve is the
natural estimate of the equality multiplier ``y``.

With ``adaptive=True`` we add Boyd's residual balancing (§3.4.1): rescale ``rho``
whenever primal/dual residuals drift apart by more than ``mu = 10``.
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import iterate_sane, solve_spd
from ..core.types import Problem, SolveResult
from ..eval.certificate import certificate

NAME = "admm"


def _kkt_solve(Q: np.ndarray, A: np.ndarray, rhs_top: np.ndarray, rhs_bot: np.ndarray, rho: float):
    """Solve [[Q+rho I, A'],[A,0]] [x;nu] = [rhs_top;rhs_bot]."""
    n, m = A.shape[1], A.shape[0]
    top = Q + rho * np.eye(n)
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = top
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A
    rhs = np.concatenate([rhs_top, rhs_bot])
    sol = solve_spd(KKT, rhs)
    return sol[:n], sol[n:]


def solve(
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    rho: float = 1.0,
    adaptive: bool = False,
) -> SolveResult:
    """Run consensus ADMM."""
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b
    n = prob.n

    z = np.zeros(n)
    u = np.zeros(n)
    best_x, best_y, best_cert = np.zeros(n), np.zeros(prob.m), np.inf
    residual_trace: list[float] = []

    for it in range(int(max_iters)):
        x, nu = _kkt_solve(Q, A, rho * (z - u) - c, b, rho)
        if not iterate_sane(x):
            break  # singular KKT on a rank-deficient A -> stop, keep best-so-far
        z_prev = z
        z = np.maximum(x + u, 0.0)
        u = u + x - z
        r_pri = float(np.linalg.norm(x - z))
        r_dual = rho * float(np.linalg.norm(z - z_prev))

        if adaptive and it % 10 == 9:
            if r_pri > 10.0 * max(r_dual, 1e-300):
                rho, u = rho * 2.0, u / 2.0
            elif r_dual > 10.0 * max(r_pri, 1e-300):
                rho, u = rho / 2.0, u * 2.0

        residual_trace.append(r_pri)

        # Multiplier recovery. Comparing the subproblem KKT
        #   (Q + rho I)x + A'nu = rho(z - u) - c
        # at the fixed point (z = x, u_i = 0 where x_i > 0, u_i <= 0 elsewhere)
        # against the true stationarity Qx + c - A'y - s = 0 gives
        #   y = -nu   and   s = -rho u >= 0   (s_i = 0 wherever x_i > 0),
        # i.e. complementarity holds automatically. Getting this sign wrong
        # leaves a dual residual of order 1.
        y = -nu
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
        iters=int(len(residual_trace)),
        cert=best_cert,
        obj=prob.objective(best_x),
        r_primal=comp["r_primal"],
        r_dual=comp["r_dual"],
        r_comp=comp["r_comp"],
        meta={"rho_final": rho, "residual_trace": residual_trace, "adaptive": adaptive},
    )
