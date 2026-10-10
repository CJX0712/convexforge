"""Chambolle-Pock primal-dual hybrid gradient (2011) with extrapolation.

Saddle form matching our stationarity convention exactly::

    min_{x >= 0} max_y  1/2 x'Qx + c'x + y'(b - A x)

so that the x-stationarity is ``Qx + c - A'y - s = 0`` and the returned ``y`` can
be fed straight into the shared certificate.

Iteration (primal extrapolation, theta = 1)::

    y   <- y + sigma (b - A x_bar)
    x+  <- proj_{x>=0}( x - tau (Q x + c - A'y) )
    x_bar <- x+ + theta (x+ - x)

Step sizes satisfy the standard Vũ-Condat / CP guards:
``tau <= 0.9 / L_Q`` for the explicit smooth gradient and
``tau*sigma*||A||^2 < 1`` for the bilinear coupling.

With ``adaptive=True`` we add a residual-driven **restart**: whenever the
certificate stops improving we kill the momentum (``x_bar <- x``) and shrink the
steps, which removes the characteristic primal-dual oscillation.
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import iterate_sane
from ..core.types import Problem, SolveResult
from ..eval.certificate import certificate

NAME = "pdhg"


def solve(
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    rho: float = 1.0,
    adaptive: bool = False,
) -> SolveResult:
    """Run PDHG and return the iterate with the smallest certified residual."""
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b
    n = prob.n

    # Curvature floor of 1.0 matters: for an LP (Q = 0) there is no quadratic
    # term, and using ||Q|| here would make tau explode to ~1e12 and blow the
    # primal step up (observed: certificate pinned at exactly 1.0).
    lip_q = max(float(np.linalg.norm(Q, 2)), 1.0)
    lip_a2 = float(np.linalg.norm(A, 2) ** 2)
    tau = 0.9 / lip_q
    sigma = 0.9 / (tau * max(lip_a2, 1e-12))

    x = np.ones(n)
    x_bar = x.copy()
    y = np.zeros(prob.m)

    best_x, best_y, best_cert = x.copy(), y.copy(), np.inf
    prev_cert = np.inf
    gap_trace: list[float] = []

    for it in range(int(max_iters)):
        y = y + sigma * (b - A @ x_bar)
        x_new = np.maximum(x - tau * (Q @ x + c - A.T @ y), 0.0)
        if not iterate_sane(x_new):
            break
        x_bar = x_new + (x_new - x)
        x = x_new

        cert = certificate(prob, x, y)["cert"]
        gap_trace.append(cert)

        if adaptive and it > 0 and cert > prev_cert:
            # Kill the momentum only — do NOT permanently shrink the steps.
            # (Empirically, shrinking tau/sigma here is strictly worse: it starves
            # the method and costs 2-3 orders of accuracy on `degen`.)
            x_bar = x.copy()
        prev_cert = cert

        if cert < best_cert:
            best_cert, best_x, best_y = cert, x.copy(), y.copy()
        if cert <= tol:
            break

    comp = certificate(prob, best_x, best_y)
    return SolveResult(
        method=NAME,
        x=best_x,
        y=best_y,
        iters=int(len(gap_trace)),
        cert=best_cert,
        obj=prob.objective(best_x),
        r_primal=comp["r_primal"],
        r_dual=comp["r_dual"],
        r_comp=comp["r_comp"],
        meta={"tau": tau, "sigma": sigma, "cert_trace": gap_trace, "adaptive": adaptive},
    )
