"""Primal-dual interior point method (Mehrotra predictor-corrector).

KKT for  min 1/2 x'Qx + c'x  s.t. Ax = b, x >= 0::

    Qx + c - A'y - s = 0,   Ax = b,   x >= 0, s >= 0,   x_i s_i = 0

Newton step for target duality gap ``sigma*mu``::

    [ Q   -A'  -I ] [dx]   [ -r_d        ]
    [ A    0    0 ] [dy] = [ r_p         ],   r_d = Qx+c-A'y-s, r_p = b-Ax
    [ S    0    X ] [ds]   [ sigma*mu*e - XSe ]

Eliminating ``ds`` gives the reduced system with ``M = Q + diag(s/x)`` (PD since
``s,x > 0``)::

    (A M^{-1} A') dy = r_p - A M^{-1} (-r_d + r_c/x)
    dx = M^{-1}(A'dy - r_d + r_c/x),    ds = r_c/x - (s/x)⊙dx

with ``r_c = sigma*mu*e - x⊙s`` (minus the second-order term
``dx_aff ⊙ ds_aff`` in the corrector pass).

The complementarity gap ``mu = x's/n`` is tracked; ``eta`` is the fraction-to-
boundary safeguard.
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import iterate_sane, solve_spd
from ..core.types import Problem, SolveResult
from ..eval.certificate import certificate

NAME = "ipm"
_ETA = 0.9995


def _fraction_to_boundary(v: np.ndarray, dv: np.ndarray) -> float:
    """Largest ``alpha`` in (0, 1] keeping ``v + alpha dv >= 0``."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(np.min(-v[neg] / dv[neg]))


def _newton_direction(
    Q: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    r_d: np.ndarray,
    r_p: np.ndarray,
    sigma: float,
    mu: float,
    corr: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One reduced Newton solve. Returns ``(dx, dy, ds)``."""
    n = len(x)
    r_c = sigma * mu * np.ones(n) - x * s
    if corr is not None:
        r_c = r_c - corr[0] * corr[1]

    M = Q + np.diag(s / x)
    rhs1 = -r_d + r_c / x
    v1 = solve_spd(M, rhs1)
    Z = solve_spd(M, A.T)
    normal = A @ Z
    dy = solve_spd(normal, r_p - A @ v1)
    dx = Z @ dy + v1
    ds = r_c / x - (s / x) * dx
    return dx, dy, ds


def solve(
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    rho: float = 1.0,
    adaptive: bool = False,
) -> SolveResult:
    """Run the Mehrotra predictor-corrector interior point method."""
    Q, c, A, b = prob.Q, prob.c, prob.A, prob.b
    n = prob.n

    x = np.ones(n)
    y = np.zeros(prob.m)
    s = np.ones(n)

    best_x, best_y, best_cert = x.copy(), y.copy(), np.inf
    mu_trace: list[float] = []

    for _ in range(int(max_iters)):
        x = np.maximum(x, 1e-12)
        s = np.maximum(s, 1e-12)
        r_d = Q @ x + c - A.T @ y - s
        r_p = b - A @ x
        mu = float(x @ s) / n
        mu_trace.append(mu)

        cert = certificate(prob, x, y)["cert"]
        if cert < best_cert:
            best_cert, best_x, best_y = cert, x.copy(), y.copy()
        if cert <= tol or mu <= 1e-14:
            break

        # --- predictor (affine scaling, sigma = 0)
        try:
            dx_a, dy_a, ds_a = _newton_direction(Q, A, x, s, r_d, r_p, 0.0, mu)
        except Exception:  # noqa: BLE001 - singular normal matrix, stop cleanly
            break
        if not all(np.all(np.isfinite(v)) for v in (dx_a, dy_a, ds_a)):
            break

        a_p = _fraction_to_boundary(x, dx_a)
        a_d = _fraction_to_boundary(s, ds_a)
        mu_aff = float((x + a_p * dx_a) @ (s + a_d * ds_a)) / n
        sigma = float(np.clip((mu_aff / mu) ** 3, 1e-8, 0.9)) if mu > 0 else 0.1

        # --- Mehrotra corrector (with second-order term)
        try:
            dx, dy, ds = _newton_direction(Q, A, x, s, r_d, r_p, sigma, mu, corr=(dx_a, ds_a))
        except Exception:  # noqa: BLE001
            dx, dy, ds = dx_a, dy_a, ds_a
        if not all(np.all(np.isfinite(v)) for v in (dx, dy, ds)):
            break

        alpha_p = min(1.0, _ETA * _fraction_to_boundary(x, dx))
        alpha_d = min(1.0, _ETA * _fraction_to_boundary(s, ds))
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds
        if not (iterate_sane(x) and iterate_sane(s)):
            break

    comp = certificate(prob, best_x, best_y)
    return SolveResult(
        method=NAME,
        x=best_x,
        y=best_y,
        iters=int(len(mu_trace)),
        cert=best_cert,
        obj=prob.objective(best_x),
        r_primal=comp["r_primal"],
        r_dual=comp["r_dual"],
        r_comp=comp["r_comp"],
        meta={"mu_trace": mu_trace, "adaptive": adaptive},
    )
