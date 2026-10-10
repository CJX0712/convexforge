"""ConvFuse — the flagship: cross-validated regime-routed solver selection.

Why routing is worth anything at all
------------------------------------
No single operator dominates. Measured (geomean certificate, 200-iteration budget,
n=80/m=30):

    regime        pen        admm       pdhg       ipm        winner
    lp            2.8e-01    4.0e-05    5.7e-03    3.2e-12    ipm
    qp_pd         1.7e-01    3.0e-07    7.5e-07    7.8e-11    ipm
    qp_singular   1.4e-01    1.7e-06    1.5e-03    1.1e-08    ipm
    degen         2.5e-01    7.7e-01    9.6e-09    5.9e-03    pdhg

``degen`` is the discriminating regime: its constraint matrix is rank deficient,
so the interior-point normal matrix ``A M^-1 A'`` is singular and Newton's method
stalls, while the purely first-order Chambolle-Pock scheme is unaffected. Routing
therefore gains roughly **1.4 orders of magnitude** over the best single engine
without ever seeing a test problem.

Leak safety
-----------
``fit_router`` only ever sees TRAIN problems; the benchmark then scores every
engine on disjoint TEST problems of the same regime.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Problem, SolveResult
from ..eval.certificate import certificate
from ..eval.metrics import geometric_mean
from . import admm as admm_mod
from . import ipm as ipm_mod
from . import pdhg as pdhg_mod

NAME = "convfuse"

#: Solvers the router is allowed to choose from (``pen`` is the naive baseline).
CANDIDATES: dict[str, object] = {
    "admm": admm_mod,
    "pdhg": pdhg_mod,
    "ipm": ipm_mod,
}


class RouteTable:
    """Maps a regime label to the base engine chosen on TRAIN data."""

    def __init__(self, table: dict[str, str], evidence: dict[str, dict[str, float]]) -> None:
        self.table = table
        self.evidence = evidence

    def base_for(self, regime: str) -> str:
        return self.table.get(regime, "ipm")

    def as_dict(self) -> dict:
        return {"route": dict(self.table), "evidence": self.evidence}


def fit_router(
    train_by_regime: dict[str, list[Problem]],
    *,
    max_iters: int,
    tol: float,
) -> RouteTable:
    """Pick, per regime, the base engine with the lowest TRAIN certificate."""
    table: dict[str, str] = {}
    evidence: dict[str, dict[str, float]] = {}
    for regime, probs in train_by_regime.items():
        scores: dict[str, float] = {}
        for cand, mod in CANDIDATES.items():
            certs = [mod.solve(p, max_iters=max_iters, tol=tol).cert for p in probs]
            scores[cand] = geometric_mean(certs)
        best = min(scores, key=lambda k: scores[k])
        table[regime] = best
        evidence[regime] = scores
    return RouteTable(table, evidence)


def solve(
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    routes: RouteTable | None = None,
    adaptive: bool = False,
) -> SolveResult:
    """Dispatch to the routed base engine and relabel the result."""
    if routes is None:
        routes = RouteTable({}, {})
    base_name = routes.base_for(prob.regime)
    mod = CANDIDATES[base_name]
    res = mod.solve(prob, max_iters=max_iters, tol=tol, adaptive=adaptive)
    comp = certificate(prob, res.x, res.y)
    return SolveResult(
        method=NAME,
        x=res.x,
        y=res.y,
        iters=res.iters,
        cert=res.cert,
        obj=res.obj,
        r_primal=comp["r_primal"],
        r_dual=comp["r_dual"],
        r_comp=comp["r_comp"],
        meta={**res.meta, "routed_to": base_name},
    )


def fingerprint(prob: Problem, x: np.ndarray) -> float:
    """Deterministic scalar used in the determinism gate."""
    return float(np.sum(np.abs(x)) + prob.f_star)
