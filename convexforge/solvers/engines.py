"""Engine registry and uniform dispatch."""

from __future__ import annotations

from ..core.types import Problem, SolveResult
from . import admm as admm_mod
from . import ipm as ipm_mod
from . import pdhg as pdhg_mod
from . import penalized as pen_mod
from .fuse import NAME as FUSE_NAME
from .fuse import RouteTable
from .fuse import solve as fuse_solve

ENGINE_NAMES: tuple[str, ...] = ("pen", "admm", "pdhg", "ipm", "convfuse")
#: Engines eligible for the "best component" comparison (excludes naive + flagship).
COMPONENTS: tuple[str, ...] = ("admm", "pdhg", "ipm")
NAIVE = "pen"
FLAGSHIP = FUSE_NAME


def base_engine(name: str):
    """Return the module implementing ``name``."""
    return {
        "pen": pen_mod,
        "admm": admm_mod,
        "pdhg": pdhg_mod,
        "ipm": ipm_mod,
    }[name]


def solve_one(
    name: str,
    prob: Problem,
    *,
    max_iters: int,
    tol: float,
    routes: RouteTable | None = None,
) -> SolveResult:
    """Solve ``prob`` with the named engine under a shared interface."""
    if name == FLAGSHIP:
        if routes is None:
            raise ValueError("convfuse requires a fitted RouteTable")
        return fuse_solve(prob, max_iters=max_iters, tol=tol, routes=routes)
    return base_engine(name).solve(prob, max_iters=max_iters, tol=tol)
