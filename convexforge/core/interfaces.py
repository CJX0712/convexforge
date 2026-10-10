"""Solver interface contract.

Uniform certificate convention (see :mod:`convexforge.eval.certificate`)::

    every solver returns (x, y); the certificate machinery then builds
    s = max(0, Qx + c - A'y) and scores (primal, dual, complementarity).

Sign convention: Lagrangian L = 1/2 x'Qx + c'x + y'(b - Ax) - s'x, s >= 0,
so stationarity is Qx + c - A'y - s = 0.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core.types import Problem, SolveResult


class Solver(Protocol):
    """A convex solver on the canonical QP."""

    name: str

    def solve(
        self,
        prob: Problem,
        *,
        max_iters: int,
        tol: float,
        adaptive: bool = False,
    ) -> SolveResult:
        """Return a :class:`SolveResult` (x >= 0 always)."""
        ...


def _project_nonneg(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the nonnegative orthant."""
    return np.maximum(v, 0.0)
