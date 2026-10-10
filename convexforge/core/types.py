"""Unified problem / result data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    """Canonical convex QP:  min 1/2 x'Qx + c'x  s.t. A x = b, x >= 0.

    ``Q`` is symmetric PSD (Q = 0 gives an LP). Instances are generated from a
    *known* optimal primal/dual triple so that ``f_star`` and ``x_star`` are
    exact — enabling certified ground-truth gates rather than self-reported ones.
    """

    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    x_star: np.ndarray
    f_star: float
    regime: str
    index: int
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ (self.Q @ x) + self.c @ x)


@dataclass
class SolveResult:
    """Outcome of one solver run on one problem."""

    method: str
    x: np.ndarray
    y: np.ndarray
    iters: int
    cert: float
    obj: float
    r_primal: float
    r_dual: float
    r_comp: float
    meta: dict = field(default_factory=dict)


@dataclass
class BenchmarkRow:
    """One aggregated measurement row."""

    method: str
    regime: str
    seed: int
    cert: float
    subopt: float
    iters: int
