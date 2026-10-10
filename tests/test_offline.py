"""Offline fallback: the whole stack must run on pure NumPy, zero downloads.

Tier-1 route: no scipy / sklearn / torch / cvxpy import anywhere in the package,
so the demo runs on a bare NumPy install with no network and no compiled kernels.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from convexforge.data.synthetic import make_problem
from convexforge.solvers.engines import ENGINE_NAMES, base_engine

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "convexforge"
FORBIDDEN = ("scipy", "sklearn", "torch", "cvxpy", "osqp", "ecos", "scs")


def test_no_heavy_third_party_imports():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for mod in FORBIDDEN:
            if re.search(rf"^\s*(import|from)\s+{mod}\b", text, re.MULTILINE):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}: {mod}")
    assert not offenders, offenders


def test_all_engines_run_on_pure_numpy():
    """Every component solves a problem without any optional backend."""
    prob = make_problem("qp_pd", 0, 7, 40, 12)
    for name in ("pen", "admm", "pdhg", "ipm"):
        r = base_engine(name).solve(prob, max_iters=60, tol=1e-10)
        assert np.all(np.isfinite(r.x))
        assert r.cert >= 0.0


def test_numpy_only_linear_algebra_is_used():
    """The SPD solve falls back to least squares rather than requiring LAPACK extras."""
    from convexforge.core.linalg import solve_spd

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    z = solve_spd(A, np.array([1.0, 2.0]))
    assert np.allclose(A @ z, [1.0, 2.0], atol=1e-10)


def test_singular_system_still_returns_finite():
    from convexforge.core.linalg import solve_spd

    A = np.zeros((3, 3))  # maximally singular
    z = solve_spd(A, np.ones(3))
    assert np.all(np.isfinite(z))


def test_engine_names_stable():
    assert ENGINE_NAMES == ("pen", "admm", "pdhg", "ipm", "convfuse")


def test_certificate_works_without_solver_backends():
    from convexforge.eval.certificate import certificate

    prob = make_problem("lp", 0, 7, 40, 12)
    comp = certificate(prob, prob.x_star, prob.meta["y_star"])
    assert set(comp) == {"r_primal", "r_dual", "r_comp", "r_nonneg", "cert"}
    assert comp["cert"] < 1e-8


@pytest.mark.parametrize("regime", ["lp", "qp_pd", "qp_singular", "degen"])
def test_every_regime_is_solvable_offline(regime):
    prob = make_problem(regime, 0, 7, 40, 12)
    r = base_engine("pdhg").solve(prob, max_iters=60, tol=1e-10)
    assert np.all(np.isfinite(r.x))
