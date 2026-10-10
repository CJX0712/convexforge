"""Per-engine behaviour: correctness, convergence, sanity."""

from __future__ import annotations

import numpy as np
import pytest
from util import small_cfg

from convexforge.core.seed import set_all
from convexforge.data.synthetic import build_problem_set, ground_truth_kkt, make_problem
from convexforge.eval.certificate import certificate
from convexforge.solvers.engines import (
    COMPONENTS,
    ENGINE_NAMES,
    FLAGSHIP,
    base_engine,
    solve_one,
)
from convexforge.solvers.fuse import CANDIDATES, fit_router

TOL = 1e-10


def test_all_engines_registered():
    assert set(ENGINE_NAMES) == {"pen", "admm", "pdhg", "ipm", "convfuse"}
    assert FLAGSHIP == "convfuse"
    assert set(COMPONENTS) <= set(ENGINE_NAMES)


def test_every_engine_returns_finite_and_nonnegative_x():
    prob = make_problem("qp_pd", 0, 7, 40, 12)
    for name in ("pen", "admm", "pdhg", "ipm"):
        r = base_engine(name).solve(prob, max_iters=120, tol=TOL)
        assert np.all(np.isfinite(r.x)), name
        assert np.all(np.isfinite(r.y)), name
        assert r.cert >= 0.0


def test_ipm_solves_an_lp_to_high_accuracy():
    """The gold-standard path: Newton should nail a well-posed LP."""
    prob = make_problem("lp", 0, 7, 40, 12)
    r = base_engine("ipm").solve(prob, max_iters=120, tol=TOL)
    assert r.cert < 1e-8, r.cert
    assert r.r_dual < 1e-8
    assert r.r_comp < 1e-8


def test_ipm_recovers_the_true_optimum_on_a_well_posed_qp():
    prob = make_problem("qp_pd", 0, 7, 40, 12)
    r = base_engine("ipm").solve(prob, max_iters=120, tol=TOL)
    err = float(np.max(np.abs(r.x - prob.x_star)))
    assert err < 1e-4, f"||x - x*||_inf = {err}"


def test_penalty_objective_is_monotone():
    prob = make_problem("lp", 1, 7, 40, 12)
    r = base_engine("pen").solve(prob, max_iters=120, tol=TOL)
    tr = r.meta["penalty_trace"]
    assert len(tr) >= 2
    diffs = np.diff(np.asarray(tr))
    assert float(np.max(diffs)) <= 1e-9, float(np.max(diffs))


def test_diverging_iterate_is_stopped_not_returned():
    """ADMM on a rank-deficient A must not return garbage-sized iterates."""
    prob = make_problem("degen", 0, 7, 40, 12)
    r = base_engine("admm").solve(prob, max_iters=120, tol=TOL)
    assert np.all(np.isfinite(r.x))
    assert float(np.max(np.abs(r.x))) < 1e13


def test_router_is_fit_on_train_only_and_is_deterministic():
    cfg = small_cfg()
    set_all(cfg.seed)
    a, _ = build_problem_set("degen", 7, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
    rt1 = fit_router({"degen": a}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    rt2 = fit_router({"degen": a}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    assert rt1.table == rt2.table
    assert rt1.table["degen"] in CANDIDATES


def test_router_picks_the_train_best():
    cfg = small_cfg()
    train, _ = build_problem_set("degen", 7, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
    rt = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    ev = rt.evidence["degen"]
    assert rt.table["degen"] == min(ev, key=lambda k: ev[k])


def test_flagship_matches_its_routed_base():
    cfg = small_cfg()
    train, test = build_problem_set("degen", 7, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
    rt = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    prob = test[0]
    fused = solve_one(FLAGSHIP, prob, max_iters=cfg.max_iters, tol=cfg.cert_tol, routes=rt)
    base = solve_one(
        rt.base_for(prob.regime), prob, max_iters=cfg.max_iters, tol=cfg.cert_tol, routes=rt
    )
    assert fused.meta["routed_to"] == rt.base_for(prob.regime)
    assert fused.cert == pytest.approx(base.cert, rel=1e-12, abs=1e-15)


def test_flagship_requires_a_route_table():
    prob = make_problem("lp", 0, 7, 40, 12)
    with pytest.raises(ValueError, match="RouteTable"):
        solve_one(FLAGSHIP, prob, max_iters=10, tol=TOL)


def test_flagship_is_non_inferior_on_a_clean_instance():
    cfg = small_cfg(max_iters=200)
    prob = make_problem("degen", 0, 7, cfg.n_vars, cfg.n_cons)
    train, _ = build_problem_set("degen", 7, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
    rt = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    fused = solve_one(FLAGSHIP, prob, max_iters=cfg.max_iters, tol=cfg.cert_tol, routes=rt)
    best = min(
        base_engine(c).solve(prob, max_iters=cfg.max_iters, tol=cfg.cert_tol).cert
        for c in COMPONENTS
    )
    assert fused.cert <= best * 1.02


def test_certificate_is_scale_free():
    """A uniform rescaling of the objective must leave the certificate unchanged.

    The scaling that *preserves* KKT exactly is::

        Q -> tQ,  c -> tc,  y* -> t y*,  s* -> t s*,  f* -> t f*,  (A, b, x*) fixed

    because stationarity ``Qx + c - A'y - s = 0`` is then multiplied through by
    ``t`` while primal feasibility and complementarity are untouched. Scaling
    ``b`` alone would break ``Ax = b`` and is *not* a valid rescaling of the
    instance — this test deliberately checks the valid one.
    """
    t = 100.0
    prob = make_problem("qp_pd", 0, 7, 40, 12)
    x, y = prob.x_star, prob.meta["y_star"]
    base = certificate(prob, x, y)["cert"]

    scaled = make_problem("qp_pd", 0, 7, 40, 12)
    scaled.Q = prob.Q * t
    scaled.c = prob.c * t
    scaled.f_star = prob.f_star * t
    y_scaled = y * t
    scaled.meta["y_star"] = y_scaled
    scaled.meta["s_star"] = scaled.meta["s_star"] * t

    # ground truth still KKT-exact after the rescaling
    assert max(ground_truth_kkt(scaled).values()) < 1e-10
    got = certificate(scaled, x, y_scaled)["cert"]
    assert got < max(base * 10.0, 1e-10)
