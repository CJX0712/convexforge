"""Determinism: one seed must reproduce identical numbers."""

from __future__ import annotations

import numpy as np
from util import small_cfg

from convexforge.core.seed import set_all
from convexforge.data.synthetic import build_problem_set
from convexforge.eval.invariants import assert_deterministic
from convexforge.pipeline.benchmark import run_benchmark
from convexforge.solvers.engines import base_engine
from convexforge.solvers.fuse import fit_router


def _scalars(res) -> dict[str, float]:
    return {
        "cert": res.cert,
        "obj": res.obj,
        "sum_x": float(np.sum(res.x)),
        "sum_y": float(np.sum(res.y)),
    }


def test_each_engine_is_bitwise_reproducible():
    cfg = small_cfg()
    for reg in ("lp", "qp_pd", "degen"):
        for name in ("pen", "admm", "pdhg", "ipm"):
            set_all(cfg.seed)
            a = base_engine(name).solve(
                build_problem_set(reg, cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)[
                    1
                ][0],
                max_iters=cfg.max_iters,
                tol=cfg.cert_tol,
            )
            set_all(cfg.seed)
            b = base_engine(name).solve(
                build_problem_set(reg, cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)[
                    1
                ][0],
                max_iters=cfg.max_iters,
                tol=cfg.cert_tol,
            )
            assert_deterministic(_scalars(a), _scalars(b))


def test_router_fit_is_reproducible():
    cfg = small_cfg()
    train, _ = build_problem_set("degen", cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
    set_all(cfg.seed)
    r1 = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    set_all(cfg.seed)
    r2 = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    assert r1.table == r2.table


def test_full_benchmark_is_reproducible():
    cfg = small_cfg(max_iters=60, n_vars=30, n_cons=8, n_train=1, n_test=1)
    a = run_benchmark(cfg)
    b = run_benchmark(cfg)
    keys = [f"per_method.{m}.log10_mean" for m in a["per_method"]]
    da = {k: a["per_method"][k.split(".")[1]]["log10_mean"] for k in keys}
    db = {k: b["per_method"][k.split(".")[1]]["log10_mean"] for k in keys}
    assert_deterministic(da, db)
    assert a["headline"]["orders_vs_best_component"] == b["headline"]["orders_vs_best_component"]
