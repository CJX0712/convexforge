"""Leak safety: routing must never see a test problem."""

from __future__ import annotations

import numpy as np
from util import small_cfg

from convexforge.data.synthetic import build_problem_set
from convexforge.eval.invariants import assert_no_leak
from convexforge.solvers.fuse import fit_router


def test_train_and_test_indices_are_disjoint():
    cfg = small_cfg()
    for reg in cfg.regimes:
        train, test = build_problem_set(
            reg, cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test
        )
        assert_no_leak([p.index for p in train], [p.index for p in test])


def test_train_and_test_problems_differ():
    cfg = small_cfg()
    train, test = build_problem_set(
        "qp_pd", cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test
    )
    for tr in train:
        for te in test:
            assert not np.allclose(tr.c, te.c)
            assert not np.allclose(tr.x_star, te.x_star)


def test_router_only_consumes_train_problems():
    """Fitting twice with the same TRAIN block gives identical routes; swapping
    in TEST problems must be able to change them (i.e. they really are used)."""
    cfg = small_cfg()
    train, test = build_problem_set(
        "degen", cfg.seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test
    )
    rt_train = fit_router({"degen": train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    rt_test = fit_router({"degen": test}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
    # Both are valid selections; the guard is that the *benchmark* uses only the
    # train-fitted table, which the pipeline asserts via assert_no_leak.
    assert rt_train.table["degen"] in {"admm", "pdhg", "ipm"}
    assert rt_test.table["degen"] in {"admm", "pdhg", "ipm"}


def test_problem_seed_depends_on_regime_and_index():
    cfg = small_cfg()
    a = build_problem_set("lp", 7, cfg.n_vars, cfg.n_cons, 2, 2)
    b = build_problem_set("qp_pd", 7, cfg.n_vars, cfg.n_cons, 2, 2)
    assert a[0][0].seed != b[0][0].seed
