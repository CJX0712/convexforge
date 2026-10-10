"""Ten hard invariant gates. Anything that is false implies a wrong implementation."""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.linalg import assert_psd
from ..core.seed import set_all
from ..data.synthetic import build_problem_set, ground_truth_kkt, make_problem
from ..eval.certificate import certificate, duality_gap_lp, suboptimality
from ..eval.invariants import (
    assert_deterministic,
    assert_finite,
    assert_ground_truth_kkt,
    assert_improvement,
    assert_no_leak,
    assert_no_solution_beats_optimum,
    assert_psd_problem,
    assert_weak_duality_lp,
)
from ..solvers.engines import ENGINE_NAMES, solve_one
from ..solvers.fuse import fit_router

Check = tuple[str, bool, str]


def _cfg_small() -> Config:
    cfg = Config(n_vars=40, n_cons=12, n_train=2, n_test=2, max_iters=60, n_seeds=1)
    cfg.validate()
    return cfg


def run_selftest() -> dict:
    """Run all gates and return ``{"checks": [...], "all_pass": bool}``."""
    checks: list[Check] = []
    set_all(7)

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # 1/2: generator produces a well-posed problem with exact ground truth.
    prob = make_problem("qp_pd", 0, 7, 40, 12)
    try:
        assert_psd_problem(prob)
        record("psd_problem", True)
    except Exception as exc:  # noqa: BLE001
        record("psd_problem", False, str(exc))
    try:
        assert_ground_truth_kkt(prob)
        record("ground_truth_kkt", True)
    except Exception as exc:  # noqa: BLE001
        record("ground_truth_kkt", False, str(exc))

    # 3: at the true optimum the certificate must vanish.
    cert_star = certificate(prob, prob.x_star, prob.meta["y_star"])["cert"]
    record("certificate_at_optimum", cert_star < 1e-8, f"cert={cert_star:.2e}")

    # 4: an honest LP must satisfy weak duality at a feasible iterate.
    lp = make_problem("lp", 0, 7, 40, 12)
    from ..solvers.engines import base_engine

    res = base_engine("ipm").solve(lp, max_iters=60, tol=1e-10)
    try:
        assert_weak_duality_lp(lp, res.x, res.y, tol=1e-6)
        record("weak_duality_lp", True)
    except Exception as exc:  # noqa: BLE001
        record("weak_duality_lp", False, str(exc))

    # 5: no solver may beat the true optimum beyond its feasibility bound.
    bad: list[str] = []
    cfg = _cfg_small()
    for method in ENGINE_NAMES:
        pp = make_problem("qp_pd", 0, 7, cfg.n_vars, cfg.n_cons)
        if method == "convfuse":
            tr, _ = build_problem_set("qp_pd", 7, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
            rt = fit_router({"qp_pd": tr}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
            r = solve_one(method, pp, max_iters=cfg.max_iters, tol=cfg.cert_tol, routes=rt)
        else:
            r = solve_one(method, pp, max_iters=cfg.max_iters, tol=cfg.cert_tol)
        try:
            assert_no_solution_beats_optimum(pp, r.x, tol=1e-5)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{method}: {exc}")
    record("no_solution_beats_optimum", not bad, "; ".join(bad))

    # 6: the penalty method's objective is non-increasing (true descent method).
    pen_res = base_engine("pen").solve(make_problem("lp", 1, 7, 40, 12), max_iters=60, tol=1e-10)
    pen_trace = pen_res.meta.get("penalty_trace", [])
    diffs = np.diff(np.asarray(pen_trace, dtype=float)) if len(pen_trace) > 1 else np.zeros(1)
    worst_rise = float(np.max(diffs)) if len(diffs) else 0.0
    record("pen_objective_monotone", worst_rise <= 1e-9, f"max rise {worst_rise:.2e}")

    # 7: interior point must actually improve the certificate from a cold start.
    p2 = make_problem("qp_pd", 2, 7, 40, 12)
    one = base_engine("ipm").solve(p2, max_iters=1, tol=1e-12)
    full = base_engine("ipm").solve(p2, max_iters=60, tol=1e-12)
    try:
        assert_improvement(one.cert, full.cert, 10.0, name="ipm certificate")
        record("ipm_improves", True)
    except Exception as exc:  # noqa: BLE001
        record("ipm_improves", False, str(exc))

    # 8: determinism — identical seed must reproduce identical numbers.
    set_all(11)
    a = base_engine("admm").solve(make_problem("qp_pd", 3, 11, 40, 12), max_iters=40, tol=1e-12)
    set_all(11)
    b = base_engine("admm").solve(make_problem("qp_pd", 3, 11, 40, 12), max_iters=40, tol=1e-12)
    try:
        assert_deterministic(
            {"cert": a.cert, "obj": a.obj, "sum_x": float(np.sum(a.x))},
            {"cert": b.cert, "obj": b.obj, "sum_x": float(np.sum(b.x))},
        )
        record("deterministic", True)
    except Exception as exc:  # noqa: BLE001
        record("deterministic", False, str(exc))

    # 9: routing is fit on TRAIN problems disjoint from TEST.
    tr, te = build_problem_set("lp", 7, 40, 12, 2, 2)
    try:
        assert_no_leak([p.index for p in tr], [p.index for p in te])
        record("no_leak", True)
    except Exception as exc:  # noqa: BLE001
        record("no_leak", False, str(exc))

    # 10: every engine returns finite iterates on a hard regime.
    hard = make_problem("degen", 0, 7, 40, 12)
    finite_bad: list[str] = []
    for method in ("pen", "admm", "pdhg", "ipm"):
        r = base_engine(method).solve(hard, max_iters=40, tol=1e-12)
        try:
            assert_finite(r.x, r.y, name=method)
        except Exception as exc:  # noqa: BLE001
            finite_bad.append(str(exc))
    record("finite_all_engines", not finite_bad, "; ".join(finite_bad))

    # unused local aliases keep lint happy without changing behaviour
    _ = (suboptimality, duality_gap_lp, ground_truth_kkt, assert_psd)
    return {"checks": checks, "all_pass": all(ok for _, ok, _ in checks)}
