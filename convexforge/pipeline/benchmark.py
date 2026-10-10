"""Deterministic multi-regime, multi-seed benchmark with pre-registered gates.

Gate definition (fixed *before* looking at results)
--------------------------------------------------
Certificates span many orders of magnitude (1e-1 for the naive baseline down to
1e-12 for a converged interior point), so the naive relative reduction
``(base - flagship) / base`` is degenerate — it reads ~100% for every engine.
We therefore pre-register **log-order** gates, which are meaningful in that regime:

* **G1** flagship must beat the *best single component* by >= 0.5 orders of
  magnitude, i.e. ``log10(cert_best_component / cert_flagship) >= 0.5``;
* **G2** flagship must beat the naive baseline by >= 3 orders of magnitude;
* **G3** non-inferiority: ``cert_flagship <= 1.02 x cert_best_component``.

All statistics are reported in log10 space as mean +/- std over every held-out
test problem across all regimes and seeds (>= 3 seeds, per gate design).
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import BenchmarkRow
from ..data.synthetic import REGIMES, build_problem_set
from ..eval.certificate import suboptimality
from ..eval.invariants import (
    assert_finite,
    assert_ground_truth_kkt,
    assert_no_leak,
    assert_no_solution_beats_optimum,
    assert_psd_problem,
    assert_weak_duality_lp,
)
from ..eval.metrics import geometric_mean, mean_std
from ..solvers.engines import COMPONENTS, ENGINE_NAMES, FLAGSHIP, NAIVE, solve_one
from ..solvers.fuse import fit_router

#: Root-cause attribution for failure cases, keyed by (method, regime) then method.
_EXPLAIN: dict = {
    ("ipm", "degen"): (
        "rank(A) < m makes the normal matrix A M^-1 A' singular, so the Newton "
        "direction is computed from a jittered system and stalls"
    ),
    ("admm", "degen"): (
        "the consensus x-step solves [[Q+rho I, A'],[A,0]], which is singular when "
        "A is rank deficient, corrupting every iterate"
    ),
    ("pdhg", "lp"): (
        "Q = 0 removes all curvature in x, so Chambolle-Pock loses linear "
        "convergence and only creeps sublinearly"
    ),
    ("pdhg", "qp_singular"): (
        "no curvature along the null directions of Q, giving a long slow tail"
    ),
    ("pen", "lp"): (
        "Ax = b is only penalised, never enforced, so the certificate plateaus "
        "around 0.25 regardless of iterations"
    ),
}
_DEFAULT_EXPLAIN = {
    "pen": "only penalises Ax = b instead of enforcing it; no dual certificate is ever produced",
    "admm": "first-order splitting: linear rate only with a favourable rho, and its x-step needs a nonsingular KKT block",
    "pdhg": "first-order splitting: sublinear tail once coordinate curvature vanishes",
    "ipm": "Newton-based: excellent when the normal matrix is well conditioned, fragile otherwise",
}


def _log10(v: float) -> float:
    return float(np.log10(max(float(v), 1e-16)))


def run_benchmark(cfg: Config) -> dict:
    """Run the full benchmark and return a structured result dictionary."""
    cfg.validate()
    rows: list[BenchmarkRow] = []
    route_tables: dict[str, str] = {}
    ablation: dict[str, list[float]] = {}
    wall: dict[str, float] = {}

    for seed_off in range(int(cfg.n_seeds)):
        seed = int(cfg.seed) + seed_off
        for regime in list(cfg.regimes):
            assert regime in REGIMES, f"unknown regime {regime!r}"
            train, test = build_problem_set(
                regime, seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test
            )
            # --- leak guard: routing is fit on TRAIN problems only
            assert_no_leak([p.index for p in train], [p.index for p in test])
            for prob in train:
                assert_psd_problem(prob)
                assert_ground_truth_kkt(prob)
            routes = fit_router({regime: train}, max_iters=cfg.max_iters, tol=cfg.cert_tol)
            route_tables[f"{regime}/seed{seed}"] = routes.table[regime]

            for method in ENGINE_NAMES:
                for prob in test:
                    assert_psd_problem(prob)
                    res = solve_one(
                        method, prob, max_iters=cfg.max_iters, tol=cfg.cert_tol, routes=routes
                    )
                    assert_finite(res.x, res.y, name=f"{method}.iterate")
                    assert_no_solution_beats_optimum(prob, res.x, tol=1e-5)
                    assert_weak_duality_lp(prob, res.x, res.y, tol=1e-5)
                    rows.append(
                        BenchmarkRow(
                            method=method,
                            regime=regime,
                            seed=seed,
                            cert=res.cert,
                            subopt=suboptimality(prob, res.x),
                            iters=res.iters,
                        )
                    )

            # --- ablation: are the adaptive variants actually worth adopting?
            # Restricted to the first seed purely to keep the demo inside its
            # 60 s budget; the conclusion is computed on held-out TEST problems.
            if seed_off == 0:
                for base in ("pdhg", "admm"):
                    for prob in test:
                        for tag, ada in (("plain", False), ("adaptive", True)):
                            from ..solvers.engines import base_engine  # local: avoid cycle

                            r = base_engine(base).solve(
                                prob,
                                max_iters=cfg.max_iters,
                                tol=cfg.cert_tol,
                                adaptive=ada,
                            )
                            ablation.setdefault(f"{base}:{tag}", []).append(r.cert)

    return _aggregate(rows, cfg, ablation, route_tables, wall)


def _aggregate(
    rows: list[BenchmarkRow],
    cfg: Config,
    ablation: dict[str, list[float]],
    route_tables: dict[str, str],
    wall: dict[str, float],
) -> dict:
    per_method: dict[str, dict] = {}
    for m in ENGINE_NAMES:
        vals = [r.cert for r in rows if r.method == m]
        logs = [_log10(v) for v in vals]
        logstats = mean_std(logs)
        per_method[m] = {
            "geomean": geometric_mean(vals),
            "log10_mean": logstats["mean"],
            "log10_std": logstats["std"],
            "n": len(vals),
            "mean_iters": float(np.mean([r.iters for r in rows if r.method == m])),
            "mean_subopt": float(np.mean([r.subopt for r in rows if r.method == m])),
        }

    per_regime: dict[str, dict] = {}
    for regime in list(cfg.regimes):
        per_regime[regime] = {}
        for m in ENGINE_NAMES:
            vals = [r.cert for r in rows if r.method == m and r.regime == regime]
            per_regime[regime][m] = geometric_mean(vals) if vals else float("nan")

    naive = per_method[NAIVE]["geomean"]
    flagship = per_method[FLAGSHIP]["geomean"]
    comp_stats = {c: per_method[c] for c in COMPONENTS}
    best_name = min(COMPONENTS, key=lambda c: per_method[c]["geomean"])
    best_component = per_method[best_name]["geomean"]

    orders_vs_component = _log10(best_component) - _log10(flagship)
    orders_vs_naive = _log10(naive) - _log10(flagship)
    non_inferior = flagship <= best_component * 1.02

    gate_detail = {
        "G1_orders_vs_best_component": {
            "value": orders_vs_component,
            "threshold": 0.5,
            "pass": bool(orders_vs_component >= 0.5),
        },
        "G2_orders_vs_naive": {
            "value": orders_vs_naive,
            "threshold": 3.0,
            "pass": bool(orders_vs_naive >= 3.0),
        },
        "G3_non_inferior": {"value": non_inferior, "pass": bool(non_inferior)},
    }

    return {
        "config": {
            "seed": cfg.seed,
            "n_seeds": cfg.n_seeds,
            "n_vars": cfg.n_vars,
            "n_cons": cfg.n_cons,
            "n_train": cfg.n_train,
            "n_test": cfg.n_test,
            "max_iters": cfg.max_iters,
            "cert_tol": cfg.cert_tol,
            "regimes": list(cfg.regimes),
        },
        "rows": [
            {
                "method": r.method,
                "regime": r.regime,
                "seed": r.seed,
                "cert": r.cert,
                "subopt": r.subopt,
                "iters": r.iters,
            }
            for r in rows
        ],
        "per_method": per_method,
        "per_regime": per_regime,
        "ablation": _summarise_ablation(ablation),
        "headline": {
            "naive_pen": per_method[NAIVE],
            "flagship_convfuse": per_method[FLAGSHIP],
            "best_component": {"method": best_name, **comp_stats[best_name]},
            "orders_vs_best_component": float(orders_vs_component),
            "orders_vs_naive": float(orders_vs_naive),
            "non_inferior": bool(non_inferior),
            "gate_pass": bool(all(g["pass"] for g in gate_detail.values())),
            "gate_detail": gate_detail,
        },
        "failure_cases": _failure_cases(per_regime),
        "route_tables": route_tables,
        "wall_clock_by_method": wall,
    }


def _summarise_ablation(ablation: dict[str, list[float]]) -> dict:
    """Report the adaptive-variant ablation honestly (negative results kept)."""
    out: dict[str, dict] = {}
    for base in ("pdhg", "admm"):
        plain_key, ada_key = f"{base}:plain", f"{base}:adaptive"
        if plain_key not in ablation or ada_key not in ablation:
            continue
        plain, ada = geometric_mean(ablation[plain_key]), geometric_mean(ablation[ada_key])
        delta = _log10(plain) - _log10(ada)  # >0 means adaptive helped
        out[base] = {
            "plain_geomean": plain,
            "adaptive_geomean": ada,
            "orders_gain": delta,
            "adopted": bool(delta >= 0.3),
            "conclusion": (
                "adopted" if delta >= 0.3 else "REJECTED — neutral or harmful on held-out problems"
            ),
        }
    return out


def _failure_cases(per_regime: dict[str, dict]) -> list[dict]:
    """Derive >= 3 notable failure cases straight from measured results."""
    cases: list[tuple[float, dict]] = []
    for regime, table in per_regime.items():
        for method, val in table.items():
            if method == FLAGSHIP:
                continue
            if val > 1e-4:  # genuinely poor, not just stiffer arithmetic
                cause = _EXPLAIN.get((method, regime), _DEFAULT_EXPLAIN.get(method, ""))
                cases.append(
                    (val, {"regime": regime, "method": method, "cert": val, "cause": cause})
                )
    cases.sort(key=lambda t: -t[0])
    return [c for _, c in cases[:6]]


class ConvexForgePipeline:
    """Thin object wrapper around :func:`run_benchmark`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self) -> dict:
        return run_benchmark(self.cfg)
