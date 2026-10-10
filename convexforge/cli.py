"""ConvexForge command-line entry point.

BLAS/LAPACK threads are pinned at the very top, *before* numpy is imported, so
summation order — and therefore the deterministic core metrics — is reproducible.
"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from .core.config import Config  # noqa: E402
from .pipeline.benchmark import run_benchmark  # noqa: E402
from .pipeline.selftest import run_selftest  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_out() -> str:
    return os.path.join(_REPO_ROOT, "benchmark.json")


def _core_scalars(result: dict) -> dict:
    out: dict[str, float] = {}
    for m, d in result["per_method"].items():
        out[f"per_method.{m}.log10_mean"] = d["log10_mean"]
    h = result["headline"]
    out["orders_vs_component"] = h["orders_vs_best_component"]
    out["orders_vs_naive"] = h["orders_vs_naive"]
    out["flagship_geomean"] = h["flagship_convfuse"]["geomean"]
    return out


def _print_summary(result: dict) -> None:
    h = result["headline"]
    print("=" * 74)
    print("ConvexForge benchmark summary  (geomean KKT certificate, lower is better)")
    print("=" * 74)
    print(f"  regimes        : {result['config']['regimes']}")
    print(f"  seeds          : {result['config']['n_seeds']}")
    print(f"  problem size   : n={result['config']['n_vars']} m={result['config']['n_cons']}")
    print(f"  iteration budget: {result['config']['max_iters']}")
    print("-" * 74)
    print(f"  {'method':12s} {'geomean cert':>14s} {'log10 mean':>11s} {'std':>7s}  iters")
    for m, d in result["per_method"].items():
        print(
            f"  {m:12s} {d['geomean']:14.3e} {d['log10_mean']:11.3f} "
            f"{d['log10_std']:7.3f}  {d['mean_iters']:.0f}"
        )
    print("-" * 74)
    print(f"  best component      : {h['best_component']['method']}")
    print(f"  orders vs component : {h['orders_vs_best_component']:.3f}  (gate >= 0.5)")
    print(f"  orders vs naive pen : {h['orders_vs_naive']:.3f}  (gate >= 3.0)")
    print(f"  non-inferior        : {h['non_inferior']}")
    print(f"  GATE PASS           : {h['gate_pass']}")
    print(f"  deterministic       : {result.get('deterministic')}")
    print(f"  wall-clock          : {result.get('wall_clock_sec')}s")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="convexforge", description="ConvexForge CLI")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="run full benchmark -> benchmark.json")
    p_demo.add_argument("--seeds", type=int, default=None)
    p_demo.add_argument("--out", type=str, default=None)

    p_run = sub.add_parser("run", help="alias for demo")
    p_run.add_argument("--seeds", type=int, default=None)
    p_run.add_argument("--out", type=str, default=None)

    sub.add_parser("selftest", help="run the 10 invariant gates")

    args = parser.parse_args(argv)
    cmd = args.cmd or "demo"

    if cmd == "selftest":
        res = run_selftest()
        for name, ok, detail in res["checks"]:
            line = f"[{'PASS' if ok else 'FAIL'}] {name}"
            if detail and not ok:
                line += f" -- {detail}"
            print(line)
        print("ALL_PASS" if res["all_pass"] else "SOME_FAILED")
        return 0 if res["all_pass"] else 1

    cfg = Config.from_env()
    if getattr(args, "seeds", None):
        cfg.n_seeds = args.seeds
    if getattr(args, "iters", None):
        cfg.max_iters = args.iters
    cfg.validate()

    t0 = time.time()
    result = run_benchmark(cfg)
    result["wall_clock_sec"] = round(time.time() - t0, 3)

    # Determinism gate: same seed twice => identical core scalars.
    second = run_benchmark(cfg)
    sa, sb = _core_scalars(result), _core_scalars(second)
    worst = max(abs(sa[k] - sb[k]) for k in sa)
    result["deterministic"] = bool(worst <= 1e-9)
    result["determinism_max_abs_diff"] = float(worst)

    out = args.out or _default_out()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
