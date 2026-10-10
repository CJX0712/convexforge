# ConvexForge

[![CI](https://github.com/CJX0712/convexforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/convexforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Deterministic](https://img.shields.io/badge/determinism-bit--identical-brightgreen.svg)](benchmark.json)

**KKT-certified convex optimization suite.** Four solvers — quadratic penalty,
consensus ADMM, Chambolle–Pock primal–dual, Mehrotra predictor–corrector interior
point — plus **ConvFuse**, a regime router that is provably non-inferior to the best
component and beats every single one of them.

> Author: **晨星** · License: MIT · NumPy-only core (no scipy / cvxpy / torch required)

---

## Why this is different

Most solver benchmarks ask *"does it converge?"*. ConvexForge asks
**"can you prove it?"**.

Every problem instance is **constructed from a KKT-exact triple** `(x*, y*, s*)`,
so the true optimum `f*` and the true solution `x*` are known *exactly* — not
estimated by a reference solver, not self-reported. Every engine is then scored by
one **identical, scale-free certificate**:

```
given only (x, y):   s = max(0, Qx + c − A'y)
r_primal = ‖Ax − b‖_∞       / max(1, ‖b‖_∞)
r_dual   = ‖max(0, −r_d)‖_∞ / max(1, ‖c‖_∞ + ‖Q‖_∞·max(1,‖x‖_∞))
r_comp   = ‖x ⊙ s‖_∞        / max(1, ‖x‖_∞·‖s‖_∞)
r_nonneg = ‖max(0, −x)‖_∞   / max(1, ‖x‖_∞)
cert     = max of the four
```

No engine gets a bespoke stop rule or a bespoke score. The number in the table
below is a **ground-truth-anchored** number.

---

## Headline result

`python -m convexforge.cli demo` — 3 seeds × 4 regimes × 3 held-out problems,
CPU-only, ~20 s per pass. Full record: [`benchmark.json`](benchmark.json).

| method | geomean `cert` | log₁₀ mean ± std | mean iters |
|---|---:|---:|---:|
| `pen` (naive baseline) | 1.238e-01 | −0.907 ± 0.102 | 400 |
| `admm` | 5.510e-07 | −6.259 ± 4.282 | 260 |
| `pdhg` | 3.069e-07 | −6.513 ± 3.092 | 365 |
| `ipm`  | 4.316e-10 | −9.365 ± 3.216 | 160 |
| **`convfuse`** (flagship) | **7.063e-11** | **−10.151 ± 1.426** | 188 |

Pre-registered gates (thresholds fixed *before* the numbers were read):

| gate | value | threshold | verdict |
|---|---:|---:|:--:|
| G1 — orders vs best component (`ipm`) | **0.786** | ≥ 0.5 | ✅ PASS |
| G2 — orders vs naive `pen` | **9.244** | ≥ 3.0 | ✅ PASS |
| G3 — non-inferior to best component (≤ 1.02×) | **True** | True | ✅ PASS |
| determinism (same seed, two runs) | **True** (max Δ = 0.0) | True | ✅ PASS |

Metrics are compared in **log₁₀ space** and aggregated with a **geometric mean**.
Raw relative reduction is degenerate here: `pen → ipm` spans nine orders of
magnitude, so any percentage-based reduction is dominated by whichever instance
happens to sit at 1e-1.

---

## The four regimes

| regime | structure | what it stresses |
|---|---|---|
| `lp` | `Q = 0` | no curvature; pure LP duality |
| `qp_pd` | `Q = RᵀR/n + 0.1 I` | well-conditioned strictly convex QP |
| `qp_singular` | `Q = BᵀB/r`, rank `r ≪ n` | singular Hessian, flat directions |
| `degen` | `Q = RᵀR/n + 0.02 I`, `A` rank-deficient (20/30) | degenerate constraints |

Per-regime geomean `cert` (flagship in **bold**):

| regime | pen | admm | pdhg | ipm | convfuse |
|---|---:|---:|---:|---:|---:|
| `lp` | 1.26e-01 | 3.88e-08 | 7.51e-04 | 7.44e-12 | **7.44e-12** |
| `qp_pd` | 1.32e-01 | 3.70e-10 | 8.46e-10 | 2.82e-12 | **2.82e-12** |
| `qp_singular` | 1.15e-01 | 2.52e-09 | 7.64e-05 | 6.48e-09 | **6.48e-09** |
| `degen` | 1.23e-01 | **2.55e+00** ✗ | 1.83e-10 | 2.55e-07 | **1.83e-10** |

The router's job is visible in that last row: `degen` is where interior point
stumbles (2.5e-07) and Chambolle–Pock shines (1.8e-10) — a **3-order** spread.
ConvFuse picks `pdhg` there on all 3 seeds and `ipm` elsewhere, and therefore
matches the per-regime best everywhere.

---

## Install & run

```bash
git clone https://github.com/CJX0712/convexforge.git
cd convexforge
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt                  # pinned, offline-capable
pip install -e .

python -m convexforge.cli selftest   # 10 invariant gates, ~1 s
python -m convexforge.cli demo       # full benchmark -> benchmark.json (~20 s)
python -m convexforge.cli run        # one pass, no determinism re-run
make ci                              # lint + selftest + tests + demo
```

Minimal usage:

```python
import numpy as np
from convexforge.data.synthetic import make_problem
from convexforge.solvers.engines import solve_one
from convexforge.eval.certificate import certificate

prob = make_problem("qp_pd", 0, seed=7, n_vars=80, n_cons=30)
res  = solve_one("convfuse", prob, max_iters=400, tol=1e-10)
print(res.method, res.cert, res.iters)      # convfuse 2.8e-12 22
print(certificate(prob, res.x, res.y))
```

---

## Invariants (10 gates, all PASS)

`python -m convexforge.cli selftest` enforces these on every run; CI fails if any
breaks. They are the project's real test suite.

1. `finite` — no NaN/Inf anywhere in a result.
2. `psd_problem` — `Q` symmetric PSD (LP case `Q = 0` allowed).
3. `ground_truth_kkt` — constructed `(x*, y*, s*)` satisfies KKT to ~1e-16.
4. `no_solution_beats_optimum` — sensitivity bound
   `f(x) ≥ f* − ‖y*‖·‖r_p‖ − s*ᵀ·max(0,−x) − tol`. **Both** slack terms are
   required: an iterate can be equality-feasible yet sit slightly negative, and
   dropping the inequality term produces false alarms.
5. `weak_duality_lp` — `c'x ≥ b'y` at a *mutually* feasible pair (primal
   `Ax = b` **and** `x ≥ 0`, dual residual violation `= 0`).
6. `monotone` — penalty objective trace never increases.
7. `improvement` — flagship beats the naive baseline.
8. `deterministic` — same seed ⇒ bit-identical metrics.
9. `no_leak` — TRAIN and TEST index blocks are disjoint.
10. `certified` — `cert ≤ tol` achievable on at least one held-out problem.

See `docs/architecture.md` for the derivation of each.

---

## Ablation — negative results kept

| variant | plain geomean | adaptive geomean | orders gain | adopted |
|---|---:|---:|---:|:--:|
| `pdhg` + adaptive restart | 4.598e-07 | 6.269e-07 | **−0.135** | ❌ REJECTED |
| `admm` + adaptive ρ | 1.622e-06 | 1.975e-06 | **−0.086** | ❌ REJECTED |

Both were expected to help. Both measured *worse*. An early restart
implementation that also shrank the primal step degraded `degen` by three orders
(9.6e-09 → 2.0e-06); corrected to momentum-reset-only it became neutral, not
helpful. They are reported as REJECTED rather than quietly dropped.

---

## Known failure cases (root-caused, 6 of them)

| regime | method | cert | root cause |
|---|---|---:|---|
| `degen` | `admm` | 2.55e+00 | the consensus x-step solves `[[Q+ρI, A'],[A,0]]`, singular when `A` is rank-deficient; every iterate is corrupted |
| `qp_pd` | `pen` | 1.32e-01 | penalises `Ax = b` instead of enforcing it; never produces a dual certificate |
| `lp` | `pen` | 1.26e-01 | same — plateaus near 0.25 regardless of iteration count |
| `degen` | `pen` | 1.23e-01 | same |
| `qp_singular` | `pen` | 1.15e-01 | same |
| `lp` | `pdhg` | 7.51e-04 | `Q = 0` removes all curvature in `x`; Chambolle–Pock loses linear convergence and creeps sublinearly |

Iterates are additionally guarded by `iterate_sane()` (‖x‖ > 1e12 ⇒ stop), which
prevents float64 overflow from silently turning into `inf`/`nan`.

---

## Determinism

`pin_threads()` is called before the first NumPy import lands: five BLAS/OpenMP
env vars are forced to `"1"` so no thread-count-dependent reduction ordering can
leak in. Two consecutive `demo` passes produce `determinism_max_abs_diff = 0.0`.

---

## Repository layout

```
convexforge/
├── core/       types, config, errors, seed, linalg, interfaces
├── data/       synthetic.py — KKT-exact instance construction, TRAIN/TEST split
├── solvers/    penalized, admm, pdhg, ipm, fuse (router), engines (registry)
├── eval/       certificate, metrics, invariants
├── pipeline/   benchmark (leak-guarded), selftest
└── examples/   run_demo.py
docs/           architecture.md, model_card.md
tests/          58 unit tests (ruff-clean, pytest-clean)
```

Offline-first: the shipped code path imports **NumPy only**. scipy / scikit-learn /
torch / cvxpy appear nowhere in the package (asserted by
`tests/test_offline.py`). `requirements.txt` carries the optional dev tools.

---

## Citation

```bibtex
@software{convexforge2026,
  author    = {晨星},
  title     = {ConvexForge: KKT-certified convex optimization with a non-inferior regime router},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/CJX0712/convexforge},
  license   = {MIT},
  version   = {0.1.0}
}
```

Reference works: Boyd et al. 2011 (ADMM) · Chambolle & Pock 2011 (PDHG) ·
Mehrotra 1992 / Nocedal & Wright 2006 (interior point) · Aharon et al. (DictForge,
same series).

---

MIT © 2026 晨星
