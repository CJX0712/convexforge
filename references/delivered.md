# ConvexForge — Delivery Report

> Author: **晨星** · License: MIT · Repo: https://github.com/CJX0712/convexforge
> Quality grade: **S** (all pre-registered gates PASS)

Delivered 2026-10-10. Part of the world-class AI series (DictForge, MatrixForge,
FlowForge, …). This file is the authoritative delivery record.

---

## What was built

A KKT-certified convex optimization suite. Canonical problem:

```
min ½ xᵀQx + cᵀx    s.t.  A x = b,  x ≥ 0      (Q symmetric PSD; Q = 0 ⇒ LP)
```

Four solvers + one flagship router:

| engine | method | reference |
|---|---|---|
| `pen` | quadratic-penalty projected gradient | textbook baseline |
| `admm` | consensus ADMM | Boyd et al. 2011 |
| `pdhg` | Chambolle–Pock primal–dual | Chambolle & Pock 2011 |
| `ipm` | Mehrotra predictor–corrector interior point | Mehrotra 1992 / Nocedal & Wright 2006 |
| `convfuse` | **flagship**: regime router over {admm, pdhg, ipm}, fit on TRAIN | this work |

---

## Why it is trustworthy (the fairness keystone)

Every instance is **constructed backwards from a KKT-exact triple** `(x*, y*, s*)`
(`data/synthetic.py`). `b = Ax*`, `c = Aᵀy* + s* − Qx*` ⇒ the true optimum `f*` and
solution `x*` are **exact by construction**, not estimated by a reference solver.
`ground_truth_kkt()` re-verifies to ~1e-16 on every instance.

Every engine is then scored by **one identical, scale-free KKT certificate**
(`eval/certificate.py`):

```
given only (x, y):   s = max(0, Qx + c − Aᵀy)
cert = max( ‖Ax−b‖_∞/‖b‖_∞, ‖max(0,−(Qx+c−Aᵀy))‖_∞/scale,
            ‖x⊙s‖_∞/(‖x‖_∞‖s‖_∞), ‖max(0,−x)‖_∞/‖x‖_∞ )
```

No engine gets a bespoke stop rule or a bespoke score.

---

## Benchmark (seed 7/8/9 · n=80 · m=30 · 400 iters · tol 1e-10)

Full artifact: `benchmark.json` (180 rows).

| method | geomean cert | log₁₀ mean ± std | mean iters |
|---|---:|---:|---:|
| `pen` (naive) | 1.238e-01 | −0.907 ± 0.102 | 400 |
| `admm` | 5.510e-07 | −6.259 ± 4.282 | 260 |
| `pdhg` | 3.069e-07 | −6.513 ± 3.092 | 365 |
| `ipm` | 4.316e-10 | −9.365 ± 3.216 | 160 |
| **`convfuse`** | **7.063e-11** | **−10.151 ± 1.426** | 188 |

### Pre-registered gates (thresholds fixed before reading the numbers)

| gate | value | threshold | verdict |
|---|---:|---:|:--:|
| G1 — orders vs best component (`ipm`) | 0.786 | ≥ 0.5 | ✅ PASS |
| G2 — orders vs naive `pen` | 9.244 | ≥ 3.0 | ✅ PASS |
| G3 — non-inferior to best component (≤ 1.02×) | True | True | ✅ PASS |
| determinism (two runs, same seed) | True | True | ✅ PASS |

Metrics aggregated by **geometric mean** in **log₁₀** space — raw relative reduction
is degenerate across nine orders of magnitude.

### Per-regime (geomean cert)

| regime | pen | admm | pdhg | ipm | convfuse |
|---|---:|---:|---:|---:|---:|
| `lp` | 1.26e-01 | 3.88e-08 | 7.51e-04 | 7.44e-12 | **7.44e-12** |
| `qp_pd` | 1.32e-01 | 3.70e-10 | 8.46e-10 | 2.82e-12 | **2.82e-12** |
| `qp_singular` | 1.15e-01 | 2.52e-09 | 7.64e-05 | 6.48e-09 | **6.48e-09** |
| `degen` | 1.23e-01 | 2.55e+00 ✗ | 1.83e-10 | 2.55e-07 | **1.83e-10** |

The router picks `pdhg` on `degen` (all 3 seeds) and `ipm` elsewhere. The entire
0.786-order gain comes from that one regime: interior point's normal matrix
`A M⁻¹ Aᵀ` is singular when `A` is rank-deficient, so Newton stalls while the
first-order Chambolle–Pock scheme is unaffected.

---

## Engineering checklist (DoD)

- ✅ One-click reproducible (`make ci` = lint + selftest + tests + demo)
- ✅ 58 unit tests, all green
- ✅ Dependency lock (`requirements.lock.txt`: numpy 2.5.3 / pytest 9.1.1 / ruff 0.16.10)
- ✅ Offline fallback: core imports **NumPy only** (`tests/test_offline.py` asserts
  no scipy/sklearn/torch/cvxpy in the package path)
- ✅ Determinism (BLAS threads pinned to 1; max|Δ| = 0.0 across two runs)
- ✅ ≥ 3 seeds (actually 3) mean ± std beating a strong baseline by pre-registered
  thresholds (G1 0.786, G2 9.244)
- ✅ ≥ 1 ablation with negative results retained (2 adaptive variants REJECTED)
- ✅ ≥ 3 failure cases with root-cause attribution (6 recorded)
- ✅ No data leak (disjoint TRAIN/TEST index blocks; `assert_no_leak`)
- ✅ Docs (README 5 badges + `docs/architecture.md` + `docs/model_card.md`)
- ✅ Demo ≤ 60 s CPU (20.4 s per pass)
- ✅ CI green matrix (py3.12/3.13 × ubuntu/windows) — `.github/workflows/ci.yml`
- ✅ Release + tag `v0.1.0`
- ✅ MIT license, author 晨星
- ✅ Secret scan clean (0 hits)

---

## Reproduce

```bash
python -m convexforge.cli selftest   # 10 invariant gates
python -m convexforge.cli demo       # -> benchmark.json, runs twice for determinism
pytest -q                            # 58 tests
ruff check . && ruff format --check .
```

---

## What this is NOT

- Not a general cone solver (only `Ax=b, x≥0`).
- Not large-scale / sparse (all engines dense).
- Not validated on real-world QP libraries (Netlib, Maros–Mészáros) — synthetic-only.
- No learned parameters; the router is a 4-entry lookup table.

See `docs/model_card.md` for the full fairness/limitation discussion.

---

MIT © 2026 晨星
