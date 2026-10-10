# Model Card — ConvexForge `convfuse` v0.1.0

> Following Mitchell et al. (2019), adapted for an optimization engine rather than
> a learned model. Author **晨星** · MIT · 2026-10-10.

---

## Model details

| field | value |
|---|---|
| **Name** | `convfuse` (ConvFuse regime-routed convex QP solver) |
| **Version** | 0.1.0 |
| **Author** | 晨星 |
| **License** | MIT |
| **Type** | Non-learned composite solver: `{admm, pdhg, ipm}` + a TRAIN-fitted regime router |
| **Parameters learned** | **0 continuous.** The router's only state is a 4-entry lookup table `regime → engine name`, fitted by geometric-mean certificate on TRAIN problems. |
| **Training data** | 12 synthetic QP instances (3 per regime × 4 regimes), constructed — not observed |
| **Input** | `Q ⪰ 0 (n×n)`, `c (n)`, `A (m×n)`, `b (m)` |
| **Output** | `x (n)`, `y (m)`, plus a KKT certificate `cert ≥ 0` |
| **Dependencies** | NumPy ≥ 2.0 only (core path) |
| **Compute** | CPU, single-threaded BLAS, ~20 s for the full 36-problem × 5-engine benchmark |

---

## Intended use

**In scope**
- Research / teaching: a reference implementation where every number is
  ground-truth-anchored and every engine is scored identically.
- Benchmarking: comparing a new first-order or interior-point method against four
  known ones under one certificate.
- Small-to-medium dense convex QPs (n ≤ a few hundred) where a **verifiable**
  answer matters more than raw speed.
- Verifying that another solver's output actually satisfies KKT.

**Out of scope**
- Non-convex problems. Nothing checks convexity beyond `Q ⪰ 0`; a non-convex `Q`
  will silently produce a meaningless certificate.
- Large-scale / sparse problems. All four engines are dense; there is no sparse
  linear algebra, no preconditioning, no presolve. `n` in the thousands will be slow.
- General cone programs (SOCP, SDP). Only `Ax = b, x ≥ 0` is supported.
- Mixed-integer or combinatorial optimization.
- Production-critical numerical decisions without independent verification. This is
  research-grade software; it has not been validated against a certified solver.

---

## Training / fitting procedure

There is no gradient descent anywhere. The "fit" is:

```
for each regime r in {lp, qp_pd, qp_singular, degen}:
    for each candidate c in {admm, pdhg, ipm}:
        score[r][c] = geometric_mean({ certificate(c.solve(p)) for p in TRAIN[r] })
    route[r] = argmin_c score[r][c]
```

- **Data**: `data/synthetic.py`. Every instance is generated *backwards* from a
  KKT-exact triple `(x*, y*, s*)`; `b = Ax*` and `c = Aᵀy* + s* − Qx*` follow
  analytically. There is no real-world data, no scraping, no labels.
- **Split**: TRAIN indices `0..n_train-1`, TEST indices `n_train..n_train+n_test-1`.
  `assert_no_leak` enforces disjointness before scoring.
- **Budget**: 400 iterations, tolerance 1e-10 (see `docs/architecture.md` §8 for
  why these values).
- **Seeds**: 7, 8, 9.

### Data provenance & bias

**All data is synthetic.** This has a real consequence worth stating plainly:
performance on these instances does **not** transfer to real-world QPs (portfolio
optimization, control, optimal transport) without re-validation. The regimes were
chosen to stress specific numerical pathologies (singular Hessian, rank-deficient
constraints, zero curvature), which makes the suite *harder* than typical random
QPs along those axes, and *easier* along every axis we did not think to stress
(ill-conditioned but full-rank `A`, badly scaled `c`, huge dynamic range in `b`).

No personally identifying information, no human subjects, no licensed corpora.

---

## Evaluation

**Metric**: `cert = max(r_primal, r_dual, r_comp, r_nonneg)`, a scale-free KKT
residual (see `docs/architecture.md` §4). Lower is better. Ground truth `f*` and
`x*` are **exact by construction**, so no reference solver is involved.

**Aggregation**: geometric mean over 36 held-out problems (3 seeds × 4 regimes × 3
TEST problems), compared in log₁₀ space.

| method | geomean `cert` | log₁₀ mean ± std | mean iters |
|---|---:|---:|---:|
| `pen` (naive baseline) | 1.238e-01 | −0.907 ± 0.102 | 400 |
| `admm` | 5.510e-07 | −6.259 ± 4.282 | 260 |
| `pdhg` | 3.069e-07 | −6.513 ± 3.092 | 365 |
| `ipm` | 4.316e-10 | −9.365 ± 3.216 | 160 |
| **`convfuse`** | **7.063e-11** | **−10.151 ± 1.426** | 188 |

Pre-registered gates: **G1** 0.786 ≥ 0.5 ✅ · **G2** 9.244 ≥ 3.0 ✅ ·
**G3** non-inferior ✅ · determinism ✅ (max Δ = 0.0).

**Suboptimality** (exact, since `f*` is known): flagship mean `f(x) − f*`
= **+2.27e-08**. Naive `pen` reports **−24.6**, which is negative and therefore
*infeasible-looking* — it is what a method that never enforces `Ax = b` produces.

### Per-regime

| regime | pen | admm | pdhg | ipm | convfuse |
|---|---:|---:|---:|---:|---:|
| `lp` | 1.26e-01 | 3.88e-08 | 7.51e-04 | 7.44e-12 | **7.44e-12** |
| `qp_pd` | 1.32e-01 | 3.70e-10 | 8.46e-10 | 2.82e-12 | **2.82e-12** |
| `qp_singular` | 1.15e-01 | 2.52e-09 | 7.64e-05 | 6.48e-09 | **6.48e-09** |
| `degen` | 1.23e-01 | 2.55e+00 ✗ | 1.83e-10 | 2.55e-07 | **1.83e-10** |

Standard deviation in log₁₀ is large for the components (3.1–4.3) and small for
the flagship (1.43). That is the routing effect: the router cuts the tail as well as
the mean.

---

## Fairness / neutrality of the comparison

- **One certificate for all engines.** A solver cannot influence its score.
- **One iteration budget for all engines** (400). No early-stopping advantage.
- **Identical problem sets.** Every engine sees the same 36 TEST problems.
- **`pen` is a real baseline, not a strawman.** It is an honest implementation of
  the textbook quadratic-penalty method; it is weak because the method is weak.
- **The baseline is included in the reported table**, not hidden in an appendix.

---

## Failure modes

| regime | method | cert | root cause |
|---|---|---:|---|
| `degen` | `admm` | 2.55e+00 | consensus KKT matrix `[[Q+ρI, Aᵀ],[A,0]]` singular when `A` is rank-deficient |
| `lp` | `pdhg` | 7.51e-04 | `Q = 0` ⇒ no curvature ⇒ Chambolle–Pock loses linear convergence |
| all | `pen` | ~1.2e-01 | penalises rather than enforces `Ax = b`; never produces a dual certificate |
| `degen` | any, unguarded | NaN | float64 overflow — mitigated by `iterate_sane()` (stop, don't crash) |

**Mitigation**: `convfuse` routes around the first two by construction (that is its
entire reason for existing). It cannot help with the third, because `pen` is not a
routing candidate. Overflow is contained, not cured.

---

## Limitations

1. **Synthetic-only evaluation.** No real-world benchmark (Netlib LP, Maros–Mészáros
   QP) has been run. Transfer is unverified.
2. **Dense only.** No sparse path.
3. **No presolve / scaling.** Badly scaled real problems will behave worse than
   these instances suggest.
4. **Iteration-budget-sensitive.** At 200 iterations the flagship *loses* to plain
   `ipm` (G1 = −0.20). The claim holds at the documented budget of 400 and improves
   beyond it; it is not budget-independent.
5. **Router is a lookup table.** It generalises only across *regimes*, not across
   problem sizes. A new regime not in `{lp, qp_pd, qp_singular, degen}` falls back
   to `ipm`.
6. **Tolerance-dependent.** At `cert_tol = 1e-6` every engine passes and the
   flagship's advantage is invisible. The gate thresholds are only meaningful at the
   documented `1e-10`.

---

## Ethical considerations

Optimization engines are dual-use in the ordinary way (portfolio optimization,
resource allocation, logistics, and the military analogues of all three). This
release is a research/teaching artifact of modest scale — dense, single-threaded,
uncompetitive with mature commercial solvers on any real workload — so it does not
lower any practical capability barrier. No data, no human subjects, no deployment.

The honest risk here is **overclaiming**: a table of small numbers can read as
"better than mature solvers" when it actually means "better than three other
textbook implementations of mine, on my own synthetic instances, at my chosen
iteration budget". The limitations section exists to prevent that reading.

---

## Reproducibility

```bash
python -m convexforge.cli demo     # writes benchmark.json, runs twice to prove determinism
```

`benchmark.json` in the repository root is the unmodified artifact of the run
reported here: 180 rows, `wall_clock_sec = 19.811`, `determinism_max_abs_diff = 0.0`.

---

MIT © 2026 晨星
