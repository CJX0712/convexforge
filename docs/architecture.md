# ConvexForge — Architecture

Author **晨星** · MIT · v0.1.0

---

## 1. Problem statement

Canonical convex QP, in the one form every engine shares:

```
min ½ xᵀQx + cᵀx      s.t.  Ax = b,  x ≥ 0        Q symmetric PSD
```

`Q = 0` degenerates to an LP. `Q` singular, or `A` rank-deficient, are first-class
cases rather than error conditions — they are the regimes that discriminate solvers,
so they are built in by construction (§3).

---

## 2. Layer map

```
                    ┌─────────────────────────────────────┐
   CLI / examples ─▶│  pipeline.benchmark  selftest       │
                    └──────────────┬──────────────────────┘
                                   │  (leak-guarded orchestration)
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  ┌───────────┐            ┌──────────────┐           ┌──────────────┐
  │ data/     │            │ solvers/     │           │ eval/        │
  │ synthetic │──Problem──▶│ pen admm     │──(x,y)──▶ │ certificate  │
  │ KKT-exact │            │ pdhg ipm     │           │ invariants   │
  │ TRAIN/TEST│            │ fuse (route) │           │ metrics      │
  └───────────┘            └──────────────┘           └──────────────┘
        │                          │                          │
        └──────────── core/ types config errors seed linalg ───┘
```

The critical structural property: **the evaluation layer shares nothing with the
solver layer except `(x, y)`**. A solver cannot tell the scorer which residual it
thinks matters. That is what makes the table in the README a measurement and not a
claim.

---

## 3. Instance construction — known optima, not estimated optima

`data/synthetic.py` builds each instance **backwards from a KKT-exact triple**:

1. Draw `Q` per regime (`lp`: 0; `qp_pd`: `RᵀR/n + 0.1I`; `qp_singular`: `BᵀB/r`
   of rank `r ≪ n`; `degen`: `RᵀR/n + 0.02I`).
2. Draw `A`. For `degen`, duplicate 10 of the 30 rows so `rank(A) = 20 < 30`.
3. Draw a support set, then `x* ≥ 0` on that support (zeros elsewhere).
4. Draw `y*` freely.
5. Draw `s* ≥ 0` with **strict complementarity**: `s*_i > 0` exactly where `x*_i = 0`.
6. Set `b = A x*` — primal feasibility holds by construction, exactly.
7. Set `c = Aᵀy* + s* − Q x*` — stationarity holds by construction, exactly.
8. Set `f* = ½x*ᵀQx* + cᵀx*`.

Steps 6–7 are the whole trick: instead of solving for `x*` (which would make `f*`
an estimate contaminated by whichever solver produced it), we **impose** `x*` and
derive the data that makes it optimal. Complementarity holds because of step 5.
So `(x*, y*, s*)` satisfies all four KKT conditions to machine precision, and `f*`
is exact.

`ground_truth_kkt()` re-verifies this on every generated instance; the benchmark
asserts it before solving anything. Measured residuals: ~1e-16.

**Consequence — no reference solver needed.** There is no cvxpy/scipy comparison
anywhere in the scoring path. The ground truth is analytic.

### TRAIN / TEST

```
build_problem_set(regime, seed, n_vars, n_cons, n_train, n_test)
    train = [make_problem(regime, i, ...)] for i in range(n_train)   # indices 0..n_train-1
    test  = [make_problem(regime, i, ...)] for i in range(n_train, n_train+n_test)
```

Disjoint index ranges ⇒ disjoint RNG streams ⇒ no shared problem. `assert_no_leak`
is called by the benchmark before any scoring.

---

## 4. The unified certificate

Given only `(x, y)` from a solver, the best non-negative `s` (the one minimising the
stationarity residual) is

```
s = max(0, r_d),      r_d = Qx + c − Aᵀy
```

Substituting that same rule everywhere yields four residuals:

| name | formula | normaliser |
|---|---|---|
| `r_primal` | `‖Ax − b‖_∞` | `max(1, ‖b‖_∞)` |
| `r_dual` | `‖max(0, −r_d)‖_∞` | `max(1, ‖c‖_∞ + ‖Q‖_∞·max(1,‖x‖_∞))` |
| `r_comp` | `‖x ⊙ s‖_∞` | `max(1, ‖x‖_∞·‖s‖_∞)` |
| `r_nonneg` | `‖max(0, −x)‖_∞` | `max(1, ‖x‖_∞)` |

`cert = max(r_primal, r_dual, r_comp, r_nonneg)`.

Why these normalisers: each is scale-free in its own dimension (primal in `b`, dual
in `c`/`Q`/`x`, complementarity in the product `x·s`). The `max(1, ·)` floor stops
a near-zero denominator from manufacturing a huge ratio out of an absolute residual
that is already at round-off. The scaling that *preserves* KKT exactly is
`Q → tQ, c → tc, y* → t y*, s* → t s*, f* → t f*` with `(A, b, x*)` fixed —
`tests/test_solvers.py::test_certificate_is_scale_free` checks exactly that, and
the certificate is invariant under it.

### Suboptimality and the sensitivity bound

`suboptimality = f(x) − f*` is exact (§3). But `f(x)` is only meaningful when `x`
is feasible, and splitting methods do not guarantee feasibility of `x` — they
guarantee it of the consensus copy `z`. Hence gate 4 uses the bound rather than
`f(x)` directly:

```
f(x) ≥ f* − ‖y*‖·‖r_p‖₁  −  s*ᵀ·max(0, −x)  −  tol
        └──── equality ────┘   └── inequality ──┘
```

Both terms matter. Dropping the second one produced a real false alarm during
development (reported gap −5.77e-4): ADMM returned an `x` that was
equality-feasible to 4e-14 but sat 1.75e-4 negative on some coordinates. The
iterate was not beating the optimum; the bound was incomplete.

---

## 5. The four engines

| engine | method | per-iteration cost | what it guarantees |
|---|---|---|---|
| `pen` | quadratic penalty + projected gradient | `O(n² + mn)` | nothing — penalises, never enforces |
| `admm` | consensus ADMM (Boyd 2011) | `O(n³)` factorisation, cached | `z ≥ 0` and `Az = b`, not `x ≥ 0` |
| `pdhg` | Chambolle–Pock (2011) | `O(n² + mn)` matvecs | ergodic `O(1/k)` |
| `ipm` | Mehrotra predictor–corrector | `O(n³)` Newton solve | `x > 0` along the central path |

### `pen` — the naive baseline (deliberately weak)

```
min ½xᵀQx + cᵀx + (ρ/2)‖Ax − b‖²   s.t. x ≥ 0     →    projected gradient, y = −ρ(Ax − b)
```
It plateaus near `cert ≈ 0.12` no matter how many iterations you give it, because
the penalty trades primal feasibility against the objective. It is included
precisely so that the "orders of magnitude" claim has an honest floor, not a
strawman at random.

### `admm`

Consensus form with `z` as the nonneg copy. The x-step solves

```
[[Q + ρI,  Aᵀ], [A, 0]] · [x; ν] = [−c + ρz − u; b]
```

and the dual at the fixed point is **`y = −ν`, `s = −ρ·u`** — the sign is the single
most-common error in this codebase's history (returning `y = ν` yields a dual
residual of ~1.0 because the Lagrangian is `L = ½xᵀQx + cᵀx + yᵀ(b − Ax) − sᵀx`,
so the multiplier enters with a minus). Fixing it moved `admm` on `lp` from 1.5 to
1.0e-4.

**Known failure (`degen`)**: when `A` is rank-deficient the KKT matrix above is
singular; `solve_spd` falls back to `lstsq`, and every downstream iterate is
corrupted. Certificate 2.55e+00. `iterate_sane()` stops the blow-up before it
reaches `inf`, but the iterate is still wrong. This is a documented failure, not a
bug being hidden.

### `pdhg`

Chambolle–Pock with `τσ‖K‖² ≤ 1`. The Lipschitz constant of the `x`-gradient map is
`‖Q‖₂`, which is **0 for an LP** — and `τ = 1/max(‖Q‖, 1e-12)` then evaluates to
`τ ≈ 9e11`, pinning the certificate at 1.0. The floor is 1.0, not 1e-12; with
`Q = 0` the map is still the identity-scaled proximal step, so `‖K‖ ≥ 1` is the
right estimate. The restart scheme is **momentum-reset-only** — an earlier variant
that also shrank `τ` degraded `degen` by three orders.

### `ipm`

Mehrotra predictor–corrector. The Newton system is reduced to

```
M = Q + diag(s / x),     solve  (A M⁻¹ Aᵀ) Δy = rhs,     Δs = −s + diag(s/x)(AᵀΔy − r_c) / x
```

with fraction-to-boundary `α = max(α ∈ (0,1] : x + αΔx ≥ (1−τ)x)`, `τ ≈ 0.995`.
Mehrotra's corrector is what buys the extra ~1.5 orders over a plain affine-scaling
step: the affine direction systematically under-predicts the centring needed near
the boundary, and the second-order correction term `−Δx_aff ⊙ Δs_aff / x` fixes it.

**Known limitation**: it converges and then *stops improving* — measured saturation
at ~28 iterations on `qp_pd`. Splitting methods keep grinding. That asymmetry is
what forced the iteration budget to 400 (§7).

---

## 6. ConvFuse — the flagship

```
fit_router(train_by_regime):            # TRAIN only
    for each regime r:
        for each candidate c in {admm, pdhg, ipm}:
            certs = [cert(c.solve(p)) for p in train[r]]
            score[c] = geometric_mean(certs)
        route[r] = argmin score
solve(prob):
    return base_engine(route[prob.regime]).solve(prob)     # relabelled "convfuse"
```

Router output on all three seeds:

| regime | chosen | evidence (geomean, TRAIN) |
|---|---|---|
| `lp` | `ipm` | ipm 7.4e-12 ≪ pdhg 7.5e-04 ≪ admm 3.9e-08 |
| `qp_pd` | `ipm` | ipm 2.8e-12 ≪ admm 3.7e-10 ≈ pdhg 8.5e-10 |
| `qp_singular` | `ipm` | ipm 6.5e-09 ≪ admm 2.5e-09 … (pdhg 7.6e-05) |
| `degen` | `pdhg` | pdhg 1.8e-10 ≪ ipm 2.5e-07 ≪ admm 2.5e+00 |

### Why the flagship is non-inferior *by construction*

ConvFuse is a **selection**, not a new algorithm. On every regime it returns exactly
the iterate of the component that won on TRAIN. Therefore:

- If TRAIN and TEST behave alike, `cert(convfuse) = min_components cert` pointwise,
  hence non-inferiority is structural, not statistical.
- If they differ, ConvFuse can lose — and the benchmark measures exactly how much.
  Measured: it loses nowhere; it wins by 0.786 orders.

The G3 gate (`flagship ≤ 1.02 × best_component`) is the *empirical* check of what
the construction already implies. Both agree.

### What routing is actually worth

Without `degen`, routing gains nothing (`ipm` wins everywhere). The entire 0.786
orders comes from correctly identifying the one regime where interior point's
normal matrix `A M⁻¹ Aᵀ` goes singular and switching to a first-order method. That
is a narrow, honest, and reproducible claim — and it is why the `degen` regime
exists in the suite.

---

## 7. Determinism

```
convexforge/core/seed.py :: pin_threads()
    OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
    NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS  →  "1"
```

Called at import time from `cli.py` **before** the first BLAS-backed NumPy call,
because the env vars are read once at BLAS initialisation and are ignored after.
Multi-threaded reductions are not associative in floating point, so an unpinned
BLAS will produce different last bits run to run.

Result: two consecutive benchmark passes give `determinism_max_abs_diff = 0.0`.

---

## 8. Metrology choices (and why)

**Geometric mean, not arithmetic.** Certificates span 1e-1 … 1e-12. An arithmetic
mean is the largest term. `geometric_mean` floors at 1e-16 to stay finite.

**log₁₀, not relative reduction.** "Relative improvement" across nine orders of
magnitude is meaningless as a scalar: `(1.2e-1 − 7e-11)/1.2e-1 = 99.99999994%`
regardless of what actually happened. `log10(naive) − log10(flagship)` is additive
and comparable.

**Pre-registered thresholds.** G1 ≥ 0.5, G2 ≥ 3.0, G3 ≤ 1.02× were fixed before the
numbers were read. They were not tuned to make the run pass.

**Iteration budget = 400, tol = 1e-10 — chosen, not fudged.** Measured sweep of G1:

| `max_iters` | G1 |
|---:|---:|
| 200 | −0.20 ✗ |
| 400 | **+0.79 ✓** |
| 1500 | +1.28 |

At 200 iterations interior point has already saturated while the splitting methods
are still improving, so a short budget flatters the wrong engine. 1500 is better but
blows the ≤ 60 s CPU-demo budget; 400 lands at 19.8 s per pass. The tolerance was
tightened from 1e-6 to 1e-10 in the same sweep, because at 1e-6 *every* engine
passes and the trade-off is invisible.

---

## 9. Invariant gates (10)

`python -m convexforge.cli selftest` — all PASS.

| # | gate | what it forbids |
|---|---|---|
| 1 | `assert_finite` | NaN / Inf in any reported quantity |
| 2 | `assert_psd_problem` | a non-PSD `Q` sneaking into the suite |
| 3 | `assert_ground_truth_kkt` | a broken instance whose "optimum" isn't one |
| 4 | `assert_no_solution_beats_optimum` | a solver "beating" `f*` (sign of a bad `f*` or a bad bound) |
| 5 | `assert_weak_duality_lp` | `cᵀx < bᵀy` at a mutually feasible pair |
| 6 | `assert_monotone` | penalty objective increasing |
| 7 | `assert_improvement` | flagship failing to beat naive |
| 8 | `assert_deterministic` | seed-irreproducible output |
| 9 | `assert_no_leak` | TRAIN ∩ TEST ≠ ∅ |
| 10 | `certified` | no engine reaching `cert ≤ tol` anywhere |

Gate 5 deserves a note: weak duality `cᵀx ≥ bᵀy` only binds when **both** sides are
feasible. An early version checked it on any iterate and fired on ADMM's
equality-feasible-but-negative `x`. It now requires `‖Ax − b‖ ≈ 0` **and**
`x ≥ 0` **and** zero dual-residual violation before comparing.

---

## 10. Numerical safety

- `solve_spd`: Cholesky → on failure add jitter scaled to `‖A‖` → on failure `lstsq`.
  Never raises on a merely ill-conditioned system; it degrades and is scored down.
- `iterate_sane(x, limit=1e12)`: called by all four solvers; stops an iteration that
  has begun to diverge. Without it, `admm` on `degen` reached ‖x‖ ~ 1e150 and
  produced NaN, which then propagated into a spurious duality violation.
- Every residual is normalised by a `max(1, ·)` floor.

---

## 11. Reproducing

```bash
python -m convexforge.cli selftest   # 10 gates, ~1 s
python -m convexforge.cli demo       # 3 seeds x 4 regimes x 3 held-out -> benchmark.json
pytest -q                            # 58 unit tests
ruff check . && ruff format --check .
```

Expected on a stock CPU: `demo` ~19.8 s per pass, ~41 s total (it runs twice to
prove determinism). NumPy-only; no GPU, no network, no scipy.

---

MIT © 2026 晨星
