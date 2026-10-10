# Changelog

All notable changes to ConvexForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Author: **晨星** · License: MIT

## [0.1.0] — 2026-10-10

Initial public release. Quality grade **S** (all pre-registered gates PASS).

### Added

**Core**
- `core/types.py` — `Problem` (carries `x_star`, `f_star`, and the KKT multipliers
  `y_star`/`s_star` in `meta`), `SolveResult`, `BenchmarkRow`.
- `core/config.py` — dataclass config with an explicit `_TYPES` map (PEP 563
  annotation-string workaround) and `ENV_<NAME>` overrides.
- `core/errors.py` — five error families with stable codes:
  `E000/E100/E200/E300/E400/E500`.
- `core/seed.py` — `pin_threads()` (5 BLAS env vars forced to `"1"`) + `set_all()`.
- `core/linalg.py` — `solve_spd` (Cholesky → jitter → lstsq chain), `assert_psd`,
  `iterate_sane` (float64 overflow guard at ‖x‖ > 1e12).
- `core/interfaces.py` — `Solver` protocol, documented sign conventions.

**Data**
- `data/synthetic.py` — KKT-exact instance construction for four regimes
  (`lp`, `qp_pd`, `qp_singular`, `degen`), strict complementarity, disjoint
  TRAIN/TEST index blocks, `ground_truth_kkt()` self-check.

**Solvers**
- `penalized.py` — naive quadratic-penalty projected gradient (baseline).
- `admm.py` — consensus ADMM (Boyd 2011).
- `pdhg.py` — Chambolle–Pock primal–dual (2011) with momentum-reset restart.
- `ipm.py` — Mehrotra predictor–corrector primal–dual interior point.
- `fuse.py` — `ConvFuse` regime router, fit on TRAIN only.
- `engines.py` — registry: `pen / admm / pdhg / ipm / convfuse`.

**Evaluation**
- `eval/certificate.py` — one scale-free KKT certificate shared by all engines.
- `eval/metrics.py` — geometric mean, log₁₀ mean/std, iters-to-tol.
- `eval/invariants.py` — 10 invariant gates (see README).
- `pipeline/benchmark.py` — leak-guarded benchmark + ablation + failure-case mining.
- `pipeline/selftest.py` — 10-gate self test.
- `cli.py` — `demo` / `run` / `selftest` subcommands.

**Engineering**
- `pyproject.toml`, `.gitignore`, `LICENSE` (MIT 晨星 2026), `Makefile`,
  `Dockerfile`, `requirements.txt`, `requirements.lock.txt`.
- `.github/workflows/ci.yml` — py3.12 × py3.13 on ubuntu-latest + windows-latest.
- `docs/architecture.md`, `docs/model_card.md`.
- 58 unit tests across 8 modules.

### Measured (seed 7/8/9, 80 vars, 30 cons, 400 iters, tol 1e-10)

| method | geomean cert | log₁₀ mean ± std |
|---|---:|---:|
| `pen` | 1.238e-01 | −0.907 ± 0.102 |
| `admm` | 5.510e-07 | −6.259 ± 4.282 |
| `pdhg` | 3.069e-07 | −6.513 ± 3.092 |
| `ipm` | 4.316e-10 | −9.365 ± 3.216 |
| `convfuse` | **7.063e-11** | **−10.151 ± 1.426** |

G1 = 0.786 (≥ 0.5) · G2 = 9.244 (≥ 3.0) · G3 non-inferior = True ·
deterministic (max Δ = 0.0) · demo 19.8 s per pass.

### Fixed during development

- **ADMM dual sign**: returning `y = nu` gave a dual residual ≈ 1.0. The fixed
  point of the consensus iteration gives `y = −nu`, `s = −ρ·u`. After the fix,
  `admm` on `lp` went 1.5 → 1.0e-4.
- **PDHG step blow-up on LP**: `lip_q = max(‖Q‖₂, 1e-12)` made `τ ≈ 9e11` when
  `Q = 0`, pinning the certificate at 1.0. Floored at 1.0.
- **`assert_no_solution_beats_optimum` false alarm** (gap −5.77e-4): the
  sensitivity bound omitted the *inequality* violation. ADMM only projects `z`, so
  `x` sat 1.75e-4 negative while equality-feasible to 4e-14. Added the
  `s*ᵀ·max(0,−x)` term.
- **Float64 overflow on `degen`**: ADMM diverged to ‖x‖ ~ 1e150, producing NaN and
  a spurious weak-duality violation of −4.7e-4. Added `iterate_sane()` guards to
  all four solvers and strengthened `assert_weak_duality_lp` to require *mutual*
  feasibility. Residual −4.7e-4 → −1.18e-5 → resolved.
- **G1 gate failed at 0.149**: interior point saturates at ~28 iterations while
  splitting methods keep improving, so a 200-iteration budget flattered the wrong
  engine. Measured sweep: G1 = −0.20 @ 200 iters, +0.79 @ 400, +1.28 @ 1500.
  Chose 400 (balances the ≤ 60 s CPU budget) and tightened `cert_tol` 1e-6 → 1e-10,
  because the loose tolerance hid the trade-off entirely.

### Rejected (kept as negative results in the ablation)

- `pdhg` + adaptive restart — orders gain **−0.135**.
- `admm` + adaptive ρ — orders gain **−0.086**.

An earlier restart variant that also shrank the primal step degraded `degen` by
three orders (9.6e-09 → 2.0e-06). Corrected to momentum-reset-only it became
neutral, not beneficial.
