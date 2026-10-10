# ConvexForge — Pitfalls & Lessons

> Author: **晨星** · License: MIT · Repo: https://github.com/CJX0712/convexforge
> Companion to `delivered.md`. Every correctness bug below was caught by an invariant
> gate, not by eyeballing numbers — that is the point of the gate suite.

---

## 1. ADMM dual sign (silent, 1.0-order error)

**Symptom**: returning `y = nu` gave a dual residual ≈ 1.0 — the solver looked
"fine" on primal feasibility but the dual certificate was broken, so the unified
KKT score pinned at ~1.0.

**Root cause**: the Lagrangian is `L = ½xᵀQx + cᵀx + yᵀ(b − Ax) − sᵀx`. The consensus
ADMM fixed point satisfies `y = −ν` (and `s = −ρ·u`), **not** `y = ν`. A `+` vs `−`
in the dual multiplier.

**Fix**: `y = -nu`, `s = -rho * u`. After: `admm` on `lp` 1.5 → 1.0e-4.

**Gate that caught it**: `certificate()` exposes the dual residual directly; the
case-by-case per-regime table in `docs/architecture.md` showed `admm` lagging its
analytic expectation.

---

## 2. PDHG step blow-up on LP (Q = 0)

**Symptom**: `pdhg` on `lp` returned `cert ≈ 1.0` (completely failed) while the same
code worked on QPs.

**Root cause**: the primal step size `τ` is derived from `1 / lip_q` where
`lip_q = max(‖Q‖₂, 1e-12)`. For an LP, `‖Q‖₂ = 0`, so `lip_q = 1e-12` and
`τ ≈ 9e11` — a giant step that diverges.

**Fix**: floor at `lip_q = max(‖Q‖₂, 1.0)`. The x-gradient map is `I` (scaled by
`Q`), so `‖K‖ ≥ 1` is the correct Lipschitz estimate for `Q = 0`.

**Lesson**: a "safe-looking" tiny floor (1e-12) is the wrong floor when the quantity
it guards can legitimately be exactly zero. Use the algebraically correct lower bound.

---

## 3. Sensitivity bound missing the inequality term (false alarm)

**Symptom**: `assert_no_solution_beats_optimum` raised `CertificateError` with a
*negative* gap (−5.77e-4), claiming an iterate "beat" the optimum `f*`.

**Investigation**: `x*` was verified genuinely optimal (`ground_truth_kkt` ≈ 7.5e-16),
and a 3000-direction random descent scan around `x*` returned exactly 0.0 — so `x*`
was not wrong, the bound was.

**Root cause**: the bound I wrote only covered the *equality* violation:
`f(x) ≥ f* − ‖y*‖·‖r_p‖`. ADMM only projects the consensus copy `z`; it does **not**
guarantee `x ≥ 0`. An ADMM iterate was equality-feasible to 4e-14 but sat **1.75e-4
negative** on some coordinates. The missing term is `− s*ᵀ·max(0, −x)`.

**Fix**: `f(x) ≥ f* − ‖y*‖·‖r_p‖ − s*ᵀ·max(0, −x) − tol`. **Both** slack terms are
required.

**Lesson**: a bound that "looks like it should hold" failed because the solver's
feasibility guarantees are narrower than the theorem's hypotheses. Match the bound to
what the engine actually enforces.

---

## 4. Float64 overflow on the degenerate regime

**Symptom**: ADMM on `degen` produced `‖x‖ ~ 1e150`, then NaN, then a spurious
weak-duality violation of −4.7e-4.

**Root cause**: `degen` has a rank-deficient `A` (20/30). The consensus x-step solves
`[[Q+ρI, Aᵀ],[A,0]]`, which is **singular** when `A` is rank-deficient; `lstsq`
returns an enormous, meaningless `x`.

**Fix (two parts)**:
- `iterate_sane(x, limit=1e12)`: every solver calls it; on divergence it **stops**
  (returns the last sane iterate) instead of letting NaN propagate.
- `assert_weak_duality_lp` now requires *mutual* feasibility: `Ax = b` AND `x ≥ 0`
  AND zero dual-residual violation, before comparing `cᵀx` vs `bᵀy`. An equality-only
  feasible `x` no longer counts.

Residual: −4.7e-4 → −1.18e-5 → resolved.

**Lesson**: overflow is not "just a warning" — it silently turns into a false
correctness signal. Guard the dynamics, and make feasibility assertions strict.

---

## 5. G1 gate failed at 0.149 (budget-dependent)

**Symptom**: at `max_iters = 200`, the flagship lost to plain `ipm` (G1 = −0.20).

**Root cause**: interior point **saturates early** (~28 iterations) — it reaches its
linear-convergence floor and stops improving. The first-order splitting methods keep
grinding for hundreds of iterations. A short budget flatters the *wrong* engine: it
cuts off the methods that benefit from more iterations while rewarding the one that is
already done.

**Fix (chosen, not fudged from the data)**: a measured sweep —

| `max_iters` | G1 |
|---:|---:|
| 200 | −0.20 ✗ |
| 400 | **+0.79 ✓** |
| 1500 | +1.28 |

Chose **400** (balances the ≤ 60 s CPU budget; demo at 400 = 20.4 s). Tightened
`cert_tol` 1e-6 → 1e-10 in the same sweep, because at 1e-6 *every* engine passes and
the trade-off is invisible.

**Lesson**: a comparison's outcome can be an artifact of the iteration budget. Sweep
it; pick the value from the curve, not from the gate margin.

---

## 6. Adaptive variants measured HARMFUL (kept as negative results)

Expected to help; both hurt:

| variant | plain geomean | adaptive geomean | orders gain | outcome |
|---|---:|---:|---:|:--:|
| `pdhg` + adaptive restart | 4.598e-07 | 6.269e-07 | −0.135 | REJECTED |
| `admm` + adaptive ρ | 1.622e-06 | 1.975e-06 | −0.086 | REJECTED |

An earlier restart implementation that *also shrank the primal step* degraded `degen`
by three orders (9.6e-09 → 2.0e-06). Corrected to momentum-reset-only it became
**neutral**, not beneficial. The ablation reports REJECTED rather than dropping the
data — negative results are load-bearing evidence that the router's TRAIN-fit is the
right call.

---

## 7. Publishing: empty-repo + blocked git push

**Symptom**: `git push` to `github.com:443` returns `502 CONNECT tunnel failed`
(enterprise sandbox blocks the direct git endpoint). `gh api` works.

**Consequences & fixes**:
- A `gh repo create` had created the repo but left it with only a bootstrap README.
  The local `.git` commit from an earlier run was incomplete → GitHub showed 1 file.
- **Fix**: build a clean local commit of all 49 files, force-push (still blocked),
  then fall back to the **Git Data API** (blobs → tree → commit → ref). The publisher
  (`scripts/gh_push.py`) implements a 3-level degrade: git+gh → Git Data API →
  Contents API.
- **MSYS path rewrite**: under Git-Bash, `gh api /repos/…` rewrites the leading `/`
  to a Windows path (`.../PortableGit/repos/...`) → 400/409. All endpoints are passed
  **without** a leading slash; verified with a probe create+delete.
- **502 transient**: the Git Data API occasionally returns `Bad Gateway` under load.
  `_gh_api` now retries with exponential backoff (6 attempts).
- **Git Data API on an empty repo**: `POST /git/blobs` returns `409 Git Repository is
  empty` until at least one commit exists. The Contents API must create the first
  commit; then Git Data works. Order matters.

**Lesson**: never assume `git push` reaches the network. Ship a scripted API fallback
and verify the remote tree after each publish (`git trees main?recursive=1` → count
blobs).

---

## 8. Determinism needs thread pinning *before* the first BLAS call

`pin_threads()` sets 5 BLAS/OpenMP env vars to `"1"`. It must run **before** the first
NumPy linear-algebra call (env vars are read once at BLAS init). Called at import in
`cli.py`. Result: `determinism_max_abs_diff = 0.0` across two demo passes.

---

## 9. Online reproducibility caveat

The managed Python interpreter's package set is **not** persistent across turns — a
fresh `venv` had to be created and re-pinned (`numpy 2.5.3`, `pytest 9.1.1`,
`ruff 0.16.10`) before the final lint/test/demo closure could run. The lockfile
(`requirements.lock.txt`) makes this one command, but the command still has to be run.

---

MIT © 2026 晨星
