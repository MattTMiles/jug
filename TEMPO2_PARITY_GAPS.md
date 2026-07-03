# Tempo2 parity — gap analysis

**Status (2026-07-03, verified):** `compatibility="tempo2"` remains **experimental** for
production nonlinear inference, but the **G1/G2 blockers that stopped IPTA DR2 J0613
autodiff notebooks are closed.** JUG autodiff + MetaPulsar `NonLinearTimingModel`
whitening works on the **`multi_consistent`** IPTA DR2 workload (EPTA + PPTA tempo2 +
NANOGrav 9y pint) when `design_matrix_method="autodiff"`.

Treat tempo2 mode as an **in-progress native port**, not a drop-in libstempo replacement
outside curated tests. Do not use **`design_matrix_method="analytic"`** on tempo2
sessions (G4).

This document complements [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md).
That brief tracks **narrow, fixture-gated parity** (raw pre-fit residuals and selected
design-matrix tests on Cases A/B/C). **Green pytest fixtures do not imply end-to-end
parity** for every IPTA host strategy, θ≠0 NumPy/JAX agreement, or composite (Borg)
workloads.

---

## Executive summary

| Layer | Tempo2 parity status |
|-------|----------------------|
| Raw pre-fit residuals vs libstempo | **Partially green** on curated fixtures (TCB Case A; NG5 TDB Cases B/C) |
| Linear WLS / host `Mmat` from libstempo | **Usable** in MetaPulsar when timing package is tempo2 and JUG is not on the hot path |
| NumPy `residual_delta` round-trip at θ=0 | **Closed (G1, 2026-07-03)** — MetaPulsar reads `HIGH_PRECISION_PARAMS` via `get_longdouble()` before `_update_param` |
| JAX `residual_delta` at θ=0 / autodiff binary dispatch | **Closed (G2, 2026-07-03)** — shared `compute_total_delay_change` + `BinaryDelayPlan`; θ=0 peak ≲10⁻¹³ s on IPTA DR2 J0613 |
| NumPy vs JAX at θ≠0 on real IPTA workloads | **Open (G2 residual)** — ms-level mismatch on binary/astrometry perturbations; not gated by CI |
| Analytic design matrix in tempo2 mode | **Known broken (G4)** — do not use; see TODOs in `optimized_fitter.py` |
| MetaPulsar NTM whitening, IPTA DR2 `multi_consistent` | **Green ad hoc (2026-07-03)** — not yet a pytest gate in either repo |
| MetaPulsar NTM whitening, composite (Borg) strategy | **Open (G3/G5)** — Schur Fisher not PD on J0613 composite host |
| MetaPulsar Discovery NUTS + JUG(tempo2) | **Experimental** — `multi_consistent` path unblocked at reference point; full NUTS not CI-gated |

**Gap scorecard:** **2 closed** (G1, G2 primary), **5 with open items** (G3, G4, G5, G6, G7).

---

## What *is* in good shape (verified 2026-07-03)

### JUG pytest (run with `pytest -o addopts=` if `pytest-cov` is missing)

- **`tests/test_designmatrix_autodiff.py`** — 23 tests: synthetic spin/DM, all binary
  families (ELL1, T2, ELL1H, DD, DDK, ecliptic PM), `residual_delta_jax(0)==0`, NumPy/JAX
  delay parity.
- **`tests/test_autodiff_j0613_mpta.py`** — trimmed real J0613 (ELL1H): autodiff zero-delta
  and finite design matrix.
- **`tests/test_tempo2_residual_parity.py`** + **`tests/test_tempo2_designmatrix_parity.py`**
  (`@pytest.mark.tempo2`) — Cases A/B/C residual and design-matrix gates; 17 passed,
  1 intentional xfail, 2 deselected (2026-07-03 run).

### MetaPulsar pytest

- **`tests/test_timing_jug_jax_backend.py`** — G1 regression, pint/tempo2 JAX vs NumPy
  on toy linear model (5/5 passed).

### IPTA DR2 J0613 ad hoc integration (MetaPulsar devcontainer, 2026-07-03)

Configuration mirrors `examples/notebooks-dev/nlt_ipta_dr2_compare_jug.ipynb`:
`ENGINES={"tempo2":"jug","pint":"jug"}`, `design_matrix_method="autodiff"`.

| Scenario | `residual_delta(0)` | `residual_delta_jax(0)` | `validate_backend_zero_delta` | NTM whitening |
|----------|---------------------|-------------------------|--------------------------------|---------------|
| `single_epta` | 0 | ≲6×10⁻¹⁴ s | pass | pass |
| **`multi_consistent`** | 0 | ≲6×10⁻¹⁴ s | pass | **pass** |
| `multi_composite` | 0 | ≲6×10⁻¹⁴ s | pass | fail (Fisher not PD) |

Per-PTA on `multi_consistent`: epta, ppta (tempo2), ng9 (pint) all pass zero-delta at
machine precision.

### Narrow fixture envelope (unchanged)

- **Case A (TCB):** equatorial TCB fixtures with IF99 / DILATEFREQ / explicit tempo2 keywords.
- **Cases B/C (TDB):** NG5 J1600 equatorial and ecliptic cross-engine fixtures.
- **Phase B/C architecture:** separate tempo2 TDB geometry provider, mode-specific TZR,
  unweighted phase mean in tempo2 mode vs weighted in pint mode.

---

## Gap G1 — NumPy nonlinear round-trip: `residual_delta(0) ≠ 0` — **CLOSED (2026-07-03)**

**Symptom (historical):** MetaPulsar validates `backend.residual_delta(0) == 0` at
`tol=1e-9` s (`metapulsar.timing.backends.base.validate_backend_zero_delta`). IPTA DR2
J0613-0200 sessions failed at ~**2–3×10⁻⁸ s** (roughly 20–30 ns peak) on both
`compatibility="tempo2"` and `compatibility="pint"`.

**Mechanism (observed 2026-07-02 on EPTA/PPTA/ng9):**

1. Reference residuals exported in `export_jax_timing_state` / `JaxTimingState` are
   computed from `ref_params` where spin parameters (`F0`, `F1`, …) may still be
   **Python `float` / float64** in the slot, while the authoritative value lives in
   `_high_precision`.
2. `residual_delta_np(0)` used `current = float(params.get(...))` then
   `_update_param(..., current + 0)` even when the perturbation is zero.
3. For `HIGH_PRECISION_PARAMS`, `_update_param` **promotes** the float64-rounded value to
   `np.longdouble` and rewrites `_high_precision` — degrading F0 by ~10⁻¹⁴ Hz.
4. Re-evaluating the full model after that promotion shifted phases by ~**27 ns** peak
   on the EPTA tempo2 baseline (F0 alone after astrometry normalization reproduced the
   full mismatch).

**Interpretation:** Reference-state inconsistency between “frozen reference” (which reads
full precision via `get_longdouble()`) and “zero perturbation evaluation path” (which
round-tripped through float64). Not libstempo disagreement.

**Closure (2026-07-03):** MetaPulsar `jug_jax_state.py` `residual_delta_np` now reads
`HIGH_PRECISION_PARAMS` via `get_longdouble()` and applies perturbations in
`np.longdouble` before `_update_param`. IPTA DR2 J0613 `multi_consistent` passes strict
`validate_backend_zero_delta(tol=1e-9)` on epta, ppta, and ng9. Regression test:
`metapulsar/tests/test_timing_jug_jax_backend.py` —
`test_jax_timing_state_residual_delta_np_preserves_high_precision_f0`.

**Verified 2026-07-03:** Still closed on all three J0613 scenarios above.

**Residual note:** JUG core still accepts float64 slots when `_high_precision` is
authoritative; any caller that round-trips high-precision params through `float()` before
`_update_param` can reintroduce this class of bug. Prefer `get_longdouble()` on read.

**Code touchpoints:**

- MetaPulsar `jug_jax_state.py` — `residual_delta_np` (fix)
- `jug/fitting/optimized_fitter.py` — `_update_param`, `HIGH_PRECISION_PARAMS`

---

## Gap G2 — JAX autodiff path binary mismatch — **CLOSED at θ=0 (2026-07-03); residual open at θ≠0**

**Historical symptom:** `make_residual_delta_jax_fn(...)(0)` returned O(**1 s**) residuals
on J0613-0200 (ELL1/T2 binary in IPTA DR2), while NumPy `residual_delta_np(0)` was 0.
This blocked `nlt_ipta_dr2_compare_jug.ipynb` at `NonLinearTimingModel.timing_param_keys()`
(Schur Fisher not PD — a whitening symptom of the θ=0 offset).

**Mechanism:**

- `GeneralFitSetup` stores **`initial_binary_delay`** from the true session binary model
  (ELL1/T2 for J0613: parameters include `TASC`, `EPS1`, `EPS2`).
- The old JAX path recomputed binary delay via a **DD-only JIT** (`derivatives_dd.py`),
  reading `T0`/`ECC`/`OM` keys that default to 0 when absent instead of consuming
  `TASC`/`EPS1`/`EPS2`.
- Component breakdown at θ=0: DM and astrometry deltas ≈ 0; **binary
  `new − initial` ≈ 2.1 s** peak — matching the total JAX offset.

**Closure (2026-07-03):**

- Added shared `compute_total_delay_change(..., xp=...)` used by both NumPy and JAX.
- Added `BinaryDelayPlan` / `resolve_binary_structure` and routed builtin binary models
  through one structural dispatcher for DD/ELL1/DDK/T2.
- Removed JAX hardcoded DD binary delay path and wired FDJUMP + unified DM handling.
- Added synthetic and trimmed-J0613 autodiff regressions to guard `delta(0)==0` and
  NumPy/JAX delay parity across binary families.

**Verified 2026-07-03:**

- JUG: 25/25 autodiff tests pass (`test_designmatrix_autodiff.py`,
  `test_autodiff_j0613_mpta.py`).
- MetaPulsar IPTA DR2 J0613: `residual_delta_jax(0)` peak ≲6×10⁻¹⁴ s on all PTAs and
  composite backends (was ≈2.1 s).
- **`multi_consistent` NTM whitening passes** — notebook blocker resolved.

**G2 residual (open, not separately numbered):** On real IPTA DR2 J0613 with a 1e-8
perturbation on binary/astrometry fitpars, NumPy vs JAX `residual_delta` disagree by
**~3–8 ms** peak (both `multi_consistent` and per-session engines). Synthetic JUG fixtures
still show tight NumPy/JAX parity. Likely MetaPulsar integration / float64-vs-longdouble
at θ≠0; **not gated by CI.** Autodiff design matrices inherit this forward-model mismatch
at nonzero θ but are correct at θ=0 where whitening is assembled.

**Code touchpoints:**

- `jug/fitting/jax_residual_delta.py` — `_compute_residual_delta_jax`, `make_residual_delta_jax_fn`
- `jug/fitting/forward_delay.py` — `compute_total_delay_change`
- `jug/fitting/binary_delay_plan.py` — `resolve_binary_structure`
- `jug/fitting/binary_registry.py` — native binary dispatch (now wired into JAX delta)

---

## Gap G3 — JAX delta builder param keys vs backend names — **OPEN (composite); partially mitigated**

**Symptom:** `_build_params_from_delta` writes perturbed values under **fit-parameter
names** passed to `make_residual_delta_jax_fn`. MetaPulsar `export_jax_timing_state`
maps canonical host names to backend JUG names (`F0_epta` → `F0`) before building the JAX
fn, and `JaxTimingState.residual_delta_np` uses `param_mapping` on the NumPy path.

**Impact at θ=0:** Resolved — G2 closure; suffixed names do not cause O(1 s) offsets when
MetaPulsar supplies the mapping.

**Impact at θ≠0:** Composite (Borg) hosts with suffixed fitpars still show ms-level NumPy/JAX
mismatch on spin/binary perturbations (e.g. `F0_epta`, `A1_epta`). The same mismatch also
appears on **`multi_consistent`** (unsuffixed names), so G3 alone does not explain all
θ≠0 divergence — see G2 residual.

**Impact on NTM:** `multi_composite` J0613 fails Schur whitening (Fisher not PD) even though
θ=0 checks pass. Treat composite + all-JUG autodiff as **unsupported** until gated.

**Needed:**

1. Close θ≠0 NumPy/JAX parity on real IPTA sessions (may span G2 residual + mapping).
2. Add MetaPulsar integration tests for composite host perturbations.
3. Promote `multi_consistent` IPTA DR2 checks from ad hoc runs to pytest gates.

---

## Gap G4 — Analytic design matrix in tempo2 mode — **OPEN**

The legacy **analytic** derivative columns in tempo2 mode were a PINT-parity mistake and
are **known broken** on tempo2-compatible setups (including ecliptic astrometry). Code
TODOs in `optimized_fitter.py` document this; **`design_matrix_method="analytic"` must not
be used** for tempo2 sessions.

Autodiff replaces hand-maintained tempo2 derivative blocks. **Use `design_matrix_method="autodiff"`**
for tempo2 nonlinear timing (G2 θ=0 path is closed).

Existing green design-matrix tests (`tests/test_tempo2_designmatrix_parity.py`) compare
**analytic** columns to libstempo on curated fixtures — not an endorsement of analytic
mode for production tempo2 sessions.

---

## Gap G5 — Fixture coverage vs real workloads

| Workload | In tempo2 parity CI? | Status (2026-07-03) |
|----------|----------------------|---------------------|
| NG5 J1600 Cases B/C | Yes | Green on raw residuals (narrow par) |
| TCB Case A | Yes | Green |
| IPTA DR2 EPTA J0613 `single_epta` (full TIM) | **Yes (2026-07-03)** | **No parity:** ~2.9 ms RMS vs libstempo (gate 5 ns) |
| IPTA DR2 EPTA J0613 single-backend excerpt | **Yes (2026-07-03)** | **Documented gap:** ~62 ns RMS vs libstempo |
| IPTA DR2 multi-PTA `multi_consistent` (J0613) | **No** | **Green ad hoc** — zero-delta + NTM whitening pass; not CI-gated |
| Composite (Borg) host strategy | **No** | θ=0 green; NTM whitening fails (Fisher not PD); G3 |
| ELL1/T2 binaries (J0613, etc.) | **Partial** | JUG autodiff + trimmed J0613 green; `ppta_j1741_ell1` residual debt (G6) |
| Ecliptic ng9 GLS (`LAMBDA`/`BETA`) | **Partial** | Raw residual parity green (Case C); tempo2 autodiff not IPTA-gated |
| `DM_SERIES` and other ignored keywords | Warn-only | Documented in project brief (G6) |

`nlt_ipta_dr2_compare_jug.ipynb` with **`multi_consistent`** is **no longer blocked** at
NTM binding / whitening by G2. **`multi_composite`** and full Discovery NUTS runs remain
experimental. Mixed-engine `nlt_ipta_dr2_compare.ipynb` (libstempo tempo2 + JUG pint,
analytic) is a separate integration path.

---

## Gap G6 — Documented residual parity debt (project brief) — **OPEN**

[`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md) §5 lists secondary
gaps, including:

- **`ppta_j1741_ell1`:** RMS ~5–8 ns vs strict 5 ns gate; orbital-harmonic structure /
  ELL1 convention mismatch. Test `test_tempo2_mode_ell1_j1741_documented_gap` passes
  documented bounds (2026-07-03).
- **`DM_SERIES`:** ignored by JUG on several fixtures (warn-only; observed on ng9 J0613).

These are **honest residual-level debts** independent of the G1/G2 nonlinear fixes.

---

## Gap G7 — IPTA DR2 EPTA multi-backend raw residuals — **OPEN (documented 2026-07-03)**

**Symptom:** On IPTA DR2 EPTA **J0613-0200** (`J0613-0200.par` + full TIM, 1369 TOAs),
`JUG(compatibility="tempo2")` pre-fit residuals do **not** match libstempo/tempo2.
This is the direct cause of MCMC differences when swapping libstempo → JUG(tempo2)
on the same single-PTA dataset.

**Measured on bundled fixture** `epta_j0613_t2_ipta_all`
(`tests/test_tempo2_ipta_dr2_j0613_parity.py`), same par+tim pair evaluated by both engines:

| Quantity | Gate (green fixtures) | J0613 EPTA |
|----------|----------------------|------------|
| TOA count | JUG == libstempo | **1369 == 1369** |
| RMS Δ | < 5 ns | **2.89×10⁶ ns (~2.9 ms)** |
| p99 \|Δ\| | < 10 ns | **4.56×10⁶ ns** |
| max \|Δ\| | < 25 ns | **4.88×10⁶ ns (~4.9 ms)** |
| WRMS Δ | < 5 ns | **~3.0×10⁶ ns** |

**Verdict:** **No residual parity.** JUG(tempo2) is not a drop-in replacement for
libstempo on this pulsar/dataset.

**Tests:**
- `test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity` — standard
  `_assert_residual_parity` gate (xfail strict)
- `test_epta_j0613_ipta_dr2_parity_debt_is_large` — pins measured debt in CI

---

## MetaPulsar integration — practical guidance

1. **`design_matrix_method="autodiff"`** — **recommended** for JUG nonlinear timing on
   binary IPTA workloads. G2 θ=0 closure makes this viable for **`multi_consistent`**
   IPTA DR2 hosts. Do **not** use **`analytic`** on tempo2 sessions (G4).
2. **`compatibility="pint"`** — preferred JUG mode for PINT-native sessions; unchanged.
3. **`compatibility="tempo2"`** — acceptable for raw-residual experiments vs libstempo on
   harmonized fixtures and for **`multi_consistent`** IPTA DR2 autodiff at the reference
   point. Not signed off for production tempo2 inference on every par/tim combination.
4. **Composite (Borg) host strategy** — avoid all-JUG autodiff until G3/G5 composite
   whitening is green and CI-gated.
5. **Enterprise / libstempo linear timing** on tempo2 PTAs remains the reliable path when
   JUG is not on the hot path.
6. **Zero-delta checks:** failing `validate_backend_zero_delta` on JUG backends likely
   indicates a float64 round-trip regression (G1 class), not libstempo disagreement.
   θ=0 JAX checks pass on IPTA DR2 J0613 after G2 closure.
7. **θ≠0 NumPy/JAX agreement** — not yet a release gate; ms-level gaps remain on real IPTA
   binary perturbations (G2 residual).

---

## Forward work themes (gap-closing)

Ordered by blocker severity for MetaPulsar nonlinear + autodiff:

1. **θ≠0 NumPy/JAX parity (G2 residual):** diagnose ms-level mismatch on IPTA DR2 J0613
   binary/astrometry perturbations; add MetaPulsar integration pytest gates.
2. **Composite / Borg hosts (G3/G5):** fix suffixed-param JAX path and Schur Fisher PD for
   `multi_composite`; add CI fixtures.
3. **Acceptance tests beyond Case A/B/C:** promote IPTA DR2 `multi_consistent` checks
   (zero-delta, `residual_delta_jax(0)`, NTM whitening) to pytest in MetaPulsar.
4. **Analytic tempo2 columns (G4):** either repair or permanently deprecate; autodiff is
   the supported path.
5. **Residual debts (G6):** narrow `ppta_j1741_ell1` gap; implement or explicitly reject
   `DM_SERIES`.
6. **Documentation sync:** keep this gap list aligned with pytest gates — do not mark
   items closed from ad hoc notebook runs alone without tests.

**Closed (2026-07-03):** G1 (NumPy reference-state). G2 primary symptom (JAX θ=0 binary
dispatch, notebook whitening blocker on `multi_consistent`).

---

## Investigation log

### 2026-07-02

Informal MetaPulsar dev investigation (scripts under `/tmp/metapulsar_rd0_investigation/`,
not part of this repo) on IPTA DR2 J0613-0200 reproduced G1 and G2 on:

- `single_epta` (EPTA tempo2),
- `multi_consistent` (EPTA + PPTA tempo2 + ng9 pint).

Findings are summarized above; raw JSON summary: `summary.json` in that temp directory
when the investigation was run in the MetaPulsar devcontainer.

### 2026-07-03 (morning)

- **G1 closed:** MetaPulsar `jug_jax_state.residual_delta_np` uses `get_longdouble()` for
  `HIGH_PRECISION_PARAMS`. Old float64 path reproduced ~27 ns on EPTA tempo2; fixed path
  is machine zero. Strict `validate_backend_zero_delta(tol=1e-9)` passes on all three
  sessions in `multi_consistent`.
- **G2 code landed:** unified binary dispatch in JAX forward model; JUG autodiff test
  suite added.

### 2026-07-03 (afternoon — verification pass)

Re-ran pytest and IPTA DR2 J0613 integration checks:

- JUG autodiff: **25/25 passed**; tempo2 parity: **17 passed**, 1 intentional xfail.
- MetaPulsar `test_timing_jug_jax_backend.py`: **5/5 passed**.
- IPTA DR2 J0613: **`residual_delta_jax(0)` ≲6×10⁻¹⁴ s** (was ≈2.1 s); **`multi_consistent`
  NTM whitening passes**; `multi_composite` whitening still fails (Fisher not PD).
- θ≠0 probe (1e-8 perturbation): ms-level NumPy/JAX mismatch on binary/astrometry params
  on real IPTA data — tracked as G2 residual, not CI-gated.

**Correction:** The morning log entry “G2 confirmed as notebook blocker” is **superseded**
by the afternoon verification — G2 θ=0 is closed and the `multi_consistent` notebook
path is unblocked at NTM binding.

---

## Related documents

| Document | Role |
|----------|------|
| [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md) | Implementation brief, locked decisions, Cases A/B/C work plan |
| [`README.md`](README.md) | Install, compatibility modes, pytest entry points |
| MetaPulsar `examples/notebooks-dev/NONLINREFACTOR-SPEC.md` | Nonlinear timing architecture (consumer side) |

**When updating parity status:** if a gap closes, add a fixture + pytest gate **first**,
then remove or downgrade the row here. Ad hoc integration runs (as above) document status
between gates but do not substitute for CI.
