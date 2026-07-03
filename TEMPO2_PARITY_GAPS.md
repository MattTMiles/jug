# Tempo2 parity — gap analysis

**Status (2026-07-02):** `compatibility="tempo2"` is **not production-ready** for nonlinear
timing, autodiff design matrices, or MetaPulsar `NonLinearTimingModel` integration on
tempo2-backed datasets. Treat it as an **in-progress native port**, not as a drop-in
libstempo replacement outside curated tests.

This document complements [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md).
That brief tracks **narrow, fixture-gated parity** (raw pre-fit residuals and selected
design-matrix tests on Cases A/B/C). **Green pytest fixtures do not imply end-to-end
parity** for real IPTA workloads, JAX-traced likelihoods, or MetaPulsar’s nonlinear
timing stack.

---

## Executive summary

| Layer | Tempo2 parity status |
|-------|----------------------|
| Raw pre-fit residuals vs libstempo | **Partially green** on curated fixtures (TCB Case A; NG5 TDB Cases B/C) |
| Linear WLS / host `Mmat` from libstempo | **Usable** in MetaPulsar when timing package is tempo2 and JUG is not on the hot path |
| NumPy `residual_delta` round-trip at θ=0 | **Not closed** — reference vs perturbation paths disagree at ~10–30 ns on IPTA DR2 |
| JAX `residual_delta` / autodiff design matrix | **Not trustworthy** on ELL1/T2 binaries — forward model mismatch at O(1 s) at θ=0 |
| Analytic design matrix in tempo2 mode | **Known broken** — do not use; see TODOs in `optimized_fitter.py` |
| MetaPulsar Discovery NUTS + JUG(tempo2) | **Unsupported today** — do not assume notebook/demo parity |

The tempo2-compatible path was developed quickly to unblock MetaPulsar cross-engine
experiments. **Passing residual parity on a handful of NG5 fixtures is necessary but
far from sufficient.** Investigations that assume “tempo2 JUG should already work” and
then debug MetaPulsar validation to 1 ns are measuring **immaturity of this layer**,
not accidental regressions in otherwise-finished code.

---

## What *is* in good shape (narrow scope)

These items are covered by [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md)
and the `@pytest.mark.tempo2` suite:

- **Case A (TCB):** equatorial TCB fixtures with IF99 / DILATEFREQ / explicit tempo2 keywords.
- **Cases B/C (TDB):** NG5 J1600 equatorial and ecliptic cross-engine fixtures — raw
  `JUG(tempo2) − libstempo` within the project’s ~5 ns gates when run as documented.
- **Phase B/C architecture:** separate tempo2 TDB geometry provider, mode-specific TZR,
  unweighted phase mean in tempo2 mode vs weighted in pint mode.
- **Design matrix parity (TCB fixtures only):** `tests/test_tempo2_designmatrix_parity.py`
  on Case A — not a guarantee for TDB, ELL1, or IPTA parfiles.

**Important:** MetaPulsar notebooks or integration tests that mix **EPTA/PPTA (tempo2) +
NANOGrav (PINT)**, **composite (Borg) host strategies**, or **IPTA DR2 parfiles** are
**outside** this green envelope unless explicitly added as fixtures.

---

## Gap G1 — NumPy nonlinear round-trip: `residual_delta(0) ≠ 0`

**Symptom:** MetaPulsar validates `backend.residual_delta(0) == 0` at `tol=1e-9` s
(`metapulsar.timing.backends.base.validate_backend_zero_delta`). IPTA DR2 J0613-0200
sessions fail at ~**2–3×10⁻⁸ s** (roughly 20–30 ns peak).

**Mechanism (observed 2026-07-02 on EPTA/PPTA/ng9):**

1. Reference residuals exported in `export_jax_timing_state` / `JaxTimingState` are
   computed from `ref_params` where spin parameters (`F0`, `F1`, …) may still be
   **Python `float` / float64**.
2. `residual_delta_np(0)` loops fit parameters and calls `_update_param(..., value + 0)`
   even when the perturbation is zero (`jug/fitting/optimized_fitter.py`).
3. For `HIGH_PRECISION_PARAMS`, `_update_param` **promotes** values to `np.longdouble`
   and syncs `_high_precision` string cache — by design for multi-year F0 phase accuracy.
4. Re-evaluating the full model after that promotion shifts phases by ~**27 ns** peak
   on the EPTA baseline (F0 promotion alone reproduces the full mismatch).

**Interpretation:** This is a **reference-state inconsistency** between “frozen reference”
and “zero perturbation evaluation path”, not libstempo disagreement. It is still a
**blocker for strict nonlinear invariants** MetaPulsar assumes.

**Not yet characterized:** Whether promoting `ref_params` to longdouble at export closes
the gap on all tempo2 fixtures, or whether additional term-by-term drift remains on TDB
ELL1 datasets.

**Code touchpoints:**

- `jug/fitting/optimized_fitter.py` — `_update_param`, `HIGH_PRECISION_PARAMS`
- MetaPulsar `jug_jax_state.py` — `residual_delta_np`, `export_jax_timing_state`

---

## Gap G2 — JAX autodiff path: ELL1 binary model uses DD Keplerian JIT

**Symptom:** `make_residual_delta_jax_fn(...)(0)` returns O(**1 s**) residuals on
J0613-0200 (ELL1/T2 binary in IPTA DR2), while NumPy `residual_delta_np(0)` stays at
O(**10 ns**) from G1 alone.

**Mechanism:**

- `GeneralFitSetup` stores **`initial_binary_delay`** from the true session binary model
  (ELL1/T2 for J0613: parameters include `TASC`, `EPS1`, `EPS2`).
- `jug/fitting/jax_residual_delta.py` recomputes binary delay via
  **`_compute_dd_binary_delay_jit`** (`derivatives_dd.py`), which reads **DD keys**
  (`T0`, `ECC`, `OM`, …).
- At θ=0, `_param_scalar(params, "T0")` and `_param_scalar(params, "ECC")` **default to 0**
  when absent; `TASC` / `EPS1` / `EPS2` are present in `params` but **not consumed** by
  the DD JIT.
- Component breakdown at θ=0: DM and astrometry deltas ≈ 0; **binary
  `new − initial` ≈ 2.1 s** peak — matching the total JAX offset.

**Impact:**

- `design_matrix_method="autodiff"` (`compute_autodiff_designmatrix_from_setup`) builds
  columns from `jax.jacfwd` of this forward model — **columns are wrong** for ELL1
  pulsars until the JAX binary path matches the setup’s binary family.
- Discovery / NumPyro NUTS timing likelihoods that call `residual_delta_jax` inherit the
  same error.

**Scope note:** This gap is triggered on **ELL1/T2 IPTA binaries**, not only on
`compatibility="tempo2"`. It blocks **MetaPulsar + JUG autodiff** broadly until the JAX
forward model respects the binary registry used at setup time (`binary_registry.py`,
`binary_t2_dispatch.py`).

**Code touchpoints:**

- `jug/fitting/jax_residual_delta.py` — `_compute_residual_delta_jax` binary branch
- `jug/fitting/derivatives_dd.py` — DD JIT (wrong family for ELL1 setups)
- `jug/fitting/binary_registry.py` — native binary dispatch (not yet wired into JAX delta)

---

## Gap G3 — JAX delta builder param keys vs backend names

**Symptom:** `_build_params_from_delta` writes perturbed values under **fit-parameter
names** (e.g. `F0_EPTA`, `RAJ_EPTA` in composite MetaPulsar hosts) instead of **backend
JUG names** (`F0`, `RAJ`, … from session param mapping).

**Impact at θ=0:** Spin terms still read correct `F0` from the copied `ref_params`, so
this does not cause the O(1 s) G2 offset. **At θ≠0**, perturbations on suffixed host
names may fail to reach the delay/spin kernels that read unsuffixed backend keys.

**Needed:** Apply the same canonical→backend mapping MetaPulsar uses in
`JugTimingBackend.from_session` when building the JAX param dict.

---

## Gap G4 — Analytic design matrix in tempo2 mode

The legacy **analytic** derivative columns in tempo2 mode were a PINT-parity mistake and
are **known broken** on tempo2-compatible setups (including ecliptic astrometry). Code
TODOs in `optimized_fitter.py` document this; **`design_matrix_method="analytic"` must not
be used** for tempo2 sessions.

Autodiff was introduced to replace finite-difference “autodiff” and avoid hand-maintained
tempo2 derivative blocks. **Autodiff is only as good as G2/G3 fix the JAX forward model.**

Existing green design-matrix tests (`tests/test_tempo2_designmatrix_parity.py`) target
**TCB Case A fixtures**, not TDB ELL1 IPTA parfiles.

---

## Gap G5 — Fixture coverage vs real workloads

| Workload | In tempo2 parity CI? | Known issue |
|----------|----------------------|-------------|
| NG5 J1600 Cases B/C | Yes | Green on raw residuals (narrow par) |
| TCB Case A | Yes | Green |
| IPTA DR2 multi-PTA (EPTA+PPTA+ng9) | **No** | G1 validation failure; G2 on ELL1 |
| Composite (Borg) host strategy | **No** | Suffixed param names (G3) |
| ELL1/T2 binaries (J0613, etc.) | **Partial** | `ppta_j1741_ell1` documented ~5–8 ns residual gap in project brief §5; G2 at JAX layer |
| Ecliptic ng9 GLS (`LAMBDA`/`BETA`) | **No** | Analytic path broken; autodiff untested to parity |
| `DM_SERIES` and other ignored keywords | Warn-only | Documented in project brief |

MetaPulsar notebooks such as `nlt_ipta_dr2_compare.ipynb` sit **squarely in the “No” column**
when they route tempo2 PTAs through JUG nonlinear/autodiff paths.

---

## Gap G6 — Documented residual parity debt (project brief)

[`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md) §5 already lists
secondary gaps, including:

- **`ppta_j1741_ell1`:** RMS ~5–8 ns vs strict 5 ns gate; orbital-harmonic structure /
  ELL1 convention mismatch.
- **`DM_SERIES`:** ignored by JUG on several fixtures.

These are **honest residual-level debts** on top of the nonlinear/JAX gaps above.

---

## MetaPulsar integration — practical guidance

Until gaps G1–G3 are closed and covered by tests:

1. **`compatibility="pint"`** — preferred JUG mode for MetaPulsar nonlinear timing,
   autodiff, and Discovery NUTS on PINT-backed sessions.
2. **`compatibility="tempo2"`** — acceptable for **experiments** comparing raw residuals
   to libstempo on harmonized fixtures; **not** for production nonlinear inference without
   explicit parity sign-off on your par/tim combination.
3. **`design_matrix_method="autodiff"`** — do **not** use on ELL1/T2 binaries until G2 is
   fixed; do **not** use **`analytic`** on tempo2 sessions (G4).
4. **Enterprise / libstempo linear timing** on tempo2 PTAs remains the reliable path when
   the host timing package is tempo2 — JUG is not required for that workflow.
5. **Strict `residual_delta(0)` checks at 1 ns** are ahead of current tempo2-compatible
   numerical closure; failing them is **expected** today, not a signal to tune tolerances
   without fixing reference-state parity (G1).

---

## Forward work themes (gap-closing)

Ordered by blocker severity for MetaPulsar nonlinear + autodiff:

1. **Binary JAX forward model:** wire `jax_residual_delta` to the same binary family as
   `GeneralFitSetup` (ELL1/T2/DD via `binary_registry` / tempo2 dispatch), not DD-only JIT.
2. **Reference-state closure (G1):** ensure θ=0 JAX and NumPy paths use identical param
   storage (longdouble promotion, ecliptic sync) as reference residual export.
3. **Param mapping in JAX builder (G3):** map host/composite fit names → backend JUG names.
4. **Acceptance tests beyond Case A/B/C:** IPTA-representative ELL1 TDB fixtures with
   raw residual gates **and** `residual_delta(0)` / autodiff-vs-libstempo-column checks.
5. **Documentation sync:** keep this gap list updated when fixtures move from “No” to “Yes”;
   avoid implying Phase E completion on TDB ELL1 until tests exist.

---

## Investigation log (2026-07-02)

Informal MetaPulsar dev investigation (scripts under `/tmp/metapulsar_rd0_investigation/`,
not part of this repo) on IPTA DR2 J0613-0200 reproduced G1 and G2 on:

- `single_epta` (EPTA tempo2),
- `multi_consistent` (EPTA + PPTA tempo2 + ng9 pint).

Findings are summarized above; raw JSON summary: `summary.json` in that temp directory
when the investigation was run in the MetaPulsar devcontainer.

---

## Related documents

| Document | Role |
|----------|------|
| [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md) | Implementation brief, locked decisions, Cases A/B/C work plan |
| [`README.md`](README.md) | Install, compatibility modes, pytest entry points |
| MetaPulsar `examples/notebooks-dev/NONLINREFACTOR-SPEC.md` | Nonlinear timing architecture (consumer side) |

**When updating parity status:** if a gap closes, add a fixture + pytest gate **first**,
then remove or downgrade the row here. Do not mark gaps closed based on ad hoc notebook
runs alone.
