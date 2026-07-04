# Tempo2 parity — gap analysis

**Status (2026-07-04, verified):** `compatibility="tempo2"` remains **experimental**
outside curated par+tim tests. **Residual parity vs libstempo/tempo2 (JUG scope):** native
PPTA IPTA DR2 ~**1.4 ns** RMS; alternate PPTA export par/tim ~**16 ns**; EPTA full
fixture ~**608 ns** RMS (G7; `-addsat` TOAs sub-µs).

Treat tempo2 mode as an **in-progress native port**, not a drop-in libstempo replacement
outside curated tests. Do not use **`design_matrix_method="analytic"`** on tempo2
sessions (G4).

This document complements [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md).
That brief tracks **narrow, fixture-gated parity** (raw pre-fit residuals and selected
design-matrix tests on Cases A/B/C). **Green pytest fixtures do not imply end-to-end
parity** for every par/tim pair, or θ≠0 NumPy/JAX agreement on real IPTA workloads.

### Scope — what JUG parity is (and is not)

**JUG parity** means: given a `.par` file and a `.tim` file, pre-fit residuals (and,
where gated, design-matrix columns) from `JUG(compatibility="tempo2")` match
libstempo/tempo2 on **the same inputs**. Nothing else participates in that definition.

**MetaPulsar has nothing to do with JUG parity.** When a notebook or export path is named
in this document, that is **dataset provenance only** — a way to identify or obtain a
particular par+tim pair. It does **not** imply that any downstream orchestration affects
JUG↔libstempo residual parity.

---

## Executive summary

| Layer | Tempo2 parity status |
|-------|----------------------|
| Raw pre-fit residuals vs libstempo | **Partially green** on curated par+tim fixtures (TCB Case A; NG5 TDB Cases B/C) |
| NumPy `residual_delta` round-trip at θ=0 (G1) | **Closed (2026-07-03)** — reference-state float64 vs `longdouble` |
| JAX `residual_delta` at θ=0 / autodiff binary dispatch (G2) | **Closed (2026-07-03)** — shared `compute_total_delay_change` + `BinaryDelayPlan`; θ=0 peak ≲10⁻¹³ s on IPTA DR2 J0613 par+tim |
| NumPy vs JAX at θ≠0 on real IPTA workloads | **Open (G2 residual)** — ms-level mismatch on binary/astrometry perturbations; not gated by CI |
| Analytic design matrix in tempo2 mode | **Known broken (G4)** — do not use; see TODOs in `optimized_fitter.py` |

**Gap scorecard:** **2 closed** (G1, G2 primary), **4 with open items** (G4, G5, G6, G7).

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

### Narrow fixture envelope (unchanged)

- **Case A (TCB):** equatorial TCB fixtures with IF99 / DILATEFREQ / explicit tempo2 keywords.
- **Cases B/C (TDB):** NG5 J1600 equatorial and ecliptic cross-engine fixtures.
- **Phase B/C architecture:** separate tempo2 TDB geometry provider, mode-specific TZR,
  unweighted phase mean in tempo2 mode vs weighted in pint mode.

---

## Gap G1 — NumPy nonlinear round-trip: `residual_delta(0) ≠ 0` — **CLOSED (2026-07-03)**

**Symptom (historical):** Zero-perturbation `residual_delta(0)` was non-zero at ~**2–3×10⁻⁸ s**
(roughly 20–30 ns peak) on IPTA DR2 J0613-0200 par+tim for both `compatibility="tempo2"`
and `compatibility="pint"`.

**Mechanism (observed 2026-07-02 on EPTA/PPTA/ng9):**

1. Reference residuals in exported timing state may use `ref_params` where spin parameters
   (`F0`, `F1`, …) are still **Python `float` / float64** in the slot, while the
   authoritative value lives in `_high_precision`.
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

**Closure (2026-07-03):** Callers must read `HIGH_PRECISION_PARAMS` via `get_longdouble()`
and apply perturbations in `np.longdouble` before `_update_param`. IPTA DR2 J0613 epta,
ppta, and ng9 par+tim pairs pass strict zero-delta checks at `tol=1e-9` s.

**Verified 2026-07-03:** Closed on IPTA DR2 J0613 par+tim workloads tested at the time.

**Residual note:** JUG core still accepts float64 slots when `_high_precision` is
authoritative; any caller that round-trips high-precision params through `float()` before
`_update_param` can reintroduce this class of bug. Prefer `get_longdouble()` on read.

**Code touchpoints:**

- `jug/fitting/optimized_fitter.py` — `_update_param`, `HIGH_PRECISION_PARAMS`

---

## Gap G2 — JAX autodiff path binary mismatch — **CLOSED at θ=0 (2026-07-03); residual open at θ≠0**

**Historical symptom:** `make_residual_delta_jax_fn(...)(0)` returned O(**1 s**) residuals
on J0613-0200 (ELL1/T2 binary in IPTA DR2), while NumPy `residual_delta_np(0)` was 0.

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
- IPTA DR2 J0613 par+tim: `residual_delta_jax(0)` peak ≲6×10⁻¹⁴ s (was ≈2.1 s).

**G2 residual (open, not separately numbered):** On real IPTA DR2 J0613 par+tim with a
1e-8 perturbation on binary/astrometry fitpars, NumPy vs JAX `residual_delta` disagree by
**~3–8 ms** peak. Synthetic JUG fixtures still show tight NumPy/JAX parity. Likely
float64-vs-longdouble at θ≠0; **not gated by CI.** Autodiff design matrices inherit this
forward-model mismatch at nonzero θ but are correct at θ=0.

**Code touchpoints:**

- `jug/fitting/jax_residual_delta.py` — `_compute_residual_delta_jax`, `make_residual_delta_jax_fn`
- `jug/fitting/forward_delay.py` — `compute_total_delay_change`
- `jug/fitting/binary_delay_plan.py` — `resolve_binary_structure`
- `jug/fitting/binary_registry.py` — native binary dispatch (now wired into JAX delta)

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
| IPTA DR2 EPTA J0613 full TIM | **Yes (2026-07-03)** | **Partial:** ~608 ns RMS vs libstempo (was ~46.8 ms; gate 5 ns) |
| IPTA DR2 EPTA J0613 single-backend excerpt | **Yes (2026-07-03)** | **Documented gap:** ~62 ns RMS vs libstempo (TRACK -2/-pn; unchanged) |
| IPTA DR2 PPTA J0613 native (`PPTA_dr1dr2` par/tim) | **No (2026-07-04)** | **Green ad hoc:** ~1.4 ns RMS vs libstempo (410 TOAs; no `-pn` in native tim) |
| IPTA DR2 PPTA J0613 alternate export par/tim | **No (2026-07-04)** | **Partial:** ~16 ns RMS vs libstempo (`TRACK -2`, `-pn` tim; see 2026-07-04 section) |
| ELL1/T2 binaries (J0613, etc.) | **Partial** | JUG autodiff + trimmed J0613 green; `ppta_j1741_ell1` residual debt (G6) |
| Ecliptic ng9 GLS (`LAMBDA`/`BETA`) | **Partial** | Raw residual parity green (Case C); tempo2 autodiff not IPTA-gated |
| `DM_SERIES` and other ignored keywords | Warn-only | Documented in project brief (G6) |

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

## Pulse phase / pulse-number semantics (2026-07-03)

Investigation of G7 (IPTA DR2 EPTA J0613 multi-backend residuals) showed the dominant
debt is **pulse-turn assignment**, not Roemer/Shapiro or DM kernels. This section records
how PINT, tempo2, and JUG differ — and what changed in the J0613 fixtures.

### JUG architecture: split at delays, shared phase path

`compatibility="pint"` and `compatibility="tempo2"` diverge in **barycentric geometry**
(`PintDelayProvider` vs `Tempo2DelayProvider`), engine conventions, TZR handling, binary
param normalization, and **mean subtraction** (weighted vs unweighted). They share a
single `compute_phase_residuals()` in `jug/residuals/simple_calculator.py`. The only
mode-specific knob inside that function is `mean_mode`.

PINT parity on `J1909_proper` (default `compatibility="pint"`) does **not** mean JUG
implements PINT `track_mode` machinery; on benign single-backend chronological data the
shared hybrid algorithm happens to agree with PINT `nearest`, tempo2 default, and JUG.

### Native semantics (summary)

| Aspect | PINT | tempo2 | JUG (both modes) |
|--------|------|--------|------------------|
| Reference TOA | Index 0 (`nearest`) or TZRMJD (`use_pulse_numbers`) | Index 0 (`obsn[0]` / `bbat[0]`) | **Earliest emission time** `argsort(dt)[0]` |
| Default wrapping | Fractional part after subtracting index-0 phase | `phas1` subtracted from all | Sequential phase-connection by `dt` |
| `TRACK -2` | `model.phase(abs) − pulse_number[col]` | `pnNew` from `bbat[i]−bbat[0]` vs `-pn` | `phas1`@index 0 + per-TOA `fortran_nlong(phase5[i])` (2026-07-03) |
| Mean removal | Weighted | Unweighted | Mode-dependent |
| Pulse path split by compatibility? | — | — | **No** |

**PINT** (`ref-packages/PINT/src/pint/residuals.py`):

- `nearest` (default): `model.phase(toas)` relative; subtract first TOA phase; keep
  fractional part only; weighted mean.
- `use_pulse_numbers` (`TRACK=-2` or `pulse_number` column): `model.phase(abs_phase=True)`
  anchored to TZRMJD; residual = model phase − stored integer per TOA.
- `compute_pulse_numbers()`: independent `int(model.phase(abs))` per TOA (not sequential).

**tempo2** (`ref-packages/tempo2`: `formResiduals.C`, `add_pulseNumber` plugin):

- Default: `phas1` = fractional phase of first **active** TOA in **tim-file index order**.
- `TRACK -2`: `pnNew[i] = nint(phase[i]) + (bbat[i]−bbat[0])` compared to `-pn` flags.
- `-pn` values are **relative to tim index 0**: `pulseN[i] − pulseN[0]`.
- Unweighted mean removal.

**JUG** (`compute_phase_residuals()`):

1. Spin Taylor phase at emission time `dt` (plus glitches, JUMPs).
2. Subtract TZR phase when `subtract_tzr=True`.
3. Sort by `dt`; assign integer pulse numbers sequentially (phase-connected wrapping).
4. If `TRACK=-2` and all `-pn` flags present: `phas1` from tim index 0; per-TOA
   `fortran_nlong(phase5[i])` fractional residual (2026-07-03 fix).
5. Subtract mean (weighted in pint mode, unweighted in tempo2 mode).

This is a **third variant**: neither PINT `nearest`, nor PINT `use_pulse_numbers`, nor
tempo2 `formResiduals`.

### G7 root cause (pre-pulse-number fixtures)

On J0613 `epta_j0613_t2_ipta_all` **without** explicit pulse numbers:

- ~2.9 ms RMS vs libstempo; max ~4.9 ms ≈ 1.6 pulse turns (F0 ≈ 326.6 Hz).
- Per-backend constant ±2–3 ms offsets that cancel globally.
- Non-chronological INCLUDE order: `obsn[0]` is JBO (~MJD 54847) while WSRT is earliest
  (~52958). JUG anchors at earliest `dt`; tempo2 anchors at tim index 0.
- Isolated sub-TIMs (JBO+NRT) match to ~0 ns; cross-backend combination amplifies the gap.

### J0613 fixture update (2026-07-03): `TRACK -2` + `-pn`

Both J0613 tempo2 fixtures now ship explicit pulse numbers so parity work is not
confounded by ambiguous default wrapping:

| Fixture | Change | Generator |
|---------|--------|-----------|
| `epta_j0613_t2_ipta_all` | `TRACK -2` in par; flat tim with `-pn` on all 1369 TOAs | `tempo2 -output add_pulseNumber` |
| `epta_j0613_t2_nrt1400` | same | same |

Provenance: fixtures use tempo2 `add_pulseNumber` for `-pn` flags and `TRACK -2` in par.
The same par/tim shape appears in some IPTA DR2 notebook exports (file source only).
Legacy per-backend files under `tims/` are retained for provenance; the manifest `tim`
path now points at the flat pn-expanded file.

**Effect on measured debt (JUG(tempo2) − libstempo):**

| Fixture | Before (`TRACK` unset, no `-pn`) | After `TRACK -2` + `-pn` (pre-fix) | After TRACK -2 (2026-07-03) | After per-TOA `-addsat` (2026-07-04) |
|---------|----------------------------------|-----------------------------------|---------------------------|-------------------------------------|
| `epta_j0613_t2_ipta_all` | RMS ~2.9 ms | RMS **~46.8 ms** (wrong anchor) | RMS **~2 µs** (scalar `-addsat`) | RMS **~608 ns**; max **~4.5 µs** |
| `epta_j0613_t2_nrt1400` | RMS ~62 ns | RMS **~62 ns** (unchanged; single backend) | **~62 ns** | **~62 ns** (unchanged) |

libstempo and tempo2 agree on `-pn` semantics. **Fixed (2026-07-03):** JUG no longer
anchors at `argsort(dt)[0]` + `base_pn + (-pn)`; it uses tempo2-style `phas1` at tim
index 0 and per-TOA `fortran_nlong`. **Fixed (2026-07-04):** per-TOA `-addsat` turn
delta with `int(F0)` vs `float(F0)` pnNew coupling closes the ~68 µs outliers on three
EFF TOAs (idx 247/256/561; CI gate **< 1 µs**). Remaining ~608 ns RMS is bulk
fractional offset across the full mix (not sub-ns gate).

---

## Gap G7 — IPTA DR2 EPTA multi-backend raw residuals — **OPEN (improved 2026-07-04)**

**Symptom:** On IPTA DR2 EPTA **J0613-0200** (`J0613-0200.par` + full TIM, 1369 TOAs),
`JUG(compatibility="tempo2")` pre-fit residuals do **not** match libstempo/tempo2 at the
sub-ns gate. Integer-turn and `-addsat` TOA debt is **fixed**; ~608 ns RMS bulk offset
remains.

Fixtures include **`TRACK -2`** and tempo2-derived **`-pn`** flags on every TOA (see
**Pulse phase / pulse-number semantics** above). Detailed EFF investigation:
[`EPTA_J0613_EFF_PARITY.md`](EPTA_J0613_EFF_PARITY.md).

**Measured on bundled fixture** `epta_j0613_t2_ipta_all`
(`tests/test_tempo2_ipta_dr2_j0613_parity.py`), same par+tim pair evaluated by both engines
with `TRACK -2` and `-pn`:

| Quantity | Gate (green fixtures) | J0613 EPTA (2026-07-04) |
|----------|----------------------|---------------------------|
| TOA count | JUG == libstempo | **1369 == 1369** |
| RMS Δ | < 5 ns | **≈ 608 ns** |
| p99 \|Δ\| | < 10 ns | **≈ 4 µs** |
| max \|Δ\| | < 25 ns | **≈ 4.5 µs** |
| WRMS Δ | < 5 ns | **~608 ns** |
| `-addsat` TOAs (idx 247/256/561) | — | **< 1 µs** each (CI gate) |

(Pre-fix with wrong TRACK -2 anchor: RMS **~46.8 ms**. Historical without pulse numbers:
RMS ~2.9 ms.)

**Verdict:** **No sub-ns parity**, but **ms-scale integer-turn debt closed**. JUG(tempo2)
is closer to libstempo on this dataset; not yet a drop-in for strict 5 ns gates.

**Tests:**
- `test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity` — standard
  `_assert_residual_parity` gate (xfail strict)
- `test_epta_j0613_ipta_dr2_track_minus2_debt_reduced` — pins measured debt in CI

---

## IPTA DR2 J0613 — per-PTA residual parity and component focus (2026-07-04)

Ad hoc JUG↔libstempo measurements on IPTA DR2 J0613 par+tim pairs. The alternate PPTA
export par/tim was copied from a notebook run for convenience (`/tmp/`); **only the
files matter for parity**, not how they were produced.

### Comparison convention

All measurements evaluate **the same par and tim files** with JUG
(`compatibility="tempo2"`, `$TEMPO2/clock`) and libstempo (`tempo2_reference`).
Any RMS gap is **JUG implementation debt vs libstempo on that par+tim pair**. Different
rows in the table below are **different par+tim inputs** to the same parity test — not
different orchestration layers.

### Measured pre-fit residual gaps (JUG − libstempo)

| Dataset (par + tim) | n TOAs | RMS Δ | max \|Δ\| | p99 \|Δ\| | Notes |
|---------------------|--------|-------|-----------|-----------|-------|
| **PPTA native** (`PPTA_dr1dr2/J0613-0200_dr1dr2.{par,tim}`) | 410 | **1.43 ns** | 5.17 ns | 4.31 ns | IPTA DR2 as-shipped; no `-pn` in tim |
| PPTA native par + `TRACK -2` + `add_pulseNumber` `-pn` tim | 410 | **1.43 ns** | 5.17 ns | 4.31 ns | Overlay on native par; debt unchanged |
| **PPTA alternate export** (IPTA DR2 export par/tim) | 410 | **15.96 ns** | 33.34 ns | 31.41 ns | See par contents below; `-pn` on all TOAs |
| **EPTA fixture** (`epta_j0613_t2_ipta_all`) | 1369 | **608 ns** | 4.5 µs | ~4 µs | G7; see [`EPTA_J0613_EFF_PARITY.md`](EPTA_J0613_EFF_PARITY.md) |

**PPTA alternate export par/tim** (dataset description only — how the files were obtained
is irrelevant to parity): 410 PPTA TOAs; par includes `TRACK -2`, `UNITS TDB`,
`CLK TT(BIPM2011)`, T2 binary parameters (`TASC`, `EPS1`, `EPS2`, …) and astrometry/DM
values copied from the IPTA DR2 EPTA reference timing solution (not the native PPTA
`PPTA_dr1dr2` numbers); tim carries tempo2 `add_pulseNumber` `-pn` flags on every TOA.

### Ablation: what changes native PPTA off ~1.4 ns?

Using the alternate export `-pn` tim throughout:

| Perturbation (start from native par) | RMS Δ |
|--------------------------------------|-------|
| Native par + native tim | **1.43 ns** |
| Native par + alternate `-pn` tim | **1.43 ns** |
| Native + `TRACK -2` only + alternate tim | **1.43 ns** |
| Individual par field → alternate-export value (`PMRA`, `CLK`, `DM`, …) one at a time | **~1.43 ns** each |
| **Full alternate-export par** + alternate tim | **15.96 ns** |

`TRACK -2`, `-pn` tim, and single-field par edits do **not** reproduce the ~16 ns gap.
The debt appears only when the **full alternate-export par** is used together with its
tim (TDB `UNITS`, EPTA-reference T2/`TASC`/`EPS1`/`EPS2`, EPTA-reference
astrometry/DM/spindown on PPTA TOAs and site context). Patching individual T2 fields
onto the native ECC/`T0`/`OM` par in isolation breaks parity catastrophically (expected
— incompatible binary parameterizations without a coherent full par).

### Structure of the alternate PPTA ~16 ns gap

On the alternate-export par/tim (same files for both engines):

- **Not integer-turn scatter:** folded RMS within one pulse period equals raw RMS (~16 ns).
- **Zero mean:** scattered fractional error, not a single global offset.
- **Moderate correlation** with JUG `roemer_shapiro_sec` (r ≈ −0.29); weaker with DM (r ≈ −0.18)
  and binary delay (r ≈ 0.05).
- JUG `pint` vs `tempo2` on the same par/tim diverges by **~14 ms** RMS
  (geometry/mean-removal split, not a libstempo comparison).

### Three parity budgets — what to focus on for ~1 ns accuracy

IPTA DR2 J0613 exposes **three separate par+tim parity budgets**. Closing one does not
automatically close the others.

#### 1. EPTA full multi-backend (~608 ns) — **dominant; fix first**

Documented in G7 and [`EPTA_J0613_EFF_PARITY.md`](EPTA_J0613_EFF_PARITY.md). Integer-turn
and per-TOA `-addsat` debt is **closed** (2026-07-04); the remaining ~608 ns is **bulk
pulse-phase / spin bookkeeping** across the multi-backend mix, not Roemer, DM, or binary
delay kernels.

**Focus:**

- `phas1` / fractional phase anchor (bulk fractional offset on many TOAs in full mix)
- Optional full `bbat` / `pnNew` path (may polish sub-µs structure; not the 608 ns driver alone)
- Fitter path: `optimized_fitter.py` still lacks `TRACK -2` / `-addsat` wiring

#### 2. Alternate PPTA export par/tim (~16 ns) — **close to target**

Native `PPTA_dr1dr2` is already at ~1.4 ns. The alternate-export pair (~16 ns) needs
refinement in JUG's tempo2 model on **that par content** (TDB, EPTA-reference parameters
on PPTA data), not in pulse-number file format alone.

**Focus:**

- `Tempo2DelayProvider` **TDB geometry** (par has `UNITS TDB`; Roemer+Shapiro shows the
  strongest correlation with Δ)
- **TZR / phase reference** on PPTA site (`TZRSITE pks`, `TZRMJD` in par)
- **T2 binary delay** at `TASC` / `EPS1` / `EPS2` in par (secondary at 16 ns scale)

#### 3. Native PPTA (~1.4 ns) — **essentially done**

No further JUG work required for parity on native `PPTA_dr1dr2` par/tim.

### Ruled out for these par+tim tests

- **Pulse-number file format alone** as the ~16 ns driver (`TRACK -2` / `-pn` overlay on
  native par leaves RMS at ~1.4 ns)
- DM and frequency conventions at ms scale on EPTA (G7 investigation)
- Different par/tim per engine (both sides always use identical paths)

### Suggested next experiments

1. Phase A term dump on alternate PPTA export par/tim with oracle Roemer/Shapiro/frequency
   terms ranked.
2. θ≠0 perturbation probes (PMRA, `TASC`, `EPS1`) on native vs alternate PPTA par/tim —
   compare JUG autodiff columns to libstempo `Mmat` on each pair.
3. Promote native PPTA (~1.4 ns) and alternate export (~16 ns) to pytest debt pins with
   checked-in or generated par/tim fixtures.

---

## Forward work themes (gap-closing)

Ordered by severity for **~1 ns residual targets** (see **IPTA DR2 J0613 — per-PTA
residual parity and component focus** above):

1. **G7 pulse-phase / spin path (EPTA ~608 ns):** `phas1` fractional anchor, optional
   `bbat`/`pnNew` polish, fitter `TRACK -2` wiring. Per-TOA `-addsat` int(F0) coupling
   **closed (2026-07-04)**. Dominant lever for the EPTA full-TIM fixture.
2. **Tempo2 TDB geometry + TZR (alternate PPTA export ~16 ns):** `Tempo2DelayProvider`
   Roemer+Shapiro under `UNITS TDB` on that par+tim pair; TZR phase reference with PPTA
   site clocks.
3. **T2 binary fine detail (alternate PPTA export par/tim):** `TASC`/`EPS1`/`EPS2` in
   par — secondary at 16 ns residual scale.
4. **θ≠0 NumPy/JAX parity (G2 residual):** diagnose ms-level mismatch on IPTA DR2 J0613
   binary/astrometry perturbations on par+tim workloads.
5. **Par+tim pytest debt pins:** native PPTA (~1.4 ns), alternate PPTA export (~16 ns),
   EPTA fixture (~608 ns).
6. **Analytic tempo2 columns (G4):** either repair or permanently deprecate; autodiff is
   the supported path.
7. **Residual debts (G6):** narrow `ppta_j1741_ell1` gap; implement or explicitly reject
   `DM_SERIES`.
8. **Documentation sync:** keep this gap list aligned with pytest gates — do not mark
   items closed from ad hoc runs alone without tests.

**Closed (2026-07-03):** G1 (NumPy reference-state). G2 primary symptom (JAX θ=0 binary
dispatch on par+tim workloads).

---

## Investigation log

### 2026-07-03 (morning)

- **G1 closed:** `get_longdouble()` for `HIGH_PRECISION_PARAMS` before zero perturbation.
  Old float64 path reproduced ~27 ns on EPTA tempo2; fixed path is machine zero.
- **G2 code landed:** unified binary dispatch in JAX forward model; JUG autodiff test
  suite added.

### 2026-07-03 (afternoon — verification pass)

- JUG autodiff: **25/25 passed**; tempo2 parity: **17 passed**, 1 intentional xfail.
- IPTA DR2 J0613: **`residual_delta_jax(0)` ≲6×10⁻¹⁴ s** (was ≈2.1 s).
- θ≠0 probe (1e-8 perturbation): ms-level NumPy/JAX mismatch on binary/astrometry params
  on real IPTA par+tim — tracked as G2 residual, not CI-gated.

### 2026-07-03 (evening — pulse numbers + phase semantics)

- Documented PINT vs tempo2 vs JUG pulse-phase paths in **Pulse phase / pulse-number
  semantics** (shared JUG phase path; G7 dominated by turn assignment).
- Updated J0613 fixtures `epta_j0613_t2_ipta_all` and `epta_j0613_t2_nrt1400` with
  `TRACK -2` and tempo2 `add_pulseNumber` `-pn` flags.
- Re-measured G7 after TRACK -2 + `-addsat` fixes: full dataset RMS **~2 µs** (was ~46.8 ms
  pre-fix); single-backend nrt1400 unchanged at ~62 ns.

### 2026-07-04 (per-PTA residual audit — PPTA par+tim pairs)

JUG↔libstempo pre-fit residual measurements on IPTA DR2 J0613 par+tim pairs. Alternate
PPTA export files were obtained from an IPTA DR2 export for convenience (`/tmp/`).

- **Native PPTA** (`PPTA_dr1dr2`, 410 TOAs): RMS **~1.4 ns**.
- **Alternate PPTA export** par/tim (EPTA-reference parameters, `TRACK -2`, `-pn`, TDB):
  RMS **~16 ns** — JUG debt on that par content, not pulse-number format alone.
- **EPTA fixture** at **~608 ns** RMS after per-TOA `-addsat` fix (was ~2 µs with scalar
  spin correction; ~68 µs max at idx 247 before int(F0) coupling).
- ~1 ns roadmap: (1) EPTA bulk fractional phase anchor, (2) TDB Roemer+Shapiro/TZR on
  alternate PPTA par, (3) T2 binary fine detail.

### 2026-07-04 (evening — per-TOA `-addsat` int(F0) coupling)

- **`_track2_addsat_turn_delta`** in `simple_calculator.py`: per-TOA `phase5` wrap after
  `-addsat`, including `int(F0)` vs `float(F0)` pnNew coupling at local fractional phase.
- EPTA fixture `epta_j0613_t2_ipta_all`: RMS **~608 ns** (was ~2055 ns); max **~4.5 µs**
  (was ~68 µs at idx 247); `-addsat` TOAs **< 1 µs** each (pytest gate).
- Commit: `fix(tempo2): per-TOA TRACK -2 -addsat int(F0) coupling for J0613 EPTA`.

---

## Related documents

| Document | Role |
|----------|------|
| [`TEMPO2_COMPATIBILITY_PROJECT.md`](TEMPO2_COMPATIBILITY_PROJECT.md) | Implementation brief, locked decisions, Cases A/B/C work plan |
| [`README.md`](README.md) | Install, compatibility modes, pytest entry points |

**When updating parity status:** if a gap closes, add a fixture + pytest gate **first**,
then remove or downgrade the row here. Ad hoc runs document status between gates but do
not substitute for CI.
