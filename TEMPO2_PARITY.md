# Tempo2 parity — status, gaps, and work queue

Living route for JUG `compatibility="tempo2"` parity: measured debt, gap scorecard,
active work queue, and investigation log.

**Policy and architecture:** [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md)

**Status (2026-07-06):** Cases A/B/C green (~1–2 ns). IPTA DR2 workloads **partially
green**. **Fix #1 (TZR, Phase C):** `epta_j0030_isolated` **15.9 → ~4.7 ns RMS** (strict
5 ns gate passes). **Fix #2 (wsrt167 TRACK −2, Phase D):** Step 1 done (pnNew); **Step 2
ruled out** — ``phase5@oracle bbat`` (~17.5 ns) is **worse** than production Taylor
(~16.4 ns). Next: WSRT ``-padd`` / ``jump_phase`` per backend; outlier TOA idx 85
(+110 ns). Native IFTE/formBats clock work is **not** the parity route. **Primary report:**
[`TEMPO2_NATIVE_CLOCK_STATUS.md`](TEMPO2_NATIVE_CLOCK_STATUS.md).

---

## 0. Diagnostic workflow

### Acceptance oracle (pytest)

Raw pre-fit residuals vs **libstempo** via `jug/testing/tempo2_reference.py`. This is
the only oracle wired into jug pytest today. Tests that require libstempo are marked
`dev_oracle` (see `jug/testing/DEV_ORACLE.md`).

```bash
cd ref-packages/jug
PYTHONPATH=. pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```

### Term-by-term debugging loop

1. Load fixture par/tim — start **`wsrt167`**, then `epta_j0613_t2_nrt1400`, then full EPTA.
2. Acceptance check: `tempo2_reference(par, tim)` vs
   `compute_residuals_simple(..., compatibility="tempo2")`.
3. Term decomposition: compare JUG `term_diagnostics` / top-level keys
   (`bbat_mjd`, `prebinary_delay_sec`, `roemer_sec`, `sw_delay_sec`, etc.) against
   libstempo properties or Phase A (`jug/testing/phase_a_comparison.py`).
4. Optional ad-hoc oracle: [`ref-packages/pytempo`](../../pytempo) `toa_diagnostics()`
   for per-TOA tempo2 `obsn[]` fields libstempo does not expose. **Not** a JUG
   dependency; **not** wired into jug test infrastructure.

   **In-repo harness (preferred for outlier work):**
   `jug/testing/tempo2_outlier_diff.py` — per-TOA clock + Roemer diff vs libstempo;
   CLI: `tools/run_tempo2_outlier_clock_roemer_diff.py`; test:
   `tests/test_tempo2_outlier_clock_roemer_diff.py`.

   ```bash
   pip install -e ref-packages/pytempo   # requires $TEMPO2 runtime; ad-hoc only
   ```

   | `bbat_mjd` | ``model_mjd − prebinary/86400`` (oracle); formBats diag **~65 s off** |
   | `roemer_sec`, `torb_sec` | `term_diagnostics` / top-level keys |
   | `phase_offset_turns` | tim ``-padd`` via ``jump_phase``; not the same as ``addPhase`` from ``pnNew`` |
   | `nphase`, `phase_turns` | ``compute_tempo2_phase5`` at pytempo ``bbat`` (Phase D harness) |
   | `pulse_number` | tim ``-pn`` (absolute-looking; **use ``pn[i]−pn[0]``** for ``pnAct``) |
   | `acceptance_residual_sec` | libstempo ``psr.residuals()`` oracle (Tier 1) |

5. Rank largest term delta → fix JUG native physics → tighten pytest debt pin → re-run.

Use raw residuals (no mean subtraction) for tempo2 acceptance gates.

Phase A term oracle: `jug/testing/tempo2_diagnostics.py` (libstempo properties only).

---

## 1. Status dashboard

### Residual parity vs libstempo (JUG − libstempo, raw pre-fit)

| Workload | n | RMS Δ | max \|Δ\| | Gate | CI |
|----------|---|-------|-----------|------|-----|
| Cases A/B/C (TCB + NG5 TDB) | 10–625 | **~1–2 ns** | < 25 ns | 5 ns | green |
| PPTA native J0613 (`PPTA_dr1dr2`) | 410 | **1.43 ns** | 5.17 ns | 5 ns | ad hoc |
| PPTA alternate export par/tim | 410 | **15.96 ns** | 33.34 ns | TBD | ad hoc |
| EPTA full (`epta_j0613_t2_ipta_all`) | 1369 | **608 ns** | 4.5 µs | 5 ns | xfail + debt pin |
| EPTA NRT1400 excerpt | 120 | **62 ns** | — | TBD | debt pin |
| **epta_j0030_isolated** | 10 | **~4.7 ns** | ~11 ns | none | **pass RMS**; p99 ~11 ns on 2×1999 TOAs |
| **wsrt167** | 167 | **~16 ns** | **~110 ns** | T2 | **fail** — WSRT padd / wrap (Phase D) |

### `-addsat` / tempo2 spin

| Item | Status |
|------|--------|
| `-addsat` regression (idx 247/256/561) | **Fixed** — each **< 1 µs** vs libstempo |
| Implicit `NE_SW = 4` cm⁻³ when par omits keyword | **Fixed (2026-07-05)** — `resolve_ne_sw_cm3()` |
| Roemer PM at POSEPOCH (not evolving `L_hat` only) | **Fixed (2026-07-05)** — `compatibility_providers.py` |
| IFTE + `formBats` clock (`tempo2_clock.py`) | **Diagnostic-only** — ~64 s formBats bat gap vs libstempo; production spin uses geometry `model_mjd` |
| Taylor spin at emission `model_mjd` (production) | **In use** — ~16 ns on wsrt167 |
| TZR post-wrap / pre-wrap / skip (`tzr_geometry.py`) | **Done (Phase C)** — J0030 15.9 → ~4.7 ns RMS |
| Native `phase5` at formBats `bbat` (`USE_NATIVE_BBAT_PHASE5`) | **Quarantined** — ~36 ns; formBats `bbat` ~65 s off pytempo |
| `track_minus2_frac_phase` pnAct | **Fixed (Phase D Step 1)** — ``pnAct = (pn[i]−pn[0]) + pnAdd``; matches legacy ``−pnAdd`` on wsrt167 |
| `compute_tempo2_phase5` at pytempo `bbat` | **Validated** — `nphase` exact; **~17.5 ns** RMS (not wired; worse than production) |
| `compute_tempo2_bbat_mjd` | **Done** — ``model − prebinary/86400`` matches libstempo/pytempo exactly |
| Longdouble clock/spin pass | **Reverted (2026-07-05)** — zero measurable benefit |
| Raw `phase5(bbat)−phase5(bbat−addsat)` wrap on legacy TRACK −2 | **Wrong** — ~67 µs at idx 247; do not use |
| `addsat_track2_turn_delta` int(F0) closure | **In use** — calibrated constants; not yet derived from `ff0` alone |
| `bbat_mjd` / `torb_sec` in JUG output | **Done** |
| BCLT iteration in `simple_calculator` | **Ruled out** — ~903 µs regression when wired |
| Fitter `TRACK -2` / `-addsat` wiring | **Open** — `optimized_fitter.py` |

Treat tempo2 mode as **experimental** outside curated par+tim tests. Do not use
`design_matrix_method="analytic"` on tempo2 sessions.

---

## 2. Gap scorecard

| Gap | Status | Summary |
|-----|--------|---------|
| **G1** NumPy `residual_delta(0) ≠ 0` | **Closed (2026-07-03)** | `get_longdouble()` for `HIGH_PRECISION_PARAMS` before zero perturbation |
| **G2** JAX autodiff at θ=0 | **Closed (2026-07-03)** | Unified `compute_total_delay_change` + `BinaryDelayPlan`; θ=0 peak ≲10⁻¹³ s |
| **G2 residual** NumPy vs JAX at θ≠0 | **Open** | ms-level mismatch on IPTA binary/astrometry perturbations; not CI-gated |
| **G4** Analytic design matrix | **Open** | Known broken; use autodiff |
| **G5** Fixture coverage | **Open** | Green on A/B/C; IPTA workloads partial (see §1) |
| **G6** Documented residual debt | **Open** | `ppta_j1741_ell1` ~5–8 ns; `DM_SERIES` warn-only |
| **G7** EPTA multi-backend | **Open (improved)** | ~608 ns bulk after integer-turn and `-addsat` fixes; wsrt167 subset ~263 ns |

**Scorecard:** 2 closed (G1, G2 primary), 5 with open items.

### G1 — closed

Historical ~27 ns peak from float64 round-trip on F0 in `residual_delta(0)`. Fixed by
reading `HIGH_PRECISION_PARAMS` via `get_longdouble()`. Touchpoint:
`jug/fitting/optimized_fitter.py`.

### G2 — closed at θ=0; residual open at θ≠0

Historical JAX binary dispatch used DD-only path (~2.1 s offset on J0613). Fixed with shared
binary dispatch. θ≠0 IPTA perturbations still show ms-level NumPy/JAX disagreement.

### G4 — analytic design matrix

Legacy analytic columns are a PINT-parity mistake on tempo2 setups. Use autodiff.

### G5 — fixture coverage

| Workload | In tempo2 CI? | Status |
|----------|---------------|--------|
| NG5 Cases B/C | Yes | Green |
| TCB Case A | Yes | Green |
| EPTA J0613 full TIM | Yes | ~608 ns RMS |
| EPTA J0613 nrt1400 | Yes | ~62 ns RMS |
| **wsrt167** (WSRT low-band) | dev_oracle | **~263 ns RMS** (debt pin) |
| PPTA native J0613 | No | ~1.4 ns ad hoc |
| PPTA alternate export | No | ~16 ns ad hoc |
| ELL1/T2 binaries | Partial | Autodiff green; `ppta_j1741_ell1` debt |

### G6 — documented residual debt

- **`ppta_j1741_ell1`:** RMS ~5–8 ns vs strict 5 ns gate; ELL1 convention mismatch.
- **`DM_SERIES`:** ignored by JUG (warn-only).

### G7 — EPTA multi-backend (~608 ns)

Integer-turn and per-TOA `-addsat` debt **closed** (2026-07-04). Remaining ~608 ns is
**bulk pulse-phase / spin bookkeeping** across the multi-backend mix — not Roemer, DM, or
binary delay kernels at ms scale.

Measured on `epta_j0613_t2_ipta_all` (`tests/test_tempo2_ipta_dr2_j0613_parity.py`):

| Quantity | Gate | EPTA (2026-07-04) |
|----------|------|-------------------|
| RMS Δ | < 5 ns | **≈ 608 ns** |
| p99 \|Δ\| | < 10 ns | **≈ 4 µs** |
| max \|Δ\| | < 25 ns | **≈ 4.5 µs** |
| `-addsat` TOAs (idx 247/256/561) | — | **< 1 µs** each |

Tests: `test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity` (xfail strict);
`test_epta_j0613_ipta_dr2_track_minus2_debt_reduced` (debt pin).

---

## 3. Active work queue

| Priority | Task | Oracle / fields | Status |
|----------|------|-----------------|--------|
| **1** | **WSRT ``-padd`` / ``jump_phase``** per backend (``.C`` vs non-``.C``) | tim ``-padd``; `jump_phase`; per-``-sys`` means | **Open** — ~10 ns inter-sys split on wsrt167 |
| **2** | **Outlier TOA idx 85** (+110 ns) | pytempo ``phase_turns`` / ``nphase``; wrap ladder | **Open** — see § Phase D idx 85 |
| **3** | Close **wsrt167** to <5 ns gate | `test_dev_oracle_wsrt167_parity.py` | **Open** — production ~16.4 ns |
| **4** | Polish **epta_j0030** p99 (~11 ns on 2×1999 TOAs) | outlier harness; early-epoch astrometry | **Open** — RMS gate passes after Phase C TZR |
| **5** | Update BIPM clock files for `epta_j0613_t2_ipta_all` | clock-file coverage to MJD 56795 | **Open** — data, not algorithm |
| **6** | Validate on `epta_j0613_t2_nrt1400` (~6 ns) | libstempo + term diagnostics | **Open** |
| **7** | Alternate PPTA ~16 ns | Roemer/Shapiro + TZR at `TZRMJD` | **Open** |
| **8** | Fitter TRACK −2 / `-addsat` wiring | after subset gates pass | **Open** |
| **Done** | **Phase C — TZR** (fix #1) | `tests/test_tempo2_tzr_parity.py`; `tzr_geometry.py` | **Done** — J0030 15.9 → ~4.7 ns RMS |
| **Done** | **Phase D Step 1 — pnNew convention** | `tests/test_tempo2_track2_pnnew.py` | **Done** — relative ``-pn`` |
| **Ruled out** | **Phase D Step 2 — wire ``phase5@bbat`` to production** | `tempo2_track2_oracle.py` | **~17.5 ns** — worse than Taylor production |
| **Defer** | formBats ``bbat`` diagnostic fix | ~65 s off oracle | — |

---

## Phase C — TZR reference phase (fix #1, 2026-07-05)

**Goal:** close **epta_j0030_isolated** (~15.9 ns RMS dominated by 2×1999 TOAs far from
``TZRMJD``).

**Status:** **Done** in production tempo2 path.

### Root cause

JUG subtracted ``tzr_phase`` in **turn space before** phase wrapping. Tempo2
``REFPHS TZR`` inserts a synthetic TZR observation and subtracts its **wrapped residual**
after ``phas1`` / ``nlong``. The error grows with distance from ``TZRMJD`` (~4010 days for
the 1999 outlier TOAs vs ≤773 days on passing fixtures).

### Fix

``jug/residuals/tzr_geometry.py`` — ``resolve_tempo2_tzr_apply_mode()``:

| Mode | When | Behaviour |
|------|------|-----------|
| `pre_wrap` | TOAs within ~2000 days of ``TZRMJD`` | legacy pre-wrap ``tzr_phase`` (ng5 near-TZR) |
| `post_wrap` | explicit ``REFPHS TZR`` | subtract wrapped TZR residual after phase5 |
| `none` | TOAs far from ``TZRMJD`` | skip TZR shift (matches libstempo default for J0030) |

### Measured (J0030 vs libstempo)

| Metric | Before | After Phase C |
|--------|--------|---------------|
| RMS | 15.9 ns | **~4.7 ns** |
| max \|Δ\| | ~38 ns | **~11 ns** |
| Outlier TOAs 0–1 | ~32–38 ns | **~7–11 ns** |

Tests: ``tests/test_tempo2_tzr_parity.py``. ng5 / binary parity unchanged.

---

## Phase D — TRACK −2 pnNew (fix #2, 2026-07-05)

**Goal:** close **wsrt167** (~16 ns RMS, max ~110 ns) without re-enabling the quarantined
formBats / ``USE_NATIVE_BBAT_PHASE5`` stack.

**Status:** **Step 1 done** (pnNew convention + oracle tests). **Step 2 ruled out**
(2026-07-06). **Next:** WSRT ``-padd`` / ``jump_phase`` per backend; outlier idx 85.

### What we found (Step 1)

1. **tim ``-pn`` is not absolute.** On IPTA DR2 exports (wsrt167), ``-pn[i] − -pn[0]`` equals
   tempo2 ``pnNew`` (after ``pn0`` anchoring at obsn[0]). JUG was using raw ``-pn`` in
   ``pnAct``, producing ``addPhase ~ 10¹⁰`` turns when combined with ``phase5``.
2. **With relative ``-pn``, ``addPhase = −pnAdd``** (+1 turn on wsrt167 where every TOA has
   ``pnAdd = −1``). That matches the legacy production wrap (`add_phase = −pn_add`).
3. **`compute_tempo2_phase5` at pytempo ``bbat`` is validated** (`nphase` exact vs tempo2).

### Step 2 investigation (2026-07-06) — ruled out

Temp-only path ranking on wsrt167 vs libstempo:

| Path | RMS Δ | Notes |
|------|-------|-------|
| Production Taylor@``model_mjd`` + legacy wrap | **16.4 ns** | **Best JUG-composed path** |
| Oracle ``phase5@bbat`` + ``track_minus2_frac_phase`` | **17.5 ns** | Step 2 target — **worse** |
| ``USE_NATIVE_BBAT_PHASE5`` + formBats ``bbat`` | ~711 ms | formBats ``bbat`` ~65 s wrong |
| pytempo ``acceptance_residual_sec`` | **0 ns** | tempo2 internal path exact |

**Oracle ``bbat`` identity (from existing JUG delay terms):**

```
bbat = model_mjd − prebinary_delay_sec / 86400
```

Matches libstempo ``toas`` / pytempo ``bbat_mjd`` to **0 s RMS**. JUG formBats
``term_diagnostics['bbat_mjd']`` remains **~65 s wrong** — diagnostic only.

**``torb`` for ``phase5``:** use ``compute_tempo2_torb_sec(bbat, dt_sec, PEPOCH)`` only.
Raw ``-prebinary_delay_sec`` or pytempo ``torb_sec`` arrays mis-paired with ``dt_sec``
trigger discrete ``nphase`` / ``pnNew`` wrap errors (~172 ns).

**Inter-backend structure (WSRT ``-sys``):**

| ``-sys`` | n | mean Δ | RMS Δ |
|----------|---|--------|-------|
| WSRT.P1.328 | 29 | +10.4 ns | 25.2 ns |
| WSRT.P1.328.C | 58 | +2.8 ns | 12.2 ns |
| WSRT.P1.382 | 24 | −0.6 ns | 16.0 ns |
| WSRT.P1.382.C | 56 | −8.1 ns | 14.5 ns |

Per-``-sys`` mean removal → **~15 ns** floor. ``jump_phase`` / tim ``-padd`` correlates
with Δ at **r ≈ −0.34** — **needs dedicated investigation**.

Clock/Roemer/sat ruled out (roemer **~0.8 ns**, sat **0 ns** on harness).

### Outlier TOA idx 85 (+110 ns max)

| Field | Value |
|-------|-------|
| Index | 85 / 167 |
| ``-sys`` | WSRT.P1.328 @ 328 MHz |
| Production Δ | **+110.5 ns** |
| tim ``-padd`` | 0.599804 (same as neighbours idx 83–87) |
| pytempo acceptance | **exact** vs libstempo (−10.395112 µs) |

**Not explained by:** clock/Roemer/sat/bbat diagnostic; padd alone (neighbours share
``padd=0.5998`` with 2–23 ns errors). Gap is **~0.000036 turns** (~110 ns) in production
Taylor fractional wrap.

**Wrap ladder mismatch (idx 84→85):** JUG ``fortran_nlong`` Δnphase = **3776867209**
(tracks ``-pn`` Δ); pytempo ``nphase`` Δ = **2569609**. Tempo2 and JUG use different
``nphase`` ladders at large ``|pn|``; residual gap concentrates at wrap boundaries.

**Do not wire ``phase5@bbat`` to production** for wsrt167 — no RMS gain. Next drill:
WSRT ``-padd`` / ``jump_phase`` per backend, then ``nphase`` ladder vs tempo2 at outliers.

### Code / tests

| Artifact | Role |
|----------|------|
| `jug/residuals/tempo2_spin.py` | ``compute_tempo2_bbat_mjd``; ``track_minus2_frac_phase`` |
| `jug/testing/tempo2_track2_oracle.py` | pytempo-backed TRACK −2 harness |
| `tests/test_tempo2_track2_pnnew.py` | pnNew identity; documents ~17.5 ns oracle floor |

```bash
cd ref-packages/jug
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_track2_pnnew.py -q
```

### WSRT167 (2026-07-05)

167-TOA WSRT low-band subset (328/382 MHz) from `epta_j0613_t2_ipta_all`.
Fixture: `tests/data_tempo2/wsrt167/`. Debt pin:
`tests/test_dev_oracle_wsrt167_parity.py`.

| Measurement | Before `32dd71a` | After `32dd71a` |
|-------------|------------------|-----------------|
| RMS vs libstempo | **~1056 ns** | **~263 ns** |
| max \|Δ\| | **~3233 ns** | **~550 ns** |
| Mean Δ | ~0 ns | \|mean\| < 50 ns |

**Root cause closed in `32dd71a`:** tempo2 uses implicit **`NE_SW = 4` cm⁻³**
(`NE_SW_DEFAULT` in `tempo2/initialise.C`) even when the par omits `NE_SW`. JUG had
default 0 → missing solar-wind `tdis2` in prebinary → wrong `bbat` → spin scatter.
Also fixed: Roemer PM at POSEPOCH. Later IFTE + `formBats` (`tempo2_clock.py`)
reduced wsrt167 to **~16 ns RMS** (still above the 5 ns gate); see
`TEMPO2_NATIVE_CLOCK_STATUS.md`.

Decomposition vs libstempo on the same 167 TOAs (raw pre-fit, unweighted mean):

| Check | RMS / correlation | Notes |
|-------|-------------------|-------|
| Residual RMS (JUG − libstempo) | **263 ns** | max ≈ 550 ns; mean ≈ 0 |
| Intra-`-sys` scatter | **≈ 263 ns** | Not a 328/382 MHz group-mean split (group means differ ≲ 40 ns) |
| `torb` vs `−binarydelay` | **0.17 ns** | Binary sign convention closed |
| `sun_shapiro` | **0 ns** | Sun Shapiro matches |
| `roemer` property | sign flip only | libstempo `roemer` = −JUG `roemer_sec`; combined Roemer+Shapiro path OK |
| JUG `bbat` vs libstempo `toas` | **330 ns** | corr(residual) ≈ 0.84 — timing, not phase-wrap |
| JUG `bbat` vs `pets − torb/86400` | **370 ns** | `pets` = tempo2 pulse-emission epoch; corr ≈ 0.79 |
| Oracle `bbat` from `pets` | **222 ns** residual | libstempo-only; shows native `bbat` formation gap |
| `tempo2_spin=True` + legacy TRACK −2 | catastrophic | Do not enable — switches to `pnNew` wrap |
| `compute_tempo2_phase5` + legacy TRACK −2 | **≈ 264 ns** | Same as Taylor when `torb` sign = `−jug_torb` |
| Weighted mean subtraction | **882 ns** | tempo2 uses unweighted (`phase_mean_mode=unweighted`) ✓ |

**Leading hypothesis:** native **`bbat = model − prebinary/86400`** differs from tempo2
`obsn.bbat` by **~300–370 ns RMS** (not float64 ULP — longdouble path unchanged). This maps
to **~220–260 ns** prefit residual scatter after TRACK −2 spin. Likely a **clock/`sat` vs
`model` split** or missing term in prebinary composition relative to tempo2 `formBats.C`
(`batCorr = clock + (tt_tb − tropo + roemer − shap − tdis)`; `bbat = bat − shklovskii/86400`).

**Ruled out this session:** Taylor-vs-`phase2` spin at bbat (same debt with correct `torb`
sign); BCLT iteration; `NE_SW`/Roemer-PM regressions; inter-band group anchor.

**Next native fix targets:** port tempo2 `bbat`/`sat`/`getCorrectionTT` split (site-clock
component ≈ 185 ns seen in pre-fix notes); verify Shklovskii subtraction on `bbat`; compare
`prebinary` term sum to tempo2 `batCorr` + `tdis` identity without libstempo.

**Historical ad-hoc notes (pre-fix, same TOAs in full 1369-TOA mix):**

- Mean Δ **+1.287 ns** uniform shift when same wsrt167 TOAs embedded in full mix
- **`-padd` / `jump_phase`:** required; without it RMS → ~6 ns
- **328 vs 382 bands:** ~0.4 ns group split; TRACK −2 / `-pn` interaction

---

## 4. J0613 three parity budgets

Closing one budget does not automatically close the others.

### 1. EPTA full multi-backend (~608 ns) — dominant

Integer-turn and `-addsat` debt closed. Remaining bulk is pulse-phase / spin bookkeeping
across multi-backend mix. Decompose via wsrt167 (~263 ns today) + mean-anchor shifts.

**Focus:** close wsrt167 (priority 1); then `-padd` vs TRACK −2 placement; full-mix mean
subtraction anchor; nrt1400 (~62 ns) before touching full mix.

### 2. Alternate PPTA export (~16 ns)

Native PPTA already ~1.4 ns. ~16 ns appears only with **full alternate-export par** +
tim (TDB, EPTA-reference T2/astrometry on PPTA TOAs) — not from `-pn` format alone.

**Focus:** TDB Roemer+Shapiro; TZR at `TZRMJD` / `TZRSITE pks`.

### 3. Native PPTA (~1.4 ns) — essentially done

No further work required for parity on native `PPTA_dr1dr2` par/tim.

### Ablation summary (alternate export)

| Perturbation | RMS Δ |
|--------------|-------|
| Native par + native tim | **1.43 ns** |
| Full alternate-export par + alternate tim | **15.96 ns** |

Single-field par edits do not reproduce the ~16 ns gap.

---

## 5. Phase and TRACK −2 semantics

### Comparison table

| Aspect | PINT | tempo2 | JUG (both modes) |
|--------|------|--------|------------------|
| Default reference | Index 0 or TZRMJD | tim index 0 (`obsn[0]`) | Sequential by `dt` (no TRACK −2) |
| `TRACK -2` | `model.phase(abs) − pn` | `pnNew` vs `-pn`; `phas1` at tim index 0 | Production: legacy ``−pnAdd``; Phase D: ``pnAct = (pn[i]−pn[0])+pnAdd`` |
| Mean removal | Weighted | Unweighted | Mode-dependent |
| Delay path split? | — | — | **Yes** (providers differ) |
| Phase path split? | — | — | **No** (shared `compute_phase_residuals`) |

### tempo2 `formResiduals.C` order (TRACK −2)

1. Compute spin phase at `bbat` → `phase5`
2. Add tim **`-padd`** to `phase5`
3. `phas1 = fortran_mod(phase5[tim_index_0], 1.0)`; subtract from all TOAs
4. `nphase = fortran_nlong(phase5[i])`; fractional residual = `phase5 − nphase`
5. Optional `addPhase` from `pnNew` vs `-pn` flags

### JUG shipping path (TRACK −2, tempo2 mode)

**Production:** emission-time Taylor spin at geometry **`model_mjd`** → add
`jump_phase` (includes tim **`-padd`** / **`-radd`**) → `phas1@tim[0]` → `fortran_nlong`
per TOA → legacy `add_phase = −pn_add` (equivalent to fixed ``pnNew`` when
``pn[i]−pn[0] == pnNew``).

**Phase D (not production):** ``compute_tempo2_phase5`` at oracle ``bbat`` +
fixed ``track_minus2_frac_phase`` → **~17.5 ns** RMS on wsrt167 (validated in
``tests/test_tempo2_track2_pnnew.py``) — **worse than production**.

**Quarantined / do not enable:** ``USE_NATIVE_BBAT_PHASE5`` — oracle harness only;
formBats ``bbat_mjd`` diagnostic remains **~65 s** off tempo2.

Key touchpoints: `jug/residuals/tempo2_spin.py`,
`jug/residuals/tempo2_native_quarantine.py`, `jug/residuals/simple_calculator.py`.

### Subset pitfall

tim ``-pn`` values are written as **pulse-number offsets relative to full-tim obsn[0]**:
``pn[i] − pn[0]`` equals tempo2 ``pnNew`` (after ``pn0`` anchoring). Raw ``-pn`` in
``pnAct`` produces ``addPhase ~ 10¹⁰`` turns. Isolated sub-tims without regenerating
``-pn`` also change obsn[0] semantics — prefer full-tim oracle pull + mask.

---

## 6. J0613 EPTA investigation summary

Condensed from the former `EPTA_J0613_EFF_PARITY.md` investigation record.

### Fixture

| Field | Value |
|-------|-------|
| ID | `epta_j0613_t2_ipta_all` |
| Par/tim | `tests/data_tempo2/epta_j0613_t2_ipta_all/` |
| TOAs | 1369 (flat tim; `TRACK -2`; `-pn` on all TOAs via `add_pulseNumber`) |
| Sibling | `epta_j0613_t2_nrt1400` (120 TOAs, NRT.BON.1400 excerpt) |
| Unit test | `wsrt167` — 167 WSRT TOAs, freq < 1000 MHz |
| PSRJ | J0613-0200; `BINARY T2`; `F0` ≈ 326.6 Hz |

First tim index (`obsn[0]`): JBO.DFB.1400, MJD ≈ 54847, `-pn 0`. Earliest emission time
is **not** index 0 (WSRT ≈ MJD 52958).

### Historical parity scale

| Configuration | RMS (JUG − libstempo) |
|---------------|----------------------|
| No TRACK −2, no `-pn` | ~2.9 ms |
| TRACK −2 + `-pn`, wrong anchor (pre-fix) | ~46.8 ms |
| TRACK −2 fix (2026-07-03) | ~57 µs → ~2 µs with scalar `-addsat` |
| Per-TOA `-addsat` fix (2026-07-04) | **~608 ns** bulk; `-addsat` TOAs < 1 µs |
| wsrt167 NE_SW + bbat spin (2026-07-05) | **~1056 ns → ~263 ns** on wsrt167 subset |

### What was fixed

1. **TRACK −2 anchor (2026-07-03):** JUG uses `phas1@tim index 0` + per-TOA
   `fortran_nlong`, not `argsort(dt)[0]` + `base_pn + (-pn)`.
2. **±326-turn EFF TOAs (2026-07-03):** eight TOAs on EFF.EBPP.1360/2639 with integer
   errors of ±(int)F0 turns — closed by TRACK −2 fix.
3. **Per-TOA `-addsat` (2026-07-04):** `addsat_track2_turn_delta` closes idx 247/256/561
   (~68 µs → < 1 µs).
4. **Implicit NE_SW + bbat spin (2026-07-05):** `resolve_ne_sw_cm3()` (default 4 cm⁻³),
   Roemer PM at POSEPOCH, Taylor spin at `bbat`; wsrt167 **~1056 ns → ~263 ns**.

### What was ruled out

- Roemer/DM/binary kernels as ms-scale drivers on full mix (post TRACK −2 fix).
- Clock dir JUG vs `$TEMPO2/clock` swap — no RMS change.
- BIPM2011/2019/2024 CLK sweep — no change.
- Full `pnNew`/`bbat` rewrite without correct spin decomposition — catastrophic (~10¹¹ ms).
- BCLT iteration wired into `simple_calculator` — ~903 µs regression vs libstempo.

### EFF backend notes

| `-sys` | n | Isolated RMS (TRACK −2) |
|--------|---|-------------------------|
| EFF.EBPP.1410 | 241 | **~69 ns** |
| EFF.EBPP.1360 | 42 | was ±326-turn outliers; now sub-µs on bad TOAs |
| EFF.EBPP.2639 | 64 | same |
| JBO / NRT / WSRT (isolated) | — | sub-µs to ~1 µs (pre wsrt167 fix); wsrt167 now ~263 ns |

Non-EFF backends match libstempo in isolation but show ~0.73 ms offsets in the **full mix**
when EFF 1360/2639 present — cross-backend `phas1`/anchor coupling (historical).

---

## 7. Other open items

- **G4:** deprecate or repair analytic tempo2 columns; autodiff is supported path.
- **G6:** narrow `ppta_j1741_ell1`; implement or reject `DM_SERIES`.
- **G2 residual:** θ≠0 NumPy/JAX on IPTA workloads.
- **pytest debt pins:** wsrt167 (~263 ns, dev_oracle); native PPTA (~1.4 ns) and alternate
  PPTA (~16 ns) still ad hoc only.

---

## 8. Investigation log

| Date | Summary |
|------|---------|
| 2026-06-01 | Phases A–E: native TDB geometry, TZR, Cases B/C green, design-matrix/fit parity |
| 2026-07-03 | G1/G2 primary closed; J0613 fixtures get TRACK −2 + `-pn`; TRACK −2 anchor fix (~46.8 ms → ~2 µs) |
| 2026-07-04 | Per-TOA `-addsat` int(F0) coupling; EPTA RMS **~608 ns**; PPTA native ~1.4 ns, alternate ~16 ns measured |
| 2026-07-04 | `tempo2_spin=True` attempt wrong for legacy stack; production restored via `addsat_track2_turn_delta` |
| 2026-07-05 | Doc consolidation: three parity markdown files merged into this doc + [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md) |
| 2026-07-05 | **wsrt167 promoted:** fixture under `tests/data_tempo2/wsrt167/`; dev-oracle debt pin; implicit **NE_SW=4**, Roemer PM at POSEPOCH, spin at **bbat** — **~1056 ns → ~263 ns RMS** (`32dd71a`) |
| 2026-07-05 | **Parity review:** longdouble reverted; native phase5 quarantined; outlier harness added |
| 2026-07-05 | **Phase C (fix #1):** TZR apply modes in ``tzr_geometry.py`` — J0030 **15.9 → ~4.7 ns RMS** |
| 2026-07-05 | **Phase D Step 1 (fix #2):** tim ``-pn`` relative to obsn[0]; ``tempo2_track2_oracle.py`` + ``test_tempo2_track2_pnnew.py`` |
| 2026-07-06 | **Phase D Step 2 ruled out:** ``phase5@oracle bbat`` ~17.5 ns vs production ~16.4 ns; ``compute_tempo2_bbat_mjd`` identity; WSRT padd/jump_phase split open; idx 85 outlier (+110 ns) documented |

**When updating parity status:** add a fixture + pytest gate **first**, then update this log.
Ad hoc runs document status between gates but do not substitute for CI.

---

## 9. Commands and tests

```bash
# JUG tempo2 parity (acceptance oracle = libstempo)
cd ref-packages/jug

# wsrt167 debt pin (dev oracle — requires libstempo + $TEMPO2)
PYTHONPATH=. pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q

# Outlier clock / Roemer harness
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_outlier_clock_roemer_diff.py -m dev_oracle -q

# TRACK −2 pnNew harness (Phase D Step 1)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_track2_pnnew.py -q

# TZR parity (Phase C)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_tzr_parity.py -q

# All tempo2 oracle tests
JUG_TEST_TEMPO2=1 PYTHONPATH=. pytest tests/test_tempo2_*.py -m dev_oracle -q

# JUG-only CI path (no libstempo)
PYTHONPATH=. pytest -m 'not dev_oracle' -q

# J0613 EPTA debt pins
PYTHONPATH=. pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -q

# Phase A term ranking (libstempo oracle)
python tools/run_phase_a_diagnostics.py --output /tmp/phase_a_report.json

# Optional ad-hoc pytempo (NOT in jug CI)
cd ref-packages/pytempo && pip install -e . && python -m pytest tests -q
```

Key modules:

| Module | Role |
|--------|------|
| `jug/residuals/simple_calculator.py` | JUG residuals under test |
| `jug/residuals/tempo2_spin.py` | ``compute_tempo2_bbat_mjd``; ``phase5`` / TRACK −2; pnAct relative to obsn[0] |
| `jug/residuals/diagnostic_conventions.py` | `resolve_ne_sw_cm3()`, conventions |
| `jug/residuals/compatibility_providers.py` | Tempo2 TDB delay provider |
| `jug/residuals/tzr_geometry.py` | TZR apply modes (Phase C) |
| `jug/testing/tempo2_track2_oracle.py` | TRACK −2 pnNew / ``phase5@bbat`` oracle (Phase D) |
| `jug/testing/tempo2_outlier_diff.py` | Per-TOA clock + Roemer diff vs libstempo (outlier work) |
| `jug/testing/tempo2_reference.py` | libstempo acceptance oracle |
| `jug/testing/tempo2_diagnostics.py` | Phase A term oracle (libstempo properties) |
| `jug/testing/phase_a_comparison.py` | Phase A ranking |
| `jug/testing/DEV_ORACLE.md` | Delete-when-standalone checklist |
| `ref-packages/pytempo` | Optional ad-hoc per-TOA diagnostic oracle (external) |

Tempo2 reference source (debug only): `ref-packages/tempo2/` — `formBats.C`,
`calculate_bclt.C`, `formResiduals.C`, `initialise.C` (`NE_SW_DEFAULT`).
