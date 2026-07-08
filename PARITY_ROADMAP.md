# Tempo2 parity — status, roadmap, and investigation log

Living tracker for JUG `compatibility="tempo2"` parity: measured debt, gap scorecard,
active work queue, and condensed investigation history.

**Theory, policy, and definitions:** [`PARITY_THEORY.md`](PARITY_THEORY.md)

*Last updated: 2026-07-08*

---

## Executive summary (2026-07-08)

Three investigations are complete: failing tests, JAX paths, and derivatives.

**Combined conclusion:** the main parity path is to make tempo2 autodiff / native
`residual_delta_jax` the **canonical tangent for fitting**, then test it against
libstempo two-parameter perturbation oracles across `staged_bclt`,
`fixed_state_nonlinear`, and `full`. The largest real residual blocker remains
**NG5 TDB's ~5.3 µs spin-epoch / TDB-TCB map issue**. Several other failures are
either stale dev-oracle assertions or unrelated hygiene bugs (`data_dir`, DM noise
convention).

**Recent wins (2026-07-07):**

- Two-part `(int_day, sec_in_day)` barycentric time wired through all three graph modes.
- `-addsat` read-time SAT shift with `mjd_str` resync (was ~±1 s; now < 1 µs).
- wsrt167 host Taylor spin: **~1.4 ns RMS** after tropo-in-dt + longdouble-wrap fixes.
- DE440 ephemeris offline resolution fixed (3 design-matrix tests unblocked).

---

## 1. Status dashboard

### Residual parity vs libstempo (JUG − libstempo, raw pre-fit)

| Workload | n | RMS Δ | max \|Δ\| | Gate | CI |
|----------|---|-------|-----------|------|-----|
| Case A (TCB) | 10 | **~1.3 ns** | ~3.2 ns | 5 ns | green |
| **NG5 Case B** (equatorial TDB) | 625 | **~5.3 µs** | ~8.2 µs | 5 ns | **fail** — spin-epoch / TDB-TCB map |
| **NG5 Case C** (ecliptic cross-engine) | 625 | **~5.3 µs** | ~8.2 µs | 5 ns | **fail** — same class as Case B |
| epta_j0030 | 10 | **1.28 ns** | 3.17 ns | 5 ns | green |
| epta_j1909 | 27 | **1.59 ns** | 3.60 ns | 5 ns | green |
| epta_j1918 | 12 | **1.31 ns** | 2.02 ns | 5 ns | green |
| ppta_j1741 | 111 | **5.27 ns** | 11.38 ns | 5 ns | close |
| ppta_j1902 | 120 | **2.41 ns** | 6.54 ns | 5 ns | green |
| epta_j0613_nrt1400 | 120 | **1.21 ns** | 4.07 ns | 5 ns | green |
| **wsrt167** | 167 | **1.43 ns** | 4.85 ns | 2.5 ns | **pass** |
| epta_j0613_addsat_min | 11 | **84.5 ns** | 173.7 ns | 1 µs | dt-chain scatter on addsat TOAs |
| **epta_j0613_ipta_all** | 1369 | **10.08 ns** | 217.9 ns | 5 ns | xfail strict |
| PPTA native J0613 (`PPTA_dr1dr2`) | 410 | **1.43 ns** | 5.17 ns | 5 ns | ad hoc |
| PPTA alternate export par/tim | 410 | **15.96 ns** | 33.34 ns | TBD | ad hoc |

> **NG5 reconciliation:** older docs called Cases B/C "green ~1.3 ns." The 2026-07-08
> re-baseline and native-JAX-path measurement show **~5.3 µs RMS** on both NG5 TDB fixtures.
> The mismatch is 100 % correlated with `dt_jug − deltaT(pytempo)` (r = 1.0); clock
> `correction_tt` matches pytempo to ~0.004 ns — ruled out. Residual shows ~1.7 µs/yr
> slope vs MJD — consistent with TDB-grid spin vs libstempo's tempo2-internal TCB/IFTE
> epoch map. **F0 design-matrix column passes** (`atol=0.02`) because the mismatch is a
> **spin-epoch / prefit offset**, not a derivative error.

### J0613 fast gates (2026-07-07)

| Gate | File | Measured debt | Pin |
|------|------|---------------|-----|
| No TRACK / no `-pn` | `tests/test_tempo2_j0613_fast_gates.py` | nrt1400 ~4.4 ns RMS | 100 ns |
| TRACK −2 `-addsat` mini | `epta_j0613_addsat_min` (11 TOAs) | bulk ~86 ns RMS; addsat TOAs ~172 ns max | 1 µs |
| wsrt167 bulk spin | `test_dev_oracle_wsrt167_parity.py` | **~1.4 ns** RMS | 2.5 ns |
| Full EPTA | `test_tempo2_ipta_dr2_j0613_parity.py` (xfail) | ~10 ns RMS | 5 ns |

### Production host routing (`pipeline.finalize_tempo2_host_residuals`)

| Condition | Route | Notes |
|-----------|-------|-------|
| `TRACK` absent | Taylor `compute_phase_residuals` (sequential) | nrt1400 ~4 ns |
| `TRACK == -2` | Taylor emission-time + legacy `-pn` wrap | libstempo parity; Taylor is **better-conditioned** than `phase5@bbat` here |
| other `TRACK` | `compute_native_eval_residuals_jax` | two-part formBats tail |

**JAX fit/autodiff** (all TRACK values): native two-part tail via
`JUG_TEMPO2_NATIVE_GRAPH_MODE` (default `staged_bclt`). Residual Jacobian uses
`phase5@bbat`.

### `-addsat` / tempo2 spin

| Item | Status |
|------|--------|
| `-addsat` regression (idx 247/256/561) | **Fixed** — each **< 1 µs** vs libstempo |
| Implicit `NE_SW = 4` cm⁻³ when par omits keyword | **Fixed (2026-07-05)** |
| Roemer PM at POSEPOCH | **Fixed (2026-07-05)** |
| IFTE + `formBats` clock (`tempo2_clock.py`) | **Diagnostic-only** — production spin uses geometry `model_mjd` |
| Taylor spin at emission `model_mjd` (production) | **In use** — ~1.4 ns on wsrt167 |
| TZR post-wrap / pre-wrap / skip (`tzr_geometry.py`) | **Done (Phase C)** — J0030 15.9 → ~4.7 ns RMS |
| Native `phase5` at formBats `bbat` (`USE_NATIVE_BBAT_PHASE5`) | **Quarantined** — ~36 ns; do not wire to production |
| `track_minus2_frac_phase` pnAct | **Fixed (Phase D Step 1)** — `pnAct = (pn[i]−pn[0]) + pnAdd` |
| Longdouble clock/spin pass | **Reverted (2026-07-05)** — zero measurable benefit |
| BCLT iteration in `simple_calculator` | **Ruled out** — ~903 µs regression when wired |
| Fitter `TRACK -2` / `-addsat` wiring | **Open** — `optimized_fitter.py` |

Treat tempo2 mode as **experimental** outside curated par+tim tests. Do not use
`design_matrix_method="analytic"` on tempo2 sessions.

---

## 2. Canonical roadmap

The path to production-ready tempo2 parity for MetaPulsar / IPTA workloads:

### Phase 1 — Canonical tangent (current priority)

- [ ] Make tempo2 autodiff / native `residual_delta_jax` the **canonical tangent** for
  all tempo2 fitting (`design_matrix_method="autodiff"`).
- [ ] Validate residual deltas against **libstempo two-parameter perturbation oracles**
  across all three graph modes:
  - [ ] `staged_bclt` (default production)
  - [ ] `fixed_state_nonlinear` (fast NUTS path)
  - [ ] `full` (oracle/dev only)
- [ ] Envelope test: `fixed_state_nonlinear` vs `staged_bclt` for PTA-scale perturbations
  (target: max |Δ| < 1 ns on wsrt167-class fixtures).

### Phase 2 — Host residual closure

- [ ] Close **NG5 TDB ~5.3 µs** spin-epoch / TDB-TCB map (top blocker).
  - Proposed fix: route TDB pars through tempo2-native `formBats` + `formResiduals` spin
    at `bbat` (or equivalent TDB→TCB epoch map before spin), not production
    `model_mjd=tdb_mjd` Taylor.
- [ ] Close **EPTA J0613 IPTA full ~10 ns** floor.
  - Requires aligning production spin with tempo2 `calculate_bclt` + `formBats` epoch
    (JAX native chain track), not clkcorr feedback on merged chains.
- [ ] Close **epta_j0613_addsat_min ~84 ns** dt-chain scatter on/near `-addsat` TOAs.

### Phase 3 — Hygiene and test debt

- [ ] Fix `data_dir` hygiene bug (called out in latest analysis).
- [ ] Fix DM noise convention mismatch (called out in latest analysis).
- [ ] Audit and retire **stale dev-oracle assertions** (several failures are assertion
  debt, not physics gaps).
- [ ] Update BIPM clock files for `epta_j0613_t2_ipta_all` (data coverage, not algorithm).

### Phase 4 — Polish and defer

- [ ] Polish epta_j0030 p99 (~11 ns on 2×1999 TOAs).
- [ ] Alternate PPTA ~16 ns export budget.
- [ ] Fitter TRACK −2 / `-addsat` wiring in `optimized_fitter.py`.
- [ ] `DMASSPLANET` reflex correction (G8 — deferred, no fixture coverage).
- [ ] Full `get_obsCoord` port (G9 — deferred, sub-cm already on wsrt167).

---

## 3. Blocker ledger

| ID | Blocker | Severity | Status | Proposed fix |
|----|---------|----------|--------|--------------|
| **B1** | NG5 TDB ~5.3 µs spin-epoch / TDB-TCB map | **Critical** | Open | Route TDB pars through tempo2-native formBats/formResiduals spin at `bbat` |
| **B2** | EPTA J0613 IPTA full ~10 ns floor | High | Open | Native BCLT/formBats epoch alignment via JAX chain |
| **B3** | epta_j0613_addsat_min ~84 ns dt-chain scatter | Medium | Open | Site-epoch handling on/near `-addsat` TOAs |
| **B4** | `data_dir` hygiene bug | Medium | Open | Fix path resolution |
| **B5** | DM noise convention mismatch | Medium | Open | Align convention with tempo2 |
| **B6** | Stale dev-oracle assertions | Low | Open | Audit/retire |
| **B7** | BIPM2011 clock extrapolation (J0613 to ~56796) | Low | Data | Shared JUG/libstempo constant extrapolation — not JUG-only |

---

## 4. Gap scorecard

| Gap | Status | Summary |
|-----|--------|---------|
| **G1** NumPy `residual_delta(0) ≠ 0` | **Closed (2026-07-03)** | `get_longdouble()` for `HIGH_PRECISION_PARAMS` before zero perturbation |
| **G2** JAX autodiff at θ=0 | **Closed (2026-07-03)** | Unified `compute_total_delay_change` + `BinaryDelayPlan`; θ=0 peak ≲10⁻¹³ s |
| **G2 residual** NumPy vs JAX at θ≠0 | **Unverified** | ms-level claim not CI-gated; may conflate JUG-vs-libstempo gaps with internal parity. Treat as scorecard debt until reproducing evidence exists. |
| **G4** Analytic design matrix | **Open** | Known broken; use autodiff |
| **G5** Fixture coverage | **Open** | Green on Case A; NG5 B/C fail at ~5.3 µs; IPTA workloads partial |
| **G6** Documented residual debt | **Open** | `ppta_j1741_ell1` ~5–8 ns; `DM_SERIES` warn-only |
| **G7** EPTA multi-backend | **Open (improved)** | ~10 ns RMS (was ~608 ns after integer-turn and `-addsat` fixes) |
| **G8** `DMASSPLANET` reflex correction | **Deferred** | Not parsed in JUG; unused in IPTA fixtures |
| **G9** Full `get_obsCoord` port | **Deferred** | Astropy/ERFA approximation already <0.01 cm on wsrt167 |

**Scorecard:** 2 closed (G1, G2 primary), 5 with open items, 2 deferred (G8, G9).

---

## 5. Active work queue

| Priority | Task | Oracle / fields | Status |
|----------|------|-----------------|--------|
| **1** | **NG5 TDB spin-epoch / TDB-TCB map** | pytempo `deltaT`; F0 column passes | **Open** — ~5.3 µs; top blocker |
| **2** | **Canonical tangent validation** | libstempo two-par perturbation oracles | **Open** — across all graph modes |
| **3** | Close **EPTA J0613 IPTA full ~10 ns** | `test_tempo2_ipta_dr2_j0613_parity.py` | **Open** — native BCLT/formBats epoch |
| **4** | Close **epta_j0613_addsat_min ~84 ns** | `tempo2_addsat_dtchain_diag.py` | **Open** — site-epoch scatter |
| **5** | Hygiene: `data_dir`, DM noise convention | — | **Open** |
| **6** | Audit stale dev-oracle assertions | `pytest -m dev_oracle` | **Open** |
| **7** | Polish **epta_j0030** p99 (~11 ns on 2×1999 TOAs) | outlier harness | **Open** |
| **8** | Update BIPM clock files for `epta_j0613_t2_ipta_all` | clock-file coverage | **Open** — data |
| **9** | Alternate PPTA ~16 ns | Roemer/Shapiro + TZR at `TZRMJD` | **Open** |
| **10** | Fitter TRACK −2 / `-addsat` wiring | after subset gates pass | **Open** |
| **Done** | **Phase C — TZR** | `tests/test_tempo2_tzr_parity.py` | **Done** — J0030 15.9 → ~4.7 ns RMS |
| **Done** | **Phase D Step 1 — pnNew** | `tests/test_tempo2_track2_pnnew.py` | **Done** |
| **Done** | **wsrt167 tropo-in-dt + longdouble wrap** | `test_dev_oracle_wsrt167_parity.py` | **Done** — 15.5 → 1.4 ns |
| **Done** | **`-addsat` mjd_str resync** | `epta_j0613_addsat_min` | **Done** — was ~±1 s |
| **Done** | **DE440 ephemeris offline resolution** | `tests/test_ephemeris_resolution.py` | **Done** |
| **Ruled out** | **Phase D Step 2 — wire `phase5@bbat` to production** | — | **~17.5 ns** — worse than Taylor |
| **Ruled out** | **Phase D Step 3 — `-padd` / `jump_phase`** | pytempo `phase_offset_turns` | **Exact match** |
| **Ruled out** | **clkcorr feedback delta on merged IPTA chains** | Track B diagnostic | **Zero effect** — chains converge in one iter |
| **Defer** | formBats `bbat` diagnostic fix | ~65 s off oracle | — |
| **Defer** | **G8 — `DMASSPLANET`** | no fixture coverage | Recorded §G8 |
| **Defer** | **G9 — full `get_obsCoord` port** | wsrt167 `< 0.01 cm` | Recorded §G9 |

---

## 6. JAX tempo2-native clock/delay chain

**Goal:** end-to-end JAX code that reproduces libstempo/tempo2 `updateBatsAll` →
`calculate_bclt.C` → `formBats.C` → `formResiduals.C` on the same per-TOA epochs,
not the JUG shortcut `(model_mjd − sat)×86400 − prebinary`.

| Layer | JUG production today | tempo2 native | Gap |
|-------|---------------------|---------------|-----|
| Clock / delays | `IFTE(tdb)` emission `model_mjd` + bundled `prebinary` | `TT+TT_TB` + `tdis` slot via `formBats` | **~286 ns** `batCorr` (Step 17) |
| Spin | Taylor Horner on `dt_sec_ld` + legacy TRACK −2 | `phase2+phase3` at `bbat` + `pnNew` | **~1.4 ns** best JUG path (was ~16 ns pre tropo/wrap fixes) |
| Oracle | pytempo `acceptance_residual_sec` | libstempo `psr.residuals()` | **0 ns** when full chain matches |

**Scope (in order):**

1. `calculate_bclt` iterative Roemer epoch — `sat + (TT+TT_TB+dt_SSB)/86400`.
2. `formBats.C` batCorr — `TT + TT_TB − tropo + roemer − shap − tdis`.
3. `formResiduals.C` spin — `phase2+phase3` at true `bbat`; `torb` from T2model.
4. Longdouble end-to-end until final float64 export.

**Explicit non-goals:**

- Do **not** patch production NumPy `simple_calculator` spin to `phase5@bbat` first
  (Step 18: **17.4 ns**, worse than Taylor **16.4 ns**).
- Do **not** use pytempo `torb_sec` in JUG-composed spin (Step 18: **172 ns** trap).
- Do **not** rely on `(model−sat)−prebin` as the JAX `batCorr` identity.

**Status:** scaffold implemented under `jug/residuals/tempo2_native/`.
`JUG_TEMPO2_NATIVE_GRAPH_MODE` selects the traced graph (default `staged_bclt`).

| Gate | Interim (dev_oracle) | Strict target | Notes |
|------|---------------------|---------------|-------|
| `batCorr` vs lib | **~286 ns** RMS | `< 1 ns` | IFTE model-epoch + JUG prebinary (production path) |
| `bat_corr_days` vs tempo2 | **~1.1 ns** RMS | `< 1 ns` | unified JAX strict formBats component sum (wsrt167) |
| `bat_mjd` / `bbat_mjd` vs tempo2 | **sub-µs** (two-part daysec) | `< 1 ns` | two-part `(int_day, sec_in_day)` in spin tail |
| `torb` closure vs pytempo | **~262 ns** RMS | `< 1 ns` | JUG `dt` + model-epoch `bbat` (production path) |
| BCLT `roemer` vs pytempo | **~18 ms** RMS | term ranking | fixed IFTE `tdis` in loop |
| Spin counterfactual | **~5.6 µs** RMS | `< 5 ns` | pending full BCLT `formBats` |
| Full residuals wsrt167 | skipped (flag off) | `< 5 ns` | flip after all gates green |

Tests: `tests/test_tempo2_native_*.py`, `tests/test_jax_tempo2_native_*.py` (`dev_oracle`).

### formBats `bat_mjd` / `bbat_mjd` assembly — unified JAX path

**Do not confuse delay parity with MJD epoch parity.** On wsrt167 the unified JAX
native chain (`jug/residuals/tempo2_native/formbats_jax.py`) shows:

| Quantity | Native vs tempo2 RMS | What it tests |
|----------|---------------------|---------------|
| `bat_corr_days` | **~1.1 ns** | Delay-component closure (physics gate) |
| `bat_mjd` | **~304 ns** | Assembled MJD epoch scalar |
| `bbat_mjd` | **~304 ns** | Same on wsrt167 (Shklovskii ≈ 0) |
| `shklovskii_sec` | **0 ns** | Not the wsrt167 blocker |

**Root cause:** tempo2 `formBats.C` uses `long double` and **splits the UTC→TT term**:

```c
// ref-packages/tempo2/formBats.C (L67–83)
batCorr = getCorrectionTT(obsn)/SECDAY
        + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
bat  = sat + getCorrectionTT(obsn)/SECDAY
     + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
bbat = bat - shklovskii/SECDAY;
```

JUG's JAX helper sums all correction seconds in float64, divides once, and
assembles with `assemble_mjd_from_day_sec`. That recipe is **internally consistent**
but **does not reproduce tempo2's split long-double assembly** when `sat` is
O(10⁴) MJD and the net correction is O(10²) s.

**Is this a residual-parity blocker?** Not automatically. The native chain defines
`torb` as a closure (`dt_emission − (bbat − PEPOCH)·86400`) and feeds
`phase5(bbat, torb)`. A constant ~304 ns shift in `bbat` can cancel against
`torb` in `deltaT = (bbat−PEPOCH)·86400 + torb` as long as the integer MJD day
of `bbat` is unchanged.

**What to gate on:**

- **Physics / delay closure:** `bat_corr_days` and per-component gates — target **< 1 ns**.
- **MJD assembly parity:** `bat_mjd` / `bbat_mjd` — requires porting tempo2's
  split summation (or equivalent compensated float64) in `formbats_jax.py`.
- **End-to-end:** `acceptance_residual_sec` vs libstempo/tempo2 — the only gate
  that ultimately matters for production parity.

---

## 7. Diagnostic workflow (§0)

### pytempo is expanded libstempo

[`ref-packages/pytempo`](../../pytempo) is **not** a second timing engine. It is an
**expanded libstempo**: same Cython → tempo2 wrapper (`updateBatsAll`, `formResiduals`,
`t2FitFunc_*`, …), forked from
[vhaasteren/libstempo `sandbox`](https://github.com/vhaasteren/libstempo/tree/sandbox).
pytempo adds per-TOA `obsn[]` field dumps via `toa_diagnostics()`.

### Acceptance oracle (pytest)

```python
# Tier-1 oracle — compare to libstempo psr.residuals()
from jug.testing.tempo2_reference import tempo2_reference
ref = tempo2_reference(par, tim)
delta = jug_residuals - ref.residuals_sec
```

### Term-by-term debugging loop

1. Run JUG + libstempo scalar gate (`test_tempo2_residual_parity.py`).
2. If failing, pull pytempo `toa_diagnostics(removemean=False)`.
3. Rank per-TOA term deltas (roemer, clock, spin, binary).
4. Localize to a single term or epoch class.
5. Fix in native JUG code; re-gate.

### Oracle tier retrospective

| Tier | Safe for parity | Trap |
|------|-----------------|------|
| **1** | `psr.residuals()`, `acceptance_residual_sec`, `bbat_mjd`, `batCorrs`, `pets` | — |
| **2** | `phase_offset_turns` | `residual_sec` on TRACK −2 |
| **3** | informational | `nphase` vs `pulse_number`; `binarydelay` stale on fresh construct |

Naïve float64 recompositions `sat + bat_corr → bat` (~304 ns on wsrt167) are
**assembly-recipe mismatches**, not delay disagreements. `bat_corr_days` is the
delay gate. libstempo `binarydelay` reads **zeros** on fresh construct — use
`torb_sec` or `prebinary − total` (0.17 ns on wsrt167).

---

## 8. J0613 three parity budgets

### 1. EPTA full multi-backend (~10 ns) — dominant

1369 TOAs, multi-backend mix. Bulk non-addsat RMS ~5.4 ns; `-addsat` TOAs at
idx 247/256/561 ~200 ns. Closing requires native BCLT/formBats epoch alignment.

### 2. Alternate PPTA export (~16 ns)

Roemer/Shapiro + TZR at `TZRMJD` budget on alternate par/tim export.

### 3. Native PPTA (~1.4 ns) — essentially done

`PPTA_dr1dr2` native export passes strict gate.

---

## 9. Phase and TRACK −2 semantics

### Comparison table

| Aspect | tempo2 `formResiduals.C` | JUG production (TRACK −2) |
|--------|--------------------------|---------------------------|
| Spin formulation | `phase2+phase3` at `bbat` | Taylor Horner at `model_mjd` |
| TRACK −2 bookkeeping | `pnNew` / `pnAct` / `addPhase` | `track_minus2_frac_phase` + legacy wrap |
| Emission epoch | `calculate_bclt` + `formBats` | IFTE `model_mjd` + `dt_sec_ld` |
| Best measured RMS | 0 ns (pytempo acceptance) | **1.4 ns** (wsrt167, post-fix) |

### Subset pitfall

tim `-pn` values are offsets relative to **full-tim** `obsn[0]`. Using raw `-pn` in
`pnAct` breaks TRACK −2 on IPTA exports. Prefer full-tim oracle pull + mask on
filtered subsets.

---

## 10. Appendix — investigation log (condensed)

### Phase C — TZR reference phase (2026-07-05) — DONE

- **Goal:** close epta_j0030_isolated (~15.9 ns RMS dominated by 2×1999 TOAs far from `TZRMJD`).
- **Root cause:** TZR geometry applied at wrong epoch / with wrong clock chain.
- **Fix:** `tzr_geometry.py` — mode-specific TZR apply modes.
- **Result:** J0030 15.9 → **~4.7 ns RMS**; max ~11 ns on 2×1999 TOAs.
- **Tests:** `tests/test_tempo2_tzr_parity.py`.

### Phase D — wsrt167 TRACK −2 (2026-07-05 to 2026-07-07)

| Step | Investigation | Verdict |
|------|---------------|---------|
| 1 | pnNew / tim `-pn` convention | **Done** — `pnAct = (pn[i]−pn[0]) + pnAdd` |
| 2 | Wire `phase5@bbat` to production | **Ruled out** — ~17.5 ns vs production ~16.4 ns |
| 3 | `-padd` / `jump_phase` | **Ruled out** — exact match to pytempo `phase_offset_turns` |
| 4 | Taylor vs tempo2 `phase2+phase3` | **Ruled out** — 0.02 ns fractional |
| 5 | Per-TOA term diff | **Done** — ~330 ns `bbat` gap localized |
| 6 | `model_mjd` vs `pet`/`torb` | **Done** — `pt_torb ≈ prebin−total` (0.17 ns) |
| 7 | `dt_sec` precision + `deltaT(pt)` counterfactual | **Done** — float64 inputs cap `dt_sec`; swap worsens |
| 8 | JAX two-part dt + compensated Taylor | **Done** — no gain (Horner ruled out again) |
| 9 | Production `IFTE(tdb_ld)` + `dt_sec_ld` best path | **Done** — 16.4 ns best JUG-composed |
| 10 | formBats signed-term probe | **Done** — +65 s JUG TT/batCorr baseline mismatch |
| 11 | getCorrectionTT / correctionTT_TB | **Done** — true `correction_tt_tb` ~14 s matches `tt2tdb.C` |
| 12 | `batCorrs` vs production `model_mjd` epoch chain | **Done** — 286 ns `batCorr` model-chain error |
| 13 | Model-epoch batCorr/bbat diagnostic rebuild | **Done** — closes libstempo `batCorrs` at 286 ns |
| 14 | `torb` closed; `bbat` epoch mismatch | **Done** — `torb` 0.17 ns; `bbat` 330 ns open |
| 15 | `model_mjd` vs `obsn.bbat` decomposition | **Done** — 286 ns `batCorr` primary lever |
| 16 | Longdouble `model_mjd` / `batCorr` replay | **Done** — 286 ns does not collapse; best ld 214 ns |
| 17 | formBats / calculate_bclt Roemer epoch | **Done** — formBats replay 0 ns; 286 ns = IFTE scatter |
| 18 | Taylor / `formResiduals.C` spin bookkeeping | **Done** — production Taylor 16.43 ns best; 0 ns needs JAX chain |

### wsrt167 15.5 ns floor — SOLVED (2026-07-07)

Two independent JUG bugs, both now fixed:

1. **Missing troposphere in the emission dt (13.1 ns of the 15.5).** Host-setup
   tropo went into `prebinary_delay_sec`/`bbat` but was **never folded into
   `total_delay_sec`/`dt_sec`** used by the Taylor spin path. Fix: fold native tropo
   into `total_delay_sec`/`dt_sec` right after host setup.
2. **float64 phase downcast in the Taylor wrap (~6 ns).** `compute_phase_residuals`
   cast longdouble Taylor phase to float64 before wrapping. Fix: keep longdouble
   through wrapping and round `nphase` in longdouble.

After both fixes: wsrt167 **1.43 ns RMS**; wrap exact (substituting tempo2 deltaT
reproduces tempo2 residuals bit-for-bit modulo mean).

### Parity-closure plan outcome (2026-07-08)

| Check | EPTA full (1369 TOA) | Verdict |
|-------|----------------------|---------|
| `feedback_delta` (clkcorr iters 3−1) | **0 ns** RMS | Merged IPTA chains converge in one iter — **not** the residual driver |
| `corr(residual, feedback_delta)` | NaN | Track B gate **not met**; fix applied anyway (no-op) |
| `corr(residual, dt_jug − deltaT_pt)` | **r ≈ 0.07** | Weak — Taylor host spin partially absorbs tempo2 `bbat+torb` |
| `sat_mjd` vs pytempo (incl `-addsat`) | **< 1 ns** | `-addsat` read-time application **closed** |
| Non-addsat bulk RMS | **~5.4 ns** | Just above 5 ns gate |
| `-addsat` TOAs (idx 247/256/561) | **~200 ns** | dt-chain / site-epoch debt |

**Track B (clkcorr feedback delta):** zero effect on IPTA DR2.
**Track E (strict gate flip):** rollback applied — EPTA full stays `xfail(strict=True)`.

### Ruled out / do-not-do list

| Action | Verdict | Why |
|--------|---------|-----|
| Wire `phase5@bbat` to production spin | **Ruled out** | ~17.5 ns — worse than Taylor ~16.4 ns |
| Change `-padd` / `jump_phase` handling | **Ruled out** | exact match to pytempo |
| Horner / compensated float64 spin alone | **Ruled out** | no gain (Step 8) |
| Pair pytempo `torb_sec` in JUG-composed `phase5` | **Trap** | **172 ns** (Step 18) |
| Longdouble clock/spin pass | **Reverted** | bit-identical RMS |
| BCLT iteration in `simple_calculator` | **Ruled out** | ~903 µs regression |
| clkcorr feedback delta on merged IPTA chains | **Ruled out** | zero effect |
| idx 85 as isolated pnNew bug | **Red herring** | tail of spin error; pytempo exact at idx 85 |
| Enable `USE_NATIVE_BBAT_PHASE5` | **Quarantined** | ~36 ns; formBats `bbat` ~65 s off |

### Phase 2 IFTE / `tt_tb` (2026-07-07) — DONE

Root cause: `jug/utils/ifteph.py` `IFTEinterp` used `frac(t[0])` where Tempo2 C
uses `modf(t[0], &dt1)` integer part in `tc` computation. Fix: line-accurate `tc`/`l`
in host port; static IFTE coefficient table in `Tempo2ModelStatic`; `ifte_delta_t_sec_jax`
evaluated inside `compute_tempo2_correction_tt_tb_jax`.

Gates (wsrt167, pytempo oracle): `tt_jax`, `teph_jax`, `tt_tb_jax` all **< 1 ns RMS**.

---

## 11. Commands and tests

```bash
cd ref-packages/jug

# JUG tempo2 parity (acceptance oracle = libstempo)
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_residual_parity.py -q -o addopts=''

# wsrt167 debt pin (dev oracle — requires libstempo + $TEMPO2)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q

# TZR parity (Phase C)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 pytest tests/test_tempo2_tzr_parity.py -q

# TRACK −2 pnNew (Phase D Step 1)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 pytest tests/test_tempo2_track2_pnnew.py -q

# All tempo2 oracle tests
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_*.py -q -o addopts=''

# JUG-only CI path (no libstempo)
pytest -m 'not dev_oracle' -q

# J0613 EPTA debt pins
PYTHONPATH=.:tests TEMPO2=$TEMPO2 pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -q

# Fast hybrid regression probes
JAX_ENABLE_X64=1 PYTHONPATH=.:tests python3 -m pytest \
  tests/test_tempo2_obs_state_export.py \
  tests/test_tempo2_native_staging_host_frozen.py \
  tests/test_tempo2_native_residual_delta_jax.py -q

# Fast native gate path (skips @pytest.mark.slow)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_native_*.py -m 'dev_oracle and not slow' --no-cov -q
```

See [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md) for the full parity table
and delete checklist.
