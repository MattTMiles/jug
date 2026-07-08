# Tempo2 parity — status and roadmap

Where JUG `compatibility="tempo2"` stands vs libstempo/tempo2: measured residual
debt, open gaps, production behavior, and the path to production-ready fitting.

**Definitions and policy:** [`PARITY_THEORY.md`](PARITY_THEORY.md)  
**Fixture provenance:** [`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md)  
**Dev oracle harness:** [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md)

*Last updated: 2026-07-08 (fact-checked vs in-repo pytest; host parity on external IPTA DR2 data measured manually, not CI-gated)*

---

## Summary

JUG `compatibility="tempo2"` has **two evaluation layers** (see [`PARITY_THEORY.md`](PARITY_THEORY.md)):

1. **Host residuals** — `compute_residuals_simple` / `TimingSession.compute_residuals`,
   compared to libstempo for parity gates.
2. **JAX fit / autodiff** — `make_residual_delta_jax_fn` +
   `design_matrix_method="autodiff"`, used for NUTS/WLS whitening and design matrices.
   Entry point: `jug/fitting/jax_residual_delta.py`; graph mode via
   `JUG_TEMPO2_NATIVE_GRAPH_MODE` (default `staged_bclt`).

**Host residual parity** on curated par+tim fixtures is **green at sub-ns to low-ns**
for TCB/TDB probes, NG5 Cases B/C, simulated option coverage, and production-scale
IPTA workloads (EPTA J0613 1369 TOAs, PPTA J0613 410 TOAs, wsrt167 TRACK −2).

**Autodiff is wired and in use** for tempo2 fitting when callers set
`design_matrix_method="autodiff"` (`jug/fitting/jax_residual_delta.py`,
`optimized_fitter.compute_designmatrix` / `_build_general_fit_setup_from_files`).
Requires `native_chain_static` populated from a prior `compute_residuals` cache with
`term_diagnostics['tempo2_obs_state']` (see `tests/test_tempo2_obs_state_export.py`).
Default WLS entry points in `optimized_fitter` still default to
`design_matrix_method="analytic"` — tempo2 nonlinear fits should opt into **autodiff**
(see TODO in `optimized_fitter.py`).

Manual validation on external IPTA DR2 multi-PTA NUTS workloads (EPTA + PPTA tempo2
sessions, `staged_bclt` autodiff whitening) has been run outside this repository; that
path is **not CI-gated**.

**Remaining gaps:**

1. **Autodiff oracle coverage** — F0 column matches libstempo on wsrt167; multi-parameter
   and binary columns lack libstempo parity gates; `full` graph mode is opt-in only.
2. **JAX compile cost** — multi-minute first compile on small fixtures; see § JAX autodiff graph.
3. **Documented residual debt** — `ppta_j1741_ell1` ~5.5 ns; partial IPTA TDB sweep
   `J0900-3144` ~5.8 ns.
4. **Hygiene / audit** — stale dev-oracle assertions; BIPM clock extrapolation on J0613
   (data, shared with libstempo).
5. **IPTA `-addsat` autodiff gates** — TRACK −2 autodiff design matrix is **green on
   wsrt167** (`test_tempo2_native_residual_delta_jax.py`, `dev_oracle`/`slow`). No pytest
   yet gates autodiff on IPTA `-addsat` fixtures (`epta_j0613_addsat_min`, full EPTA J0613).

Treat tempo2 mode as **experimental** outside curated par+tim tests. Do not use
`design_matrix_method="analytic"` for tempo2 **nonlinear** fits (FD/binary/astrometry
together); analytic columns exist for selected params but are not the supported
tempo2 tangent.

**Default dev loop (~1 min):** mini J0613 gates + simulated structural tests,
`-m 'not slow'`. **Reduced regression gate (~5 min):** adds host residual-parity
tests → **16 passed, 1 xfailed** (pint-default diagnostic). Sim residual parity,
NG5, EPTA full, wsrt167, and dev-oracle autodiff tests are `slow` or `dev_oracle`.

---

## Residual parity dashboard

Oracle: **JUG − libstempo**, raw pre-fit residuals (microseconds → nanoseconds).
Measured 2026-07-08 vs live libstempo on in-repo fixtures unless noted.

**Out-of-repo checks:** the PPTA J0613 row and partial IPTA DR2 TDB sweep below used
full IPTA DR2 par+tim trees on a local checkout (not committed under `tests/`). Those
numbers are manual oracle measurements, not pytest gates.

### Real fixtures (`tests/data_tempo2/`)

| Workload | n | RMS Δ | max \|Δ\| | Gate | CI |
|----------|---|-------|-----------|------|-----|
| Case A / `epta_j0030_isolated` (TCB) | 10 | **0.98 ns** | 1.71 ns | 5 ns | green |
| **NG5 Case B** (equatorial TDB) | 625 | **1.11 ns** | 2.63 ns | 5 ns | green (`slow`) |
| **NG5 Case C** (ecliptic TDB) | 625 | **1.11 ns** | 2.63 ns | 5 ns | green (`slow`) |
| `epta_j1909_t2` | 27 | **1.22 ns** | 3.27 ns | 5 ns | green |
| `epta_j1918_ddh` | 12 | **0.51 ns** | 1.15 ns | 5 ns | green |
| `ppta_j1741_ell1` | 111 | **5.50 ns** | 11.45 ns | 5 ns | **documented gap** |
| `ppta_j1902_ell1h` | 120 | **1.56 ns** | 3.21 ns | 5 ns | green |
| `epta_j0613_t2_nrt1400` | 120 | **0.95 ns** | 2.24 ns | 5 ns | green |
| **`wsrt167`** (TRACK −2 spin) | 167 | **1.19 ns** | 3.95 ns | 2.5 ns | green (`slow`) |
| `epta_j0613_addsat_min` | 11 | **1.43 ns** | 1.90 ns | 5 ns | green (`slow`) |
| **`epta_j0613_t2_ipta_all`** | 1369 | **1.22 ns** | 6.94 ns | 5 ns | green (`slow`) |
| PPTA J0613 (IPTA DR2 `PPTA_dr1dr2` layout) | 410 | **1.12 ns** | 3.83 ns | 5 ns | manual oracle |

`-addsat` TOAs on full EPTA J0613 (idx 247, 256, 561): max **2.33 ns**.

### Simulated fixtures (`tests/data_tempo2_sim/`)

15 libstempo-generated fixtures, 6–10 TOAs each. Default gate: **5 ns RMS**.
Generator: `tools/generate_tempo2_sim_fixtures.py`. Tests:
`tests/test_tempo2_simulated_fixtures.py`.

| Fixture class | RMS Δ (representative) | CI |
|---------------|------------------------|-----|
| TCB isolated / T2 / DD / DDH / DDK / FD / multisys / TRACK −2 | **0.4–0.7 ns** | green |
| TDB / ecliptic (`sim_dd_tdb`, `sim_dd_ecliptic_tcb`) | **< 0.5 ns** | green |
| TRACK −2 `-addsat` (`sim_t2_track2_addsat`) | **0.84 ns** | green (strict; relaxed gate legacy) |
| ELL1 (`sim_ell1_tcb`) | **2.5 ns** | green |
| BT (`sim_bt_tcb`) | **4.4 ns** | green (`slow`, under 5 ns gate) |

Green sim tests verify **option coverage** and isolate regressions; they do not
imply parity on arbitrary production par+tim pairs.

### Partial IPTA DR2 TDB sweep (incomplete)

A 65-pulsar all-PTA campaign was started and stopped after the early alphabetical
subset. Do **not** treat it as full validation. Spot result requiring follow-up:

| Pulsar | RMS Δ | Status |
|--------|-------|--------|
| `J0034-0534` TDB | ~0.5 ns | green |
| `J0437-4715` TDB | ~0.34 ns | green |
| `J1713+0747` TDB | ~1.34 ns | green |
| `J0900-3144` TDB | ~5.8 ns | **open** |

Run one pulsar or a short named batch at a time; capture RMS/p99/max before expanding.

---

## CI verification map

What is gated today (derive status from tests, not from this doc alone):

| Capability | Test module | Marker | Status |
|------------|-------------|--------|--------|
| Host residuals, Case A + binaries | `test_tempo2_residual_parity.py` | `tempo2`, not `slow` | **green** (J1741 documented gap) |
| Host residuals, NG5 TDB B/C | `test_tempo2_residual_parity.py` | `slow` | **green** |
| Host residuals, sim fixtures | `test_tempo2_simulated_fixtures.py` | `slow` | **green** (5 ns gate) |
| Host residuals, EPTA J0613 full | `test_tempo2_ipta_dr2_j0613_parity.py` | `slow` | **green** (5 ns strict) |
| Host residuals, wsrt167 | `test_dev_oracle_wsrt167_parity.py` | `dev_oracle`, `slow` | **green** |
| Mini fast gates | `test_tempo2_j0613_fast_gates.py` | not `slow` | **green** |
| WLS fit vs libstempo | `test_tempo2_fit_parity.py` | `tempo2` | **green** (selected fixtures; analytic DM) |
| Analytic design matrix vs libstempo | `test_tempo2_designmatrix_parity.py` | `tempo2` | **green** (selected columns/fixtures) |
| Sim autodiff DM nonzero (astrometry) | `test_tempo2_simulated_fixtures.py` | `tempo2` | **green** (finite, RAJ/DECJ ≠ 0) |
| Tempo2 autodiff F0 vs libstempo | `test_tempo2_native_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (wsrt167 TRACK −2) |
| Autodiff θ=0, jacfwd/jacrev, fixed vs staged | `test_tempo2_native_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (wsrt167 TRACK −2) |
| `native_chain_static` / obs_state export | `test_tempo2_obs_state_export.py` | `slow` | **green** |
| Graph mode selector | `test_tempo2_native_graph_modes.py` | — | **green** |
| Native chain component gates | `test_tempo2_native_*.py` | `dev_oracle` | mixed (see DEV_ORACLE.md) |
| Multi-PTA tempo2 autodiff (NUTS/WLS) | — | manual | **in use** externally; not in pytest |
| IPTA `-addsat` autodiff design matrix | — | — | **open** — host residuals green; no autodiff gate |

---

## Open gaps

### Blockers

| ID | Gap | Severity | Notes |
|----|-----|----------|-------|
| **O1** | Autodiff oracle coverage | **High** | Production path live; F0 vs libstempo green on wsrt167 (TRACK −2). Need multi-param/binary columns vs libstempo; `full` mode; IPTA `-addsat` autodiff smoke |
| **O2** | JAX autodiff compile cost | **Medium** | Multi-minute first compile; not a parity blocker — see § JAX autodiff graph |
| **O3** | `ppta_j1741_ell1` ~5.5 ns | **Medium** | ELL1 convention — `test_tempo2_mode_ell1_j1741_documented_gap` |
| **O4** | Stale dev-oracle assertions | Low | Audit/retire (`pytest -m dev_oracle`) |
| **O5** | BIPM2011 clock extrapolation (J0613 to ~56796) | Low | **Data** — shared JUG/libstempo constant extrapolation |
| **O6** | IPTA `J0900-3144` TDB ~5.8 ns | Medium | From partial sweep; needs focused probe |

### Gap scorecard (internal parity)

| Gap | Status | Summary |
|-----|--------|---------|
| **G1** NumPy `residual_delta(0) ≠ 0` | Closed | `get_longdouble()` for `HIGH_PRECISION_PARAMS` |
| **G2** JAX autodiff at θ=0 | Closed | Unified delay kernel; θ=0 peak ≲10⁻¹³ s |
| **G2 residual** NumPy vs JAX at θ≠0 | **Partial** | θ=0 closed; wsrt167 autodiff finite-difference spot check on F0 |
| **G4** Analytic design matrix | **Partial** | Selected columns match libstempo on gated fixtures; tempo2 nonlinear fits must use autodiff |
| **G5** Fixture coverage | Good | Host green on Case A, NG5 B/C, sim, EPTA J0613 full; autodiff F0 gated on wsrt167 (TRACK −2) |
| **G6** Documented residual debt | **Open** | `ppta_j1741_ell1`; `DM_SERIES` warn-only |
| **G8** `DMASSPLANET` reflex | Deferred | Not parsed; unused in IPTA fixtures |
| **G9** Full `get_obsCoord` port | Deferred | Astropy/ERFA already <0.01 cm on wsrt167 |

### Work queue (priority order)

| # | Task | Gate / oracle |
|---|------|---------------|
| 1 | Broaden autodiff oracle coverage | Multi-param/binary columns vs libstempo; IPTA `-addsat` autodiff; `full` mode |
| 2 | Trim JAX autodiff graph (T1–T6) | Compile time; parity gates stay green |
| 3 | `ppta_j1741_ell1` convention gap | Document or close to <5 ns |
| 4 | Audit stale dev-oracle assertions | `pytest -m dev_oracle` |
| 5 | BIPM clock files for J0613 | Data update |
| 6 | IPTA `J0900-3144` TDB probe | Per-pulsar oracle |
| 7 | Default tempo2 WLS to autodiff | `optimized_fitter` default remains analytic — callers must opt in |

---

## Roadmap phases

### Phase 1 — Autodiff tangent validation *(current priority)*

Production wiring is **done**; oracle breadth is **not**.

- [x] Tempo2 `residual_delta_jax` + `design_matrix_method="autodiff"` in
  `jax_residual_delta.py` / `optimized_fitter.py` (requires `native_chain_static`).
- [x] Default graph mode `staged_bclt`; selector tested (`test_tempo2_native_graph_modes.py`).
- [x] `native_chain_static` / obs_state export (`test_tempo2_obs_state_export.py`).
- [x] wsrt167 (TRACK −2): F0 autodiff column vs libstempo; jacfwd/jacrev agreement; reverse-mode grad finite.
- [x] wsrt167 (TRACK −2): `fixed_state_nonlinear` vs `staged_bclt` envelope at PTA-scale ε (< 1 ns).
- [x] Sim tempo2 autodiff: astrometry columns nonzero (`test_simulated_tempo2_autodiff_designmatrix_astrometry_nonzero`; no TRACK −2 / `-addsat`).
- [ ] Multi-parameter autodiff columns vs libstempo (RAJ, DECJ, DM, binary on gated fixtures).
- [ ] Autodiff design matrix on IPTA `-addsat` workloads (`epta_j0613_addsat_min`, full EPTA J0613).
- [ ] `full` in-graph graph mode validation (dev-only today; multi-minute compile).
- [ ] Make autodiff the default for tempo2 in `fit_parameters_optimized` /
  `_build_general_fit_setup_from_files`.

### Phase 2 — Host residual closure *(complete on gated fixtures)*

- [x] NG5 TDB Cases B/C, ecliptic frame, sim TDB/ecliptic.
- [x] EPTA J0613 full IPTA + `-addsat` mini + wsrt167 TRACK −2.
- [ ] Residual debt: `ppta_j1741_ell1`, `J0900-3144` TDB (partial sweep).

### Phase 3 — Hygiene

- [ ] Dev-oracle audit; BIPM clock coverage on J0613.

### Phase 4 — Efficiency and defer

- [ ] Trim JAX computational graph (§ below).
- [ ] `DMASSPLANET` (G8), full `get_obsCoord` port (G9).

---

## Production behavior

### Host vs JAX fit routing

| Layer | TRACK absent | TRACK == −2 | other TRACK |
|-------|--------------|-------------|-------------|
| **Host residuals** | Taylor `compute_phase_residuals` | Taylor + legacy `-pn` wrap (~1.2 ns) | `compute_native_eval_residuals_jax` |
| **JAX autodiff** | Native chain → bbat Δ + Taylor phase Δ | Same — wsrt167 F0 vs libstempo gated (`dev_oracle`/`slow`) | Same |

**Host residuals** (`pipeline.finalize_tempo2_host_residuals`): see
`jug/residuals/tempo2_native/pipeline.py`.

**JAX fit/autodiff** (`jax_residual_delta.py`, all TRACK values):

1. Trace tempo2-native chain via `JUG_TEMPO2_NATIVE_GRAPH_MODE` (default `staged_bclt`).
2. Compute **`compute_native_bbat_delay_change_sec_jax`** (bbat displacement between
   θ and θ+Δθ, plus binary delay change when fitted).
3. Apply **`_phase_residual_delta_jax`** — Taylor phase difference on frozen
   `dt_sec_cached` and the delay change (not the host Taylor spin path).

The traced native chain includes BCLT, formBats, and spin terms needed to obtain bbat;
the **fitting tangent** uses bbat displacement + Taylor phase, not host `model_mjd` spin.
Requires `native_chain_static` with `term_diagnostics['tempo2_obs_state']`. Prime by
calling `compute_residuals(..., force_recompute=True)` (or loading an equivalent
session cache) before `_build_general_fit_setup_from_cache` /
`_build_general_fit_setup_from_files` — pattern in `tests/test_tempo2_obs_state_export.py`.

**Design matrix:** `compute_autodiff_designmatrix_from_setup` = `jacfwd(residual_delta)`;
whitening in NUTS uses this path when `design_matrix_method="autodiff"`.

### TRACK −2 and spin

| Aspect | tempo2 `formResiduals.C` | JUG production |
|--------|--------------------------|----------------|
| Spin | `phase2+phase3` at `bbat` | Taylor Horner at `model_mjd` |
| TRACK −2 | `pnNew` / `pnAct` / `addPhase` | `track_minus2_frac_phase` + legacy wrap |
| Emission epoch | `calculate_bclt` + `formBats` | IFTE `model_mjd` + `dt_sec_ld` |
| Measured RMS | 0 ns (acceptance) | **~1.2 ns** on gated fixtures |

**Subset pitfall:** tim `-pn` values are offsets relative to **full-tim** `obsn[0]`.
Using raw `-pn` in `pnAct` breaks TRACK −2 on filtered subsets. Prefer full-tim
oracle pull + mask.

**Do not wire to production:** native `phase5@bbat` host spin (`USE_NATIVE_BBAT_PHASE5`,
~36 ns); BCLT iteration in `simple_calculator` (~903 µs regression).

### `-addsat` and tempo2 options

| Item | Status |
|------|--------|
| `-addsat` SAT shift + `mjd_str` resync | Closed — idx 247/256/561 max **2.33 ns** (host residuals) |
| `-addsat` autodiff design matrix | **Open** — no pytest gate on IPTA `-addsat` fixtures |
| `track_minus2_frac_phase` pnAct | Closed — `pnAct = (pn[i]−pn[0]) + pnAdd` |
| TZR geometry (`tzr_geometry.py`) | Closed — e.g. `epta_j0030_isolated` 0.98 ns |
| Implicit `NE_SW = 4` cm⁻³ when par omits keyword | Closed |
| Roemer PM at POSEPOCH | Closed |
| IFTE + `formBats` clock (`tempo2_clock.py`) | Diagnostic-only — production host uses `model_mjd` |

### Native host-path capabilities (tempo2 mode)

These are implemented in the native JUG host path and underpin current parity:

- TDB vs TCB ephemeris scaling (IFTE/`SI_UNITS` only for SI/TCB)
- Per-TOA multi-observatory ITRF geometry
- Tempo1 emulation (`EPHVER < 5`, T2C_TEMPO, legacy obliquity)
- Coverage-aware clock chains (`ClockGraph`)
- Longdouble SAT subtraction in clock feedback
- Tempo2-native T2/Kopeikin binary (model id 6)
- Native overlay fold-in (Roemer/DM/SW → `total_delay_sec` / `dt_sec`)
- Tempo2 `observatories.dat` aliases (Effelsberg, Jodrell, WSRT, IPTA `w`/`aoutc`)
- Ecliptic `equ2ecl` on tempo2-native observatory state for ELONG/ELAT pulsars

---

## Fast gates and CI tiers

### J0613 inner loop

| Gate | Fixture | n | Measured | Pin |
|------|---------|---|----------|-----|
| No TRACK / no `-pn` | `epta_j0613_nrt1400_mini` | 20 | stored oracle | 100 ns |
| TRACK −2 `-addsat` | `epta_j0613_addsat_min` | 11 | **1.43 ns** RMS | 5 ns / 1 µs anti-wrap | host only |
| WSRT TRACK −2 spin | `wsrt167_mini` | 20 | strict | 5 ns RMS |
| Full wsrt167 | `wsrt167` | 167 | **1.19 ns** RMS | 2.5 ns + 5 ns strict |
| Full EPTA | `epta_j0613_t2_ipta_all` | 1369 | **1.22 ns** RMS | 5 ns strict |

Mini fixtures run in ~1 min. Full fixtures and the 11-TOA addsat **host-residual**
gate are `slow` (release / explicit-request jobs).

### Test commands

```bash
# From the jug repository root (directory containing tests/ and jug/)

# Default dev loop (~1 min)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_j0613_fast_gates.py \
         tests/test_tempo2_simulated_fixtures.py \
  -q -o addopts='' --no-cov -m 'not slow'

# Reduced regression gate (~5 min)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_j0613_fast_gates.py \
         tests/test_tempo2_simulated_fixtures.py \
         tests/test_tempo2_residual_parity.py \
  -q -o addopts='' --no-cov -m 'not slow'

# Slow acceptance (NG5, EPTA full, wsrt167 — explicit request)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_residual_parity.py -k ng5 -q -o addopts='' --no-cov
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -q -o addopts='' --no-cov

# Dev-oracle native chain (requires libstempo)
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_native_*.py -m 'dev_oracle and not slow' --no-cov -q

# JUG-only CI (no libstempo)
pytest -m 'not dev_oracle' -q
```

Full `pytest -k tempo2` and 65-pulsar IPTA campaigns are multi-hour — launch only on
explicit request.

---

## JAX native delay chain

**Goal:** JAX code reproducing libstempo/tempo2 `updateBatsAll` → `calculate_bclt` →
`formBats` → `formResiduals` on the same per-TOA epochs.

| Layer | JUG production | tempo2 native | Dev-oracle gap |
|-------|----------------|---------------|----------------|
| Clock / delays | IFTE `model_mjd` + `prebinary` | `TT+TT_TB` + `tdis` via `formBats` | ~286 ns `batCorr` |
| Spin | Taylor on `dt_sec_ld` + TRACK −2 | `phase2+phase3` at `bbat` | **~1.2 ns** host residuals on gated fixtures |
| End-to-end residuals | libstempo oracle | `psr.residuals()` | **< 7 ns** on gated fixtures |

Scaffold: `jug/residuals/tempo2_native/`. Graph mode:
`JUG_TEMPO2_NATIVE_GRAPH_MODE` (default `staged_bclt`).

### Dev-oracle gates (component-level)

| Gate | Current | Target | Notes |
|------|---------|--------|-------|
| `bat_corr_days` vs tempo2 | ~1.1 ns RMS | < 1 ns | Delay physics gate |
| `bat_mjd` / `bbat_mjd` vs tempo2 | ~304 ns RMS | < 1 ns | MJD assembly recipe — see below |
| `batCorr` vs lib | ~286 ns RMS | < 1 ns | Model-epoch chain |
| `torb` vs pytempo | ~262 ns RMS | < 1 ns | Production path |
| Full residuals wsrt167 | **1.19 ns** | < 5 ns | **Green** |

Component gaps can coexist with green end-to-end residuals when Taylor spin at
`model_mjd` absorbs epoch-scalar offsets.

### formBats MJD assembly

**Do not confuse delay parity with MJD epoch parity.**

| Quantity | vs tempo2 RMS | What it tests |
|----------|---------------|---------------|
| `bat_corr_days` | ~1.1 ns | Delay-component closure |
| `bat_mjd` / `bbat_mjd` | ~304 ns | Assembled MJD scalar |
| End-to-end residuals | ~1.2 ns | Production gate |

tempo2 `formBats.C` splits the UTC→TT term in `long double`:

```c
batCorr = getCorrectionTT(obsn)/SECDAY
        + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
```

JUG JAX sums all correction seconds in float64, divides once. Internally consistent
but does not reproduce tempo2's split assembly when `sat` is O(10⁴) MJD. A constant
~304 ns `bbat` shift can cancel in `deltaT = (bbat−PEPOCH)·86400 + torb` if the
integer MJD day is unchanged.

**Gate on:** `bat_corr_days` and end-to-end `acceptance_residual_sec`. MJD scalar
parity requires porting split summation in `formbats_jax.py` — not a current
residual blocker.

### Graph modes (`staged_bclt` vs `fixed_state_nonlinear`)

| Layer | `staged_bclt` (default) | `fixed_state_nonlinear` |
|-------|-------------------------|-------------------------|
| Ephemeris / clocks / obs vectors | Host-frozen from `tempo2_obs_state` | Same |
| BCLT iteration | **Recomputed in JAX** (`lax.scan` × 12/TOA) | **Frozen** at reference BCLT |
| formBats / Shklovskii / spin tail | In JAX | In JAX |

`NativeFrozenDeltaPack` = frozen **host inputs**, not frozen BCLT output.
See [`PARITY_THEORY.md`](PARITY_THEORY.md) §5.

### JAX autodiff graph — compile debt

Fitting uses `make_residual_delta_jax_fn()` + `jax.jacfwd` at bind time. Tempo2
`staged_bclt` traces a much larger graph than PINT-family sessions. **Not a parity
blocker** — host residuals are green; this is compile/runtime debt.

Each `residual_delta_jax` call runs **two** native chains (θ=0 and θ+Δθ). Each chain
traces pulsar vectors, per-TOA DM, BCLT `lax.scan` (12 iter/TOA), formBats ×2,
Shklovskii, and native spin terms to obtain **bbat**. The returned delta uses only
**bbat displacement + Taylor `_phase_residual_delta_jax`** on frozen host `dt_sec_cached`.

**Trim targets (priority order):**

| # | Trim | Saves | Validation |
|---|------|-------|------------|
| **T1** | `bbat`-only subgraph — skip native spin trace where only bbat Δ is consumed | Largest compile win | `jacfwd(0)` vs libstempo; NUTS smoke |
| **T2** | Host-precomputed reference bbat at pack build | Halves BCLT+formBats per call | θ=0 delta exactly zero |
| **T3** | Skip second formBats when Shklovskii absent | Most IPTA pulsars | Envelope on PX/PMRA/PMDEC |
| **T4** | `fixed_state_nonlinear` for NUTS after envelope | Drops BCLT scan | vs `staged_bclt` < 1 ns |
| **T5** | Do not trim compensated two-part day/sec without envelope | — | Required for TDB parity |
| **T6** | XLA cache, mini fixtures in dev loop | Amortize compile | Operational |

**Do not trim:** host-frozen ephemeris/clocks; BCLT in `staged_bclt` (use
`fixed_state_nonlinear` to freeze); Taylor `_phase_residual_delta_jax` wrapper.

**Implementation anchors:**

| Component | Location |
|-----------|----------|
| JIT residual factory | `jug/fitting/jax_residual_delta.py` |
| Bbat delay change | `jug/residuals/tempo2_native/chain_jax.py` |
| Full tail (trim candidate) | `jug/residuals/tempo2_native/model_jax.py` |
| BCLT scan | `jug/residuals/tempo2_native/calculate_bclt_jax.py` |
| Compensated formBats | `formbats_jax.py`, `compensated.py` |

---

## Debugging workflow

### Oracles

**Tier 1 (acceptance):** `psr.residuals()`, `acceptance_residual_sec`, `batCorrs`,
`pets`, `bbat_mjd`.

**Tier 2:** `phase_offset_turns`. Trap: `residual_sec` on TRACK −2.

**Tier 3 (informational):** `nphase` vs `pulse_number`; libstempo `binarydelay` is
**zero** on fresh construct — use `torb_sec` or `prebinary − total`.

```python
from jug.testing.tempo2_reference import tempo2_reference
ref = tempo2_reference(par, tim)
delta_ns = (jug_residuals_us - ref.residuals_us) * 1000.0
```

### pytempo (optional external dev oracle)

**pytempo** is an expanded libstempo fork (same tempo2 wrapper + per-TOA
`toa_diagnostics()` dumps). It is **not** part of this repository and is **not** a
runtime dependency; use it only for term-by-term debugging when libstempo scalar gates fail:

1. Run `test_tempo2_residual_parity.py` gate.
2. Pull `toa_diagnostics(removemean=False)`.
3. Rank roemer / clock / spin / binary deltas.
4. Fix in native JUG code; re-gate.

Naïve `sat + bat_corr → bat` float64 recomposition (~304 ns on wsrt167) is an
**assembly-recipe mismatch**, not a delay disagreement. Use `bat_corr_days` for
delay physics.

### Design constraints (do not retry)

| Approach | Outcome | Why |
|----------|---------|-----|
| Wire `phase5@bbat` to production host spin | ~17 ns | Worse than Taylor ~1.2 ns |
| pytempo `torb_sec` in JUG-composed spin | ~172 ns | Trap |
| BCLT iteration in `simple_calculator` | ~903 µs | Regression |
| clkcorr feedback delta on merged IPTA chains | 0 effect | Chains converge in one iter |
| `USE_NATIVE_BBAT_PHASE5` | ~36 ns | Quarantined |

---

## Related docs

| Doc | Contents |
|-----|----------|
| [`PARITY_THEORY.md`](PARITY_THEORY.md) | Parity definition, two evaluation layers, graph modes |
| [`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md) | Fixture sizes, provenance, regeneration |
| [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md) | Oracle harness, hybrid gates, delete checklist |
