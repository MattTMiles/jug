# Tempo2 parity — status and roadmap

Where JUG `compatibility="tempo2"` stands vs libstempo/tempo2: measured residual
debt, open gaps, production behavior, and the path to production-ready fitting.

**Definitions and policy:** [`PARITY_THEORY.md`](PARITY_THEORY.md)  
**Fixture provenance:** [`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md)  
**Dev oracle harness:** [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md)

*Last updated: 2026-07-12 (picosecond-parity closure on `tempo2-dev`; see § Parity closure 2026-07-12 below)*

---

## Parity closure 2026-07-12 (picosecond tier)

Host residual parity vs libstempo/pytempo was driven from the ~1.2 ns floor to
the **~20–30 ps** level by closing six convention/precision gaps, each verified
per-TOA against pytempo `toa_diagnostics()` and the tempo2 C source:

1. **Spin-phase axis** — the host Taylor spin argument now uses tempo2's
   `sat + correctionTT + correctionTT_TB (− shklovskii)` clock axis (longdouble
   fold in `run_tempo2_host_stage`) instead of the astropy UTC→TDB axis; the
   two timescale realizations (erfa dtdb vs ifteph) differed by the entire
   ~1.2 ns floor (corr +1.000 with the residual delta).
2. **`jpl_pleph` JD rounding** — the SPK epoch is assembled in longdouble and
   rounded to double once (`tempo2_read_ephemeris_jd`), matching
   `readEphemeris.C`; multi-stage float64 rounding flipped a 41 µs JD ulp on
   isolated TOAs (~1 m Earth position, up to 3.5 ns Roemer).
3. **Native `freqSSB`** — barycentric frequency computed per `dm_delays.C`
   from the bootstrap obs state (was: astropy-provider value; rel 6e-11 off).
4. **Exact `einsteinRate`** — `tt2tdb.C` recipe from IFTE Chebyshev
   derivatives (`compute_tempo2_einstein_rate_exact`), replacing astropy
   numerical differentiation; was the entire tdis1 gap (0.17 ns).
5. **Per-hop clock feedback** — the observatory clock chain is interpolated at
   raw SAT (tempo2 shifts each hop only by corrections accumulated before it);
   shifting by the full TT−UTC sampled noisy maser segments ~66 s off-epoch
   (up to 7 ns on EFF/JBO).
6. **Tempo2 binary conventions** — ELL1 uses tempo2's `ELL1model.C`
   truncation (`drep = x·cosΦ`, no ε harmonics; the `an·x²·ε` cross terms were
   0.1–2 ns at 1Φ/3Φ), and the binary is evaluated at bbat *including the FD
   delay* (PINT applies FD after the binary) — together closing the
   `ppta_j1741_ell1` documented gap (5.5 → 0.033 ns) and `J0900-3144`
   (5.8 → 0.023 ns). PINT-mode kernels are unchanged (mode-gated flag).

Measured after closure (JUG − libstempo, host residuals):

| Workload | n | was | now |
|---|---|---|---|
| EPTA J0613 full raw IPTA DR2 | 1369 | 1.22 ns | **0.022 ns RMS / 0.078 ns max** |
| `epta_j0613_t2_ipta_all` (doctored) | 1369 | 1.22 ns | **0.022 ns** |
| `wsrt167` (TRACK −2) | 167 | 1.19 ns | **0.023 ns** |
| `ppta_j1741_ell1` (FD, SINI/M2) | 111 | 5.50 ns | **0.033 ns** (strict gate now) |
| `J0900-3144` TDB | 875 | ~5.8 ns | **0.023 ns** |
| `J1713+0747` (T2 Keplerian) | 1188 | ~1.34 ns | **0.021 ns** |

The 5 ns CI gates remain; the full-J0613 slow test additionally pins
< 0.1 ns RMS / < 0.3 ns max. Numbers below this section predate the closure.

---

## Summary

JUG `compatibility="tempo2"` has **two evaluation layers** (see [`PARITY_THEORY.md`](PARITY_THEORY.md)):

1. **Host residuals** — `compute_residuals_simple` / `TimingSession.compute_residuals`,
   compared to libstempo for parity gates.
2. **JAX fit / autodiff** — `make_residual_delta_jax_fn` +
   `design_matrix_method="autodiff"`, used for NUTS/WLS whitening and design matrices.
   Entry point: `jug/fitting/jax_residual_delta.py`; graph mode via
   `tempo2_native` session kwarg (default `staged_bclt`).

**Host residual parity** on curated par+tim fixtures is **green at sub-ns to low-ns**
for TCB/TDB probes, NG5 Cases B/C, simulated option coverage, and production-scale
IPTA workloads (EPTA J0613 1369 TOAs, PPTA J0613 410 TOAs, wsrt167 TRACK −2).

**Autodiff is wired and in use** for tempo2 fitting when callers set
`design_matrix_method="autodiff"` (`jug/fitting/jax_residual_delta.py`,
`optimized_fitter.compute_designmatrix` / `_build_general_fit_setup_from_files`).
Requires `native_chain_static` populated from a prior `compute_residuals` cache with
`term_diagnostics['tempo2_obs_state']` (see `tests/test_tempo2_obs_state_export.py`).
Default WLS uses `design_matrix_method="analytic"` with PINT-style simplified tangents
(`jug/fitting/designmatrix_assembly.py`), gated in
`tests/test_tempo2_analytic_designmatrix.py` vs simplified-model autodiff. Opt into
**native autodiff** for NUTS whitening and libstempo column parity.

Manual validation on external IPTA DR2 multi-PTA NUTS workloads (EPTA + PPTA tempo2
sessions, `staged_bclt` autodiff whitening) has been run outside this repository; that
path is **not CI-gated**.

**Remaining gaps:**

1. **JAX compile cost** — multi-minute first compile on small fixtures without session
   cache; see § JAX autodiff graph. With warm XLA cache on wsrt167 (167 TOAs), 4-param
   residual+jacobian JIT is ~1.5 s (`staged_bclt`), ~1.0 s (`fixed_state_bclt`),
   ~0.6 s (`fixed_state_stripped`).
2. **Documented residual debt** — `ppta_j1741_ell1` ~5.5 ns; partial IPTA TDB sweep
   `J0900-3144` ~5.8 ns.
3. **Model-epoch host diagnostics** — IFTE `model_mjd` batCorr/bbat scalars vs libstempo
   remain pinned (~272–350 ns); production host residuals and JAX `bbat` paths are green.
4. **Hygiene / audit** — BIPM clock extrapolation on J0613 (data, shared with libstempo).
5. **Native autodiff default** — WLS still defaults to analytic simplified tangents;
   callers must opt into `design_matrix_method="autodiff"` for native NUTS tangents.
6. **IPTA `-addsat` autodiff breadth** — F0 column gated on `epta_j0613_addsat_min`;
   full EPTA J0613 addsat TOAs and multi-param columns on that workload are not yet gated.
7. **`full` mode autodiff** — component parity CI is green (`test_tempo2_full_mode_parity.py`);
   libstempo design-matrix columns through `full` are not yet gated.

Treat tempo2 mode as **experimental** outside curated par+tim tests. Analytic design
matrices are **supported** for WLS (simplified tangent, approximate vs native `bbat`).
Native autodiff remains the fidelity path for NUTS and libstempo column gates.

**Default dev loop (~20 s):** `pytest -m smoke --no-cov -q` — curated
cross-engine smoke set. **Thorough (~12 min):** the full suite; the old `slow`
tier was removed 2026-07-12 (no test exceeds ~6 s). See § Test commands.

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
| Tempo2 autodiff F0/RAJ/DECJ/DM vs libstempo | `test_tempo2_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (wsrt167; staged/fixed/stripped) |
| Autodiff θ=0, jacfwd/jacrev, stripped envelope | `test_tempo2_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (< 1 ns vs staged on wsrt167) |
| Binary autodiff PB/EPS1/EPS2 vs libstempo | `test_tempo2_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (`epta_j1909_t2`) |
| `-addsat` autodiff F0 vs libstempo | `test_tempo2_residual_delta_jax.py` | `dev_oracle`, `slow` | **green** (`epta_j0613_addsat_min`) |
| `full` graph mode component parity | `test_tempo2_full_mode_parity.py` | `dev_oracle`, `slow` | **green** (wsrt167 delay terms < 1 ns) |
| Stripped lite BBAT / formBats chain | `test_tempo2_stripped_mode.py` | `dev_oracle`, `slow` | **green** (< 1 ns vs pytempo) |
| `native_chain_static` / obs_state export | `test_tempo2_obs_state_export.py` | `slow` | **green** |
| Graph mode selector | `test_tempo2_graph_modes.py` | — | **green** |
| Native chain component gates (formBats, bbat, torb, batcorr) | `test_tempo2_*.py` | `dev_oracle` | **green** on wsrt167-class probes |
| Multi-PTA tempo2 autodiff (NUTS/WLS) | — | manual | **in use** externally; not in pytest |
| IPTA full J0613 `-addsat` autodiff (multi-param) | — | — | **open** — mini F0 gated; full workload not |
| `full` mode autodiff vs libstempo | — | — | **open** — component gates only |

---

## Open gaps

### Blockers

| ID | Gap | Severity | Notes |
|----|-----|----------|-------|
| **O1** | Autodiff oracle breadth (residual) | **Low** | wsrt167 F0/RAJ/DECJ/DM + binary (`epta_j1909`) + addsat F0 (`epta_j0613_addsat_min`) gated. Remaining: `full` autodiff columns; full EPTA J0613 addsat multi-param |
| **O2** | JAX autodiff compile cost | **Medium** | Multi-minute first compile; not a parity blocker — see § JAX autodiff graph |
| **O3** | `ppta_j1741_ell1` ~5.5 ns | **Closed 2026-07-12** | Tempo2 ELL1 truncation + FD-in-binary-time; now 0.033 ns strict gate (`test_tempo2_mode_ell1_j1741_strict_parity`) |
| **O4** | Model-epoch batCorr scalar debt | Low | Host IFTE `model_mjd` chain ~272 ns vs lib (pinned < 500 ns); does not block residuals or JAX bbat |
| **O5** | BIPM2011 clock extrapolation (J0613 to ~56796) | Low | **Data** — shared JUG/libstempo constant extrapolation |
| **O6** | IPTA `J0900-3144` TDB ~5.8 ns | **Closed 2026-07-12** | Same closure fixes; measured 0.023 ns |

### Gap scorecard (internal parity)

| Gap | Status | Summary |
|-----|--------|---------|
| **G1** NumPy `residual_delta(0) ≠ 0` | Closed | `get_longdouble()` for `HIGH_PRECISION_PARAMS` |
| **G2** JAX autodiff at θ=0 | Closed | Unified delay kernel; θ=0 peak ≲10⁻¹³ s |
| **G2 residual** NumPy vs JAX at θ≠0 | **Closed (gated)** | wsrt167 autodiff columns vs libstempo for F0/RAJ/DECJ/DM; binary on `epta_j1909`; addsat F0 |
| **G4** Analytic design matrix | **Partial** | Selected columns match libstempo on gated fixtures; tempo2 nonlinear fits must use autodiff |
| **G5** Fixture coverage | Good | Host green on Case A, NG5 B/C, sim, EPTA J0613 full; JAX oracle gates on wsrt167 + epta_j1909 + addsat mini |
| **G10** JAX native delay chain vs pytempo | **Closed (gated)** | formBats, JAX `bbat_mjd`, `torb_sec`, stripped lite BBAT < 1 ns on wsrt167 |
| **G6** Documented residual debt | **Open** | `ppta_j1741_ell1`; `DM_SERIES` warn-only |
| **G8** `DMASSPLANET` reflex | Deferred | Not parsed; unused in IPTA fixtures |
| **G9** Full `get_obsCoord` port | Deferred | Astropy/ERFA already <0.01 cm on wsrt167 |

### Work queue (priority order)

| # | Task | Gate / oracle |
|---|------|---------------|
| 1 | `full` mode autodiff vs libstempo | Design-matrix columns through in-graph chain |
| 2 | IPTA full J0613 `-addsat` autodiff | Multi-param columns on addsat TOAs |
| 3 | Trim JAX autodiff graph (T1–T6) | Compile time; parity gates stay green |
| 4 | `ppta_j1741_ell1` convention gap | Document or close to <5 ns |
| 5 | Model-epoch batCorr scalar (optional) | Tighten from 500 ns pin if production needs IFTE diagnostic |
| 6 | BIPM clock files for J0613 | Data update |
| 7 | IPTA `J0900-3144` TDB probe | Per-pulsar oracle |
| 8 | Default tempo2 WLS to autodiff | `optimized_fitter` default remains analytic — callers must opt in |

---

## Roadmap phases

### Phase 1 — Autodiff tangent validation *(largely complete)*

Production wiring and core oracle gates are **done**. Remaining work is breadth on
`full` autodiff and full IPTA `-addsat` workloads.

- [x] Tempo2 `residual_delta_jax` + `design_matrix_method="autodiff"` in
  `jax_residual_delta.py` / `optimized_fitter.py` (requires `native_chain_static`).
- [x] Default graph mode `staged_bclt`; selector tested (`test_tempo2_graph_modes.py`).
- [x] `native_chain_static` / obs_state export (`test_tempo2_obs_state_export.py`).
- [x] wsrt167 (TRACK −2): F0/RAJ/DECJ/DM autodiff columns vs libstempo on
  `staged_bclt`, `fixed_state_bclt`, and `fixed_state_stripped`.
- [x] wsrt167: jacfwd/jacrev agreement; reverse-mode grad finite; stripped vs staged
  envelope < 1 ns RMS.
- [x] Binary autodiff columns vs libstempo on `epta_j1909_t2` (PB, EPS1, EPS2).
- [x] `-addsat` autodiff F0 vs libstempo on `epta_j0613_addsat_min`.
- [x] Sim tempo2 autodiff: astrometry columns nonzero (`test_simulated_tempo2_autodiff_designmatrix_astrometry_nonzero`).
- [x] `full` in-graph **component** parity on wsrt167 (`test_tempo2_full_mode_parity.py`, < 1 ns).
- [x] Tempo2 **analytic** design matrix vs simplified-model autodiff
  (`test_tempo2_analytic_designmatrix.py`; wsrt167 F0/RAJ/DECJ/DM, binary PB/EPS1/EPS2,
  `tempo2_native` invariance).
- [ ] `full` in-graph **autodiff** columns vs libstempo.
- [ ] Autodiff design matrix on full EPTA J0613 `-addsat` TOAs (multi-param).
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
| **Host residuals** | Taylor `compute_phase_residuals` | Taylor + legacy `-pn` wrap (~1.2 ns) | `compute_eval_residuals_jax` |
| **JAX autodiff** | Native chain → bbat Δ + Taylor phase Δ | Same — wsrt167 F0 vs libstempo gated (`dev_oracle`/`slow`) | Same |

**Host residuals** (`finalize_tempo2_host_residuals`): see
`jug/residuals/tempo2/host.py`.

**JAX fit/autodiff** (`jax_residual_delta.py`, all TRACK values):

1. Trace tempo2-native chain via `tempo2_native` (default `staged_bclt`).
2. Compute **`compute_bbat_delay_change_sec_jax`** (bbat displacement between
   θ and θ+Δθ, plus binary delay change when fitted).
3. Apply **`_phase_residual_delta_jax`** — Taylor phase difference on frozen
   `dt_sec_cached` and the delay change (not the host Taylor spin path).

The traced native chain includes BCLT, formBats, and spin terms needed to obtain bbat;
the **fitting tangent** uses bbat displacement + Taylor phase, not host `model_mjd` spin.
Requires `native_chain_static` with `term_diagnostics['tempo2_obs_state']`. Prime by
calling `compute_residuals(..., force_recompute=True)` (or loading an equivalent
session cache) before `_build_general_fit_setup_from_cache` /
`_build_general_fit_setup_from_files` — pattern in `tests/test_tempo2_obs_state_export.py`.

**Design matrix — analytic (default WLS):** `assemble_analytic_designmatrix` in
`designmatrix_assembly.py` — PINT-style simplified tangents, independent of
`tempo2_native`. Gated vs `compute_simplified_autodiff_designmatrix_from_setup`.

**Design matrix — native autodiff:** `compute_autodiff_designmatrix_from_setup` =
`jacfwd(residual_delta)` through the `tempo2_native` graph; NUTS whitening when
`design_matrix_method="autodiff"`.

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
| `-addsat` autodiff design matrix | **Partial** — F0 gated on `epta_j0613_addsat_min`; full EPTA J0613 not |
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

### Test commands (re-tiered 2026-07-12)

After the clock-cache/native-chain perf work the whole suite runs in ~12 min
and no single test exceeds ~6 s, so the stale `slow` marks were removed (the
marker stays registered for future genuinely slow additions). Two tiers:

```bash
# From the jug repository root (directory containing tests/ and jug/)

# SMOKE — curated cross-engine set, ~20 s wall: pint residual workflow +
# fit diagnostics, tempo2 wsrt167-mini TRACK -2 strict gate, no-track mini,
# raw J0613 live-libstempo parity, engine-convention / white-noise /
# high-precision-epoch units.
pytest -m smoke --no-cov -q -p no:cacheprovider

# FULL — everything (~12 min, 624 tests; includes all tempo2 + dev_oracle gates)
pytest --no-cov -q -p no:cacheprovider

# Subsets
pytest -m 'tempo2 or dev_oracle' --no-cov -q   # tempo2 stack only (~8.5 min, 261 tests)
pytest -m 'not dev_oracle' -q                  # JUG-only CI (no libstempo)
```

65-pulsar IPTA campaigns remain explicit-request (each pulsar runs in seconds now).

---

## JAX native delay chain

**Goal:** JAX code reproducing libstempo/tempo2 `updateBatsAll` → `calculate_bclt` →
`formBats` → `formResiduals` on the same per-TOA epochs.

| Layer | JUG production | tempo2 native (JAX path) | Dev-oracle gap |
|-------|----------------|---------------------------|----------------|
| Clock / delays | IFTE `model_mjd` + `prebinary` (host) | Split formBats + compensated daysec (JAX) | Host model-epoch `batCorr` ~272 ns (pinned); JAX `bat_corr_days` < 1 ns |
| Spin | Taylor on `dt_sec_ld` + TRACK −2 | `phase2+phase3` at `bbat` (diagnostic tail) | **~1.2 ns** host residuals on gated fixtures |
| End-to-end residuals | libstempo oracle | `psr.residuals()` | **< 7 ns** on gated fixtures |

Scaffold: `jug/residuals/tempo2/`. Graph mode: `tempo2_native` kwarg (default `staged_bclt`).

### Dev-oracle gates (component-level)

| Gate | Current (wsrt167) | Target | Notes |
|------|-------------------|--------|-------|
| `bat_corr_days` vs pytempo | **< 1 ns** RMS | < 1 ns | **Green** — `test_tempo2_formbats_closure.py` |
| JAX `bbat_mjd` vs pytempo | **< 1 ns** RMS | < 1 ns | **Green** — split formBats + two-part daysec |
| Bundled `bat_mjd` vs libstempo | **< 1 µs** RMS | < 1 µs | **Green** — `test_tempo2_batcorr_from_model_probe.py` |
| `torb_sec` (JAX) vs pytempo | **~0.17 ns** RMS | < 1 ns | **Green** — host `torb_binary_sec` (prebinary − total) |
| Host `term_diagnostics['bbat_mjd']` vs pytempo | ~350 ns RMS | informational | Taylor host path only; pinned in `test_tempo2_j0613_delay_terms.py` |
| Model-epoch `batCorr` vs lib | ~272 ns RMS | < 500 ns (pin) | IFTE diagnostic; not production JAX path |
| Stripped lite BBAT vs pytempo | **< 1 ns** RMS | < 1 ns | **Green** — `test_tempo2_stripped_mode.py` |
| Full residuals wsrt167 | **1.19 ns** | < 5 ns | **Green** |

Component gaps can coexist with green end-to-end residuals when Taylor spin at
`model_mjd` absorbs epoch-scalar offsets.

### formBats MJD assembly

**Do not confuse delay parity with MJD epoch parity on the host Taylor path.**

| Quantity | JAX / pytempo (wsrt167) | Host Taylor diagnostic | What it tests |
|----------|-------------------------|------------------------|---------------|
| `bat_corr_days` | **< 1 ns** | same | Delay-component closure |
| JAX `bbat_mjd` | **< 1 ns** | ~350 ns vs pytempo | Assembled MJD scalar |
| End-to-end residuals | **~1.2 ns** | same | Production gate |

tempo2 `formBats.C` splits the UTC→TT term in `long double`:

```c
batCorr = getCorrectionTT(obsn)/SECDAY
        + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
```

JUG JAX implements this split in `compute_formbats_daysec` (`formbats_jax.py`):
TT seconds are added in a first `add_seconds_daysec` pass; remaining terms in a
second pass; `bat_corr_day` uses `two_sum(tt_day, other_day)`. Compensated
two-part `(int_day, sec_in_day)` arithmetic (`compensated.py`) closes JAX
`bbat_mjd` vs pytempo to sub-ns on wsrt167.

The **host Taylor** path still exports a single-float64 `term_diagnostics['bbat_mjd']`
from `tempo2_clock_terms` (~350 ns vs pytempo). That scalar is **not** the JAX
production bbat and does not affect gated host residuals or autodiff tangents.

**Gate on:** JAX `bat_corr_days`, JAX `bbat_mjd`, and end-to-end
`acceptance_residual_sec`. Host diagnostic `bbat_mjd` debt is pinned, not a
residual blocker.

### Graph modes (`staged_bclt`, `fixed_state_bclt`, `fixed_state_stripped`, `full`)

| Layer | `staged_bclt` (default) | `fixed_state_bclt` | `fixed_state_stripped` | `full` |
|-------|-------------------------|--------------------|------------------------|--------|
| Ephemeris / clocks / obs vectors | Host-frozen from `tempo2_obs_state` | Same | Same | In XLA |
| BCLT iteration | **Recomputed in JAX** (`lax.scan` × 12/TOA) | **One-pass** at frozen `dt_ssb_ref` | **One-pass** at frozen `dt_ssb_ref` | Full chain |
| formBats / Shklovskii | In JAX | In JAX (full tail) | BBAT lite only | In XLA |
| BBAT / tail eval | Ref + pert full tail | Ref + pert full tail | **Single pert** vs lite-built ref BBAT | Full chain |
| Spin / phase5 | In JAX tail | In JAX tail | Omitted (Taylor wrapper only) | In XLA |

`NativeDeltaPack` with `mode=staged_bclt` freezes **host inputs**, not BCLT output.
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
| **T4** | `fixed_state_stripped` for NUTS after envelope; `fixed_state_bclt` as envelope reference | Drops BCLT scan + full tail | vs `fixed_state_bclt` / `staged_bclt` < 1 ns |
| **T5** | Do not trim compensated two-part day/sec without envelope | — | Required for TDB parity |
| **T6** | XLA cache, mini fixtures in dev loop | Amortize compile | Operational |

**Do not trim:** host-frozen ephemeris/clocks; BCLT in `staged_bclt` (use
`fixed_state_bclt` or `fixed_state_stripped` to freeze); Taylor `_phase_residual_delta_jax` wrapper.

**Warm-cache JIT (wsrt167, 167 TOAs, 4-param residual+jacobian):**

| `tempo2_native` | Sum JIT time | vs stripped |
|-----------------|--------------|-------------|
| `staged_bclt` | ~1.5 s | 2.6× slower |
| `fixed_state_bclt` | ~1.0 s | 1.7× slower |
| `fixed_state_stripped` | ~0.6 s | 1.0× |

Harness: `tools/run_tempo2_graph_timing_wsrt167.py`. Dominant pytest cost is cold
host `compute_residuals` (~60 s); amortize via `wsrt167_fit_setup_factory` in
`tests/conftest.py`.

**Implementation anchors:**

| Component | Location |
|-----------|----------|
| JIT residual factory | `jug/fitting/jax_residual_delta.py` |
| Bbat delay change | `jug/residuals/tempo2/terms.py` |
| Full tail (trim candidate) | `jug/residuals/tempo2/model/full.py` |
| BCLT scan | `jug/residuals/tempo2/calculate_bclt_jax.py` |
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

Naïve `sat + bat_corr → bat` float64 recomposition is an **assembly-recipe mismatch**,
not a delay disagreement (historically ~304 ns on wsrt167 before split formBats).
The JAX path now closes pytempo to < 1 ns; use `bat_corr_days` for delay physics
and JAX `bbat_mjd` for epoch-scalar gates.

### Design constraints (do not retry)

| Approach | Outcome | Why |
|----------|---------|-----|
| Wire `phase5@bbat` to production host spin | ~17 ns | Worse than Taylor ~1.2 ns |
| Taylor `dt_sec` closure for `torb_sec` diagnostic | ~60 ns vs pytempo | Use host `torb_binary_sec` (prebinary − total); fitting tangent unchanged |
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
