# Tempo2 parity — status, gaps, and work queue

Living route for JUG `compatibility="tempo2"` parity: measured debt, gap scorecard,
**pytempo diagnostic workflow**, active work queue, and investigation log.

**Policy and architecture:** [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md)

**Status (2026-07-05):** Cases A/B/C green (~1–2 ns). IPTA DR2 workloads **partially
green**. Use **pytempo** for all new term-by-term debugging.

---

## 0. pytempo diagnostic workflow

[`ref-packages/pytempo`](../../pytempo) is the **primary per-TOA oracle** for parity
debugging. It exposes tempo2 `obsn[]` fields after `updateBats` / `formResiduals` that
libstempo properties alone do not surface (`bbat_mjd`, `nphase`, `phase_offset_turns`,
etc.). **68 tests pass** on the bundled J1909 fixture.

### Setup

```bash
pip install -e ref-packages/pytempo   # requires $TEMPO2 runtime
```

### Standard comparison loop

1. Load fixture par/tim (start **wsrt167** ad hoc subset, then `epta_j0613_t2_nrt1400`,
   then full EPTA).
2. Oracle: `diag = pytempo.sandbox.tempopulsar(par, tim, dofit=False).toa_diagnostics(removemean=False)`
3. JUG: `jug = compute_residuals_simple(par, tim, compatibility="tempo2", verbose=False)`
4. Compare per-TOA (examples):

   | pytempo field | JUG field / note |
   |---------------|------------------|
   | `bbat_mjd` | `jug["bbat_mjd"]` |
   | `roemer_sec`, `torb_sec` | `term_diagnostics` / top-level keys |
   | `phase_offset_turns` | `jump_phase` / tim `-padd` |
   | `nphase`, `phase_turns` | TRACK −2 path in `compute_phase_residuals` |
   | `residual_sec` | `jug["residuals_us"]` × 1e−6 |

5. Rank largest term delta → fix JUG → add pytest debt pin → re-run.

Use `removemean=False` on pytempo for term dumps. Use raw residuals for acceptance gates
(via libstempo `tempo2_reference`, not pytempo mean-subtraction settings).

Full API: [`pytempo/README.md`](../../pytempo/README.md).

### Future code (not yet implemented)

- Rewire `jug/testing/tempo2_diagnostics.py` to call pytempo instead of thin libstempo
  properties.
- Add `wsrt167` fixture under `tests/data_tempo2/` with pytempo-vs-JUG pytest gate.
- Extend Phase A CLI to default to pytempo oracle.

Current Phase A still uses libstempo properties in `tempo2_diagnostics.py`.

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
| **WSRT167 isolated** (ad hoc) | 167 | **1.06 ns** | 3.2 ns | ~2 ns (proposed) | not in CI |

### `-addsat` / tempo2 spin

| Item | Status |
|------|--------|
| `-addsat` regression (idx 247/256/561) | **Fixed** — each **< 1 µs** vs libstempo |
| Full `tempo2_spin=True` at `bbat` | **Not enabled** — ~1.5 ms RMS on full mix; debug on clean subsets first |
| Raw `phase5(bbat)−phase5(bbat−addsat)` wrap on legacy TRACK −2 | **Wrong** — ~67 µs at idx 247; do not use |
| `addsat_track2_turn_delta` int(F0) closure | **In use** — calibrated constants; not yet derived from `ff0` alone |
| `bbat_mjd` / `torb_sec` in JUG output | **Done** |
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
| **G7** EPTA multi-backend | **Open (improved)** | ~608 ns bulk after integer-turn and `-addsat` fixes |

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
| PPTA native J0613 | No | ~1.4 ns ad hoc |
| PPTA alternate export | No | ~16 ns ad hoc |
| ELL1/T2 binaries | Partial | Autodiff green; `ppta_j1741_ell1` debt |

### G6 — documented residual debt

- **`ppta_j1741_ell1`:** RMS ~5–8 ns vs strict 5 ns gate; ELL1 convention mismatch.
- **`DM_SERIES`:** ignored by JUG (warn-only).

### G7 — EPTA multi-backend (~608 ns)

Integer-turn and per-TOA `-addsat` debt **closed** (2026-07-04). Remaining ~608 ns is
**bulk pulse-phase / spin bookkeeping** across the multi-backend mix — not Roemer, DM, or
binary delay kernels at ms scale. Production uses emission-Taylor spin (`tempo2_spin=False`).

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

All items assume **pytempo term dumps** as the primary debug loop. Items marked
**(future code)** are documented follow-on — not in scope of the doc consolidation pass.

| Priority | Task | pytempo fields | Status |
|----------|------|----------------|--------|
| **0** | Rewire Phase A / `tempo2_diagnostics.py` to pytempo | full `toa_diagnostics` | **(future code)** |
| **1** | Promote `wsrt167` fixture + pytest (~2 ns gate) | `phase_offset_turns`, `nphase`, `phase_turns`, `residual_sec`, `bbat_mjd` | **(future code)** — ad hoc at 1.06 ns today |
| **2** | `-padd` placement vs TRACK −2 | `phase_offset_turns` vs JUG `jump_phase`; tempo2 order: spin → `-padd` → `phas1` → `nlong` | **(future code)** |
| **3** | 328 vs 382 band split (~0.4 ns) | group-wise `nphase`, `phase_turns`, `phase_offset_turns` | **(future code)** |
| **4** | Full-mix +1.29 ns mean anchor | same TOAs: pytempo full vs isolated tim; `removemean=True/False` | **(future code)** |
| **5** | Validate on `epta_j0613_t2_nrt1400` (~62 ns) | same fields; gate before `tempo2_spin=True` | **(future code)** |
| **6** | Alternate PPTA ~16 ns | Roemer/Shapiro + TZR at `TZRMJD` | **(future code)** |
| **7** | Fitter TRACK −2 / `-addsat` wiring | after wsrt167 green | **(future code)** |
| **Defer** | `tempo2_spin=True` | until nrt1400 + wsrt167 pytempo spin fields match | — |

### WSRT167 findings (2026-07-05, ad hoc)

167-TOA WSRT low-band subset (328/382 MHz) from `epta_j0613_t2_ipta_all`:

| Measurement | Isolated sub-tim | Same TOAs in full 1369-TOA mix |
|-------------|------------------|--------------------------------|
| RMS vs libstempo | **1.056 ns** | **1.665 ns** |
| Mean Δ | **0 ns** | **+1.287 ns** (uniform shift) |

- **`-padd` / `jump_phase`:** required; without it RMS → ~6 ns. `jump_phase` correlates
  1:1 with tim `-padd` flags.
- **328 vs 382 bands:** group mean offsets +0.17…+0.20 ns vs −0.18…−0.27 ns (~0.4 ns
  split); `-padd` differs by 0.0037 turns — split is TRACK −2 / `-pn` interaction, not
  padd magnitude alone.
- **`tempo2_spin` swap:** not correlated with ~1 ns isolated debt (`corr ≈ 0`).

Proposed fixture source: filter WSRT site, freq < 1000 MHz from EPTA J0613 par/tim.

---

## 4. J0613 three parity budgets

Closing one budget does not automatically close the others.

### 1. EPTA full multi-backend (~608 ns) — dominant

Integer-turn and `-addsat` debt closed. Remaining bulk is pulse-phase / spin bookkeeping
across multi-backend mix. Decomposed via wsrt167 (~1 ns units) + mean-anchor shifts.

**Focus (future code):** `-padd` vs TRACK −2 placement; full-mix mean subtraction anchor;
optional `tempo2_spin` only after subset gates pass.

### 2. Alternate PPTA export (~16 ns)

Native PPTA already ~1.4 ns. ~16 ns appears only with **full alternate-export par** +
tim (TDB, EPTA-reference T2/astrometry on PPTA TOAs) — not from `-pn` format alone.

**Focus (future code):** TDB Roemer+Shapiro; TZR at `TZRMJD` / `TZRSITE pks`.

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
| `TRACK -2` | `model.phase(abs) − pn` | `pnNew` vs `-pn`; `phas1` at tim index 0 | `phas1@tim[0]` + `fortran_nlong` per TOA (2026-07-03) |
| Mean removal | Weighted | Unweighted | Mode-dependent |
| Delay path split? | — | — | **Yes** (providers differ) |
| Phase path split? | — | — | **No** (shared `compute_phase_residuals`) |

### tempo2 `formResiduals.C` order (TRACK −2)

1. Compute spin phase → `phase5`
2. Add tim **`-padd`** to `phase5`
3. `phas1 = fortran_mod(phase5[tim_index_0], 1.0)`; subtract from all TOAs
4. `nphase = fortran_nlong(phase5[i])`; fractional residual = `phase5 − nphase`
5. Optional `addPhase` from `pnNew` vs `-pn` flags

### JUG shipping path (TRACK −2)

Emission-time Taylor spin → add `jump_phase` (includes `-padd`/`-radd`) → `phas1@tim[0]`
→ `fortran_nlong` per TOA → legacy `add_phase = −pn_add` (not full tempo2 `pnNew`).

Spin at `bbat` (`tempo2_spin=True`) is scaffolded but **not production**.

### Subset pitfall

`-pn` flags are relative to **full-tim** `obsn[0]`, not isolated sub-tim index 0.
Filtering to a sub-tim without regenerating `-pn` produces comparison artefacts.

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

### What was fixed

1. **TRACK −2 anchor (2026-07-03):** JUG uses `phas1@tim index 0` + per-TOA
   `fortran_nlong`, not `argsort(dt)[0]` + `base_pn + (-pn)`.
2. **±326-turn EFF TOAs (2026-07-03):** eight TOAs on EFF.EBPP.1360/2639 with integer
   errors of ±(int)F0 turns — closed by TRACK −2 fix.
3. **Per-TOA `-addsat` (2026-07-04):** `addsat_track2_turn_delta` closes idx 247/256/561
   (~68 µs → < 1 µs).

### What was ruled out

- Roemer/DM/binary kernels as ms-scale drivers on full mix (post TRACK −2 fix).
- Clock dir JUG vs `$TEMPO2/clock` swap — no RMS change.
- BIPM2011/2019/2024 CLK sweep — no change.
- Full `pnNew`/`bbat` rewrite without correct spin decomposition — catastrophic (~10¹¹ ms).

### EFF backend notes

| `-sys` | n | Isolated RMS (TRACK −2) |
|--------|---|-------------------------|
| EFF.EBPP.1410 | 241 | **~69 ns** |
| EFF.EBPP.1360 | 42 | was ±326-turn outliers; now sub-µs on bad TOAs |
| EFF.EBPP.2639 | 64 | same |
| JBO / NRT / WSRT (isolated) | — | sub-µs to ~1 µs |

Non-EFF backends match libstempo in isolation but show ~0.73 ms offsets in the **full mix**
when EFF 1360/2639 present — cross-backend `phas1`/anchor coupling (historical).

---

## 7. Other open items

- **G4:** deprecate or repair analytic tempo2 columns; autodiff is supported path.
- **G6:** narrow `ppta_j1741_ell1`; implement or reject `DM_SERIES`.
- **G2 residual:** θ≠0 NumPy/JAX on IPTA workloads.
- **pytest debt pins (future code):** native PPTA (~1.4 ns), alternate PPTA (~16 ns), wsrt167 (~1 ns).

---

## 8. Investigation log

| Date | Summary |
|------|---------|
| 2026-06-01 | Phases A–E: native TDB geometry, TZR, Cases B/C green, design-matrix/fit parity |
| 2026-07-03 | G1/G2 primary closed; J0613 fixtures get TRACK −2 + `-pn`; TRACK −2 anchor fix (~46.8 ms → ~2 µs) |
| 2026-07-04 | Per-TOA `-addsat` int(F0) coupling; EPTA RMS **~608 ns**; PPTA native ~1.4 ns, alternate ~16 ns measured |
| 2026-07-04 | `tempo2_spin=True` attempt wrong for legacy stack; production restored via `addsat_track2_turn_delta` |
| 2026-07-05 | **pytempo ready:** 68 tests; `nphase`, `phase_offset_turns` on J1909; documented as primary diagnostic oracle |
| 2026-07-05 | **WSRT167 ad hoc:** 167 TOA WSRT low-band; **1.056 ns** isolated; **+1.287 ns** uniform mean shift in full mix; padd/jump_phase required; 328/382 band split ~0.4 ns |
| 2026-07-05 | Doc consolidation: three parity markdown files merged into this doc + [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md) |

**When updating parity status:** add a fixture + pytest gate **first**, then update this log.
Ad hoc runs document status between gates but do not substitute for CI.

---

## 9. Commands and tests

```bash
# pytempo (diagnostic oracle)
cd ref-packages/pytempo && pip install -e . && python -m pytest tests -q

# JUG tempo2 parity (acceptance oracle = libstempo)
cd ref-packages/jug
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_*.py -q -o addopts=''

# J0613 EPTA debt pins
python -m pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -o addopts=

# Phase A term ranking (legacy libstempo oracle today)
python tools/run_phase_a_diagnostics.py --output /tmp/phase_a_report.json
```

Key modules:

| Module | Role |
|--------|------|
| `jug/residuals/simple_calculator.py` | JUG residuals under test |
| `jug/testing/tempo2_reference.py` | libstempo acceptance oracle |
| `jug/testing/tempo2_diagnostics.py` | Legacy term oracle (target: pytempo) |
| `jug/testing/phase_a_comparison.py` | Phase A ranking |
| `ref-packages/pytempo` | **Primary diagnostic oracle** |
