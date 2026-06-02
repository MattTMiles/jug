# Tempo2 compatibility mode — project brief

This document is the implementation brief for `compatibility="tempo2"` in JUG. It
records investigation findings, locked product decisions, measured parity gaps,
code audit results, and the phased work plan.

**Status (2026-06-01):** Case A (TCB regression fixtures) and Cases B/C (NG5 TDB) are
**green** (~1.3 ns raw WRMS vs libstempo on B/C). Phase B split the tempo2 TDB path
from the pint path: jplephem DE405 SPK for sun/planet vectors, separate provider
modules, and ``EngineConventionProfile`` for runtime conventions.

**Next step:** Phase D/E (CI consolidation, design matrix/fit on TDB). Phase C TZR
(mode-specific native path) is implemented. §3 decisions are locked.

---

## 1. Context

The MetaPulsar notebook `examples/notebooks-dev/demo-jug-like-libstempo.ipynb`
compares four backends on a harmonized NG5 dataset. Residual-delta panels
involving libstempo show a visible ~few-ns annual pattern, while pint-family
backend pairs can agree at sub-ns **only after a weighted mean is removed** —
which is misleading for tempo2 acceptance (see §3 and §5).

| Backend | Role |
|---------|------|
| **libstempo** | tempo2 reference (`tempopulsar`) |
| **JUG (`compatibility="tempo2"`)** | Intended tempo2-equivalent JUG path |
| **JUG (`compatibility="pint"`)** | PINT-equivalent JUG path |
| **PINT** | Native PINT reference |

### Fixture matrix (do not conflate these)

| Case | Par | Purpose |
|------|-----|---------|
| **A. TCB regression fixtures** | `tests/data_tempo2/*` (`UNITS=TCB`, IF99, DILATEFREQ, equatorial) | Existing green tempo2 parity tests |
| **B. NG5 equatorial TDB** | Original NG5 J1600 after `T2CMETHOD` removal only | MetaPulsar regression anchor (~6 ns PINT vs tempo2 floor) |
| **C. NG5 ecliptic cross-engine** | Converted par below (Layer-B harmonization) | Notebook / cross-engine demo |

Case **C** is a harmonized artifact built for MetaPulsar cross-engine work.
Case **B** is the simpler canonical TDB anchor. Both must be tracked; fixing
only C risks overfitting to the conversion pipeline.

**Case C dataset** (intended for the notebook; commit under MetaPulsar when available):

- Par: `examples/notebooks-dev/data/demo-jug-like-libstempo/J1600-3053_NANOGrav_dfg+12_cross_engine.par`
- Tim: `examples/notebooks-dev/data/demo-jug-like-libstempo/J1600-3053_NANOGrav_dfg+12.tim` (625 TOAs)

Built from the NG5 equatorial source by:

1. Converting `RAJ`/`DECJ`/`PMRA`/`PMDEC` → `LAMBDA`/`BETA`/`PMLAMBDA`/`PMBETA` (IERS2003 obliquity),
2. Setting `ECL IERS2003`,
3. Removing `T2CMETHOD TEMPO` (MetaPulsar Layer-B harmonization).

Key par keywords (case C):

```text
PSRJ 1600-3053
LAMBDA / BETA / PMLAMBDA / PMBETA   (ecliptic astrometry)
ECL IERS2003                        (ignored by tempo2; hard-coded IERS2003 ecliptic math)
BINARY DD
TZRMJD / TZRFRQ / TZRSITE gbt       (absolute phase)
UNITS TDB
CLK TT(BIPM2011)
EPHEM DE405
CORRECT_TROPOSPHERE N
(no TIMEEPH, no DILATEFREQ, no T2CMETHOD)
```

The notebook delta plots subtract a **weighted mean** (weights = 1/σ²) before
display. That is appropriate for comparing PINT-family backends to each other,
but **not** for assessing tempo2 compatibility (see §3 and §8).

---

## 2. Executive summary

| Area | Finding |
|------|---------|
| **Bottom line** | `compatibility="tempo2"` does **not** reproduce tempo2 residuals on TDB models today. It tracks Astropy ephemeris + JUG combined delay kernel (PINT family). TCB parity machinery (IF99 epoch map, IFTE, unweighted phase mean) explains green case A tests only. |
| **Real gap vs libstempo (NG5 TDB)** | ~6 ns raw WRMS, annual-shaped — same cluster as PINT, not acceptable for JUG(tempo2) acceptance. |
| **Misleading metric** | Weighted-mean-centered JUG(tempo2)−JUG(pint) ≈ 0.002 ns hides ~61 ns phase-convention offset and identical delay physics. |
| **Root cause** | Missing tempo2-native ephemeris/Roemer/Shapiro on TDB; TZR not mode-specific; `_extract_binary_params` ignores `compatibility`. |
| **Tests** | `tests/test_tempo2_residual_parity.py`: 8 passed, 1 xfailed (pint-default TCB baseline). No case B/C fixtures in `tests/data_tempo2/manifest.json` yet. |

---

## 3. Decisions (locked — do not reopen without explicit review)

All product choices for `compatibility="tempo2"` are decided here. §8 lists
**investigation priority** for Phase A only; it is not a second decisions table.

| Question | Decision |
|----------|----------|
| What does `compatibility="tempo2"` mean? | Match tempo2 **residuals and phase conventions end-to-end**, not isolated delay-term tweaks or post-hoc centering tricks. |
| Parity metric for tempo2 mode | **Raw pre-fit residuals** vs libstempo — same gate as `tests/test_tempo2_residual_parity.py` (RMS, p99, max, WRMS on uncentered δ). **Do not** subtract a weighted (or any other) mean for tempo2 acceptance. |
| Phase / mean subtraction | tempo2 uses an **unweighted** phase offset; pint mode uses **weighted**. JUG(tempo2) must apply tempo2 phase semantics internally; parity compares residuals **as returned**, with no extra mean removal. |
| Implementation strategy | **Native only.** Reimplement tempo2-equivalent physics inside JUG. **Do not** wrap tempo2, libstempo, or tempo2 plugins for geometry, delays, clocks, ephemeris, TZR, or fitting — at runtime or as a fallback. |
| Test oracle vs runtime | **libstempo** (via `jug/testing/tempo2_reference.py`) remains the pytest oracle for raw δ only. Oracle use does not permit wrapping tempo2 in the JUG(tempo2) code path. |
| Shared PINT-family stack in tempo2 mode | On TDB, tempo2 mode must **not** rely on the pint-mode delay pipeline (Astropy ephemeris + JUG combined delay kernel + PINT-like troposphere/FD conventions) for terms tempo2 implements differently. Phase A ranks **which native ports come first**; Phase A does **not** choose wrap vs native. |
| Ephemeris / Roemer / Shapiro | In tempo2 mode, use **tempo2-equivalent native** table integration and delay geometry (e.g. bundled DE405 path, tempo2 Roemer/Shapiro conventions). Matching the `EPHEM` keyword alone is insufficient. |
| Omitted par keywords on TDB (`TIMEEPH`, `DILATEFREQ`, etc.) | Follow **tempo2 implicit defaults** for the par as loaded into libstempo, not PINT defaults. Document each mapping when implemented. |
| TZR / absolute phase | TZR handling is **mode-specific**: tempo2 mode uses tempo2-equivalent native TZR geometry and clocks; pint mode keeps the pint path. Do not share a single TZR geometry path across compatibility modes. |
| Demo / notebook display | **Raw δ** for any panel labeled tempo2 compatibility. **Weighted-mean-centered δ** only when comparing pint-family backends to each other. |
| Canonical TDB fixtures | **Both** case B (equatorial NG5) and case C (ecliptic cross-engine), plus existing case A TCB fixtures must stay green. |
| PINT vs tempo2 cross-engine floor | Closing PINT↔tempo2 gaps inside PINT is **out of scope**. JUG(tempo2) must match libstempo on cases A–C regardless of where PINT sits. |

---

## 4. Measured residual deltas

### 4.1 Case C — NG5 cross-engine par (original notebook run, 2026-06-01)

#### Reproducibility

| Component | Version / note |
|-----------|----------------|
| Environment | `/opt/venvs/pta` (Python 3.12) |
| JUG | `ref-packages/jug` branch `tempo2-compat` @ `79ac1c5` (update via `git rev-parse HEAD`) |
| libstempo / tempo2 | conda-forge build in devcontainer; `TEMPO2` runtime required |
| PINT | ~1.1.4 in pta venv (DE405 + BIPM2011 on main TOAs, BIPM2023 on auxiliary TZR TOA in logs) |

Re-run from MetaPulsar repo root (requires case C par/tim on disk):

```python
import sys
import numpy as np

sys.path.insert(0, "examples/notebooks-dev")
sys.path.insert(0, "ref-packages/jug")
from demo_jug_like_libstempo_lib import load_backends, get_residuals_us, DEMO_NG5_DATA_DIR

par = DEMO_NG5_DATA_DIR / "J1600-3053_NANOGrav_dfg+12_cross_engine.par"
tim = DEMO_NG5_DATA_DIR / "J1600-3053_NANOGrav_dfg+12.tim"
bundle = load_backends(par, tim, harmonize_cross_engine=False)

ls = np.asarray(bundle.libstempo.residuals(), dtype=np.float64) * 1e6
err = np.asarray(bundle.libstempo.toaerrs, dtype=np.float64)
jt2 = get_residuals_us(bundle, "jug_t2")
jp = get_residuals_us(bundle, "jug_pint")
pr = get_residuals_us(bundle, "pint")

def raw_wrms_ns(a, b):
    return float(np.sqrt(np.mean(((a - b) * 1000) ** 2)))

def weighted_centered_wrms_ns(a, b):
    d = a - b
    d -= np.average(d, weights=1 / err**2)
    return float(np.sqrt(np.mean((d * 1000) ** 2)))
```

#### Raw deltas (correct tempo2 parity view)

Each backend returns residuals with its own internal phase/mean convention applied.
Compare **directly** — no post-hoc weighted-mean removal.

| Comparison | Raw WRMS | Notes |
|------------|---------:|-------|
| JUG(tempo2) − libstempo | **~6.0 ns** | Structured; ~3.4 ns annual in sin/cos fit |
| JUG(pint) − libstempo | **~6.0 ns** | Same shape as above |
| PINT − libstempo | **~6.0 ns** | PINT vs tempo2 cross-engine floor |
| JUG(tempo2) − JUG(pint) | **~61 ns** | Dominated by weighted vs unweighted phase mean |
| JUG(tempo2) − PINT | **~6.0 ns** | Same as JUG(tempo2) − libstempo (PINT-family cluster) |

The ~61 ns JUG(tempo2) − JUG(pint) offset is **expected** until pint mode is
compared on its own terms. It is not evidence that tempo2 mode is “almost
right”; the ~6 ns structured mismatch vs libstempo is the real gap.

#### Weighted-mean-centered deltas (notebook view only)

If a weighted mean is removed **after the fact** (as the demo notebook does):

| Comparison | Centered WRMS |
|------------|--------------:|
| JUG(tempo2) − libstempo | ~6.0 ns |
| JUG(tempo2) − JUG(pint) | ~0.002 ns |
| JUG(pint) − PINT | ~0.02 ns |

This centered view hides the phase-convention offset between tempo2 and pint
modes and makes JUG(tempo2) look identical to JUG(pint). **Do not use it as
the tempo2 compatibility acceptance metric.**

#### Interpretation (case C)

```text
tempo2 reference (libstempo)   vs   {PINT, JUG(tempo2), JUG(pint)} (PINT family)
         ~6 ns structured mismatch (raw residuals)

{PINT, JUG(tempo2), JUG(pint)} among themselves
         ~6 ns vs libstempo; ~61 ns tempo2↔pint offset from phase mean convention
```

On TDB, `compatibility="tempo2"` was meant to match libstempo at raw residual
level but still sits in the PINT-family delay pipeline. The ~6 ns annual
structure matches the documented **PINT vs tempo2 geometric delay floor** on
NG5-class data. That floor is acceptable for MetaPulsar **combination**
diagnostics, but **not** for JUG(tempo2) acceptance.

### 4.2 Case A — TCB fixtures (verified 2026-06-01, jug repo)

Command:

```bash
cd ref-packages/jug
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_residual_parity.py -q -o addopts=''
```

Result: **8 passed, 1 xfailed** (`test_pint_default_baseline_vs_tempo2_isolated` — intentional).

Independent raw-δ measurements (libstempo oracle, `PYTHONPATH=tests:.`):

| Fixture | JUG(pint) − libstempo | JUG(tempo2) − libstempo | JUG(tempo2) − JUG(pint) |
|---------|----------------------:|------------------------:|------------------------:|
| `epta_j0030_isolated` | 2766 ns | **1.13 ns** | 2766 ns |
| `epta_j1909_t2` | 46.5 ns | **1.90 ns** | 46.5 ns |
| `epta_j1918_ddh` | 6201 ns | **1.44 ns** | 6201 ns |

On TCB, pint vs tempo2 **delay physics differ strongly** (multi-µs Roemer/total-delay
deltas between modes). Tempo2 mode reaches sub-ns parity vs libstempo.

### 4.3 Other local runs (sanity / pitfalls)

| Dataset | JUG(tempo2)−libstempo | JUG(pint)−libstempo | JUG(t2)−JUG(pint) raw | Notes |
|---------|----------------------:|--------------------:|----------------------:|-------|
| `tests/data_mpta/j1909_t2/J1909-3744_tdb.par` | **1.67 ns** | 25.2 ns | 25.2 ns | Par says **`UNITS TCB`** (filename “tdb” is misleading). Tempo2 TCB paths active. |
| `tests/data_golden/J1909_proper.par` (100 TOAs, `UNITS TDB`) | 74.3 ns | 76.4 ns | 17.5 ns | corr(δ_t2−ls, δ_pint−ls) ≈ **1.0** — same delay-shaped gap vs libstempo; not NG5-like. |
| NG11 J1600 ecliptic TDB (500 TOAs, ad hoc) | ~6.6 ms | ~6.6 ms | 494 µs / **0.003 ns centered** | Invalid libstempo oracle setup; illustrates metric trap only. |

**Case B/C NG5 files** were **not present** in the MetaPulsar workspace during the
2026-06-01 verification pass; §4.1 numbers remain from the original notebook run.
Commit case B/C par/tim before Phase A automation.

### 4.4 Delay-level diagnostic (pint vs tempo2 modes)

When `max|total_delay_sec(pint) − total_delay_sec(tempo2)|` is negligible on TDB
ecliptic models, residual differences vs libstempo are **highly correlated**
between modes (corr ≈ 1), and centered JUG(t2)−JUG(pint) collapses to ~0 ns while
both remain ~6 ns vs libstempo. **Phase mean convention is implemented; delay
physics is not yet tempo2-native on TDB.**

On TCB (`epta_j0030_isolated`), `max|roemer_shapiro|` pint−tempo2 ≈ **7722 ns**;
TCB machinery separates modes correctly.

---

## 5. Why existing tempo2 tests pass but NG5 TDB fails

JUG tempo2 parity tests today:

- `tests/test_tempo2_residual_parity.py` — raw pre-fit residuals vs libstempo (< 5 ns RMS gate)
- `tests/test_tempo2_designmatrix_parity.py` — TCB fixtures only
- `tests/test_tempo2_fit_parity.py`
- Fixtures: `tests/data_tempo2/` (case A only; `tests/data_tempo2/manifest.json`)

Case A example (`epta_j0030_isolated`):

```text
UNITS TCB
TIMEEPH IF99
DILATEFREQ Y
T2CMETHOD IAU2000B
EPHEM DE440
RAJ / DECJ  (equatorial)
```

**`compatibility="tempo2"` works on case A** because TCB-specific paths activate:

- `convert_tdb_epoch_to_tempo2_tcb` on `model_mjd` when `UNITS=TCB`
- IFTE scaling (`IFTE_K`) on SSB/sun/planet vectors
- Unweighted phase mean in `compute_phase_residuals`

On TDB cases B/C, those paths are inactive; tempo2 mode still shares Astropy
geometry and the PINT-family delay kernel → ~6 ns vs libstempo.

### Known secondary gaps (case A and beyond)

| Item | Status |
|------|--------|
| `ppta_j1741_ell1` | Documented in tests: RMS ~5–8 ns vs strict 5 ns gate; orbital-harmonic structure → ELL1 convention mismatch |
| `DM_SERIES` par keyword | Warned as **ignored by JUG** on several fixtures |
| Design matrix / fit on new TDB fixtures | Phase E — after raw residuals match |
| `fingerprint.py` | Allows TCB/TDB in tempo2 mode; TDB not enforced by pytest yet |

---

## 6. Implementation audit (`compatibility="tempo2"`)

Primary file: `jug/residuals/simple_calculator.py` (`compute_residuals_simple`).

### What tempo2 mode toggles today

| Feature | Condition | Pint mode | Tempo2 mode |
|---------|-----------|-----------|-------------|
| Model epoch times | TDB par | TDB MJD from clock chain | Same |
| TCB epoch map | `UNITS=TCB` | — | `convert_tdb_epoch_to_tempo2_tcb` on `model_mjd` |
| Native ecliptic Roemer | `_ecliptic_coords` | Equatorial `L_hat` from converted RAJ/DECJ | `compute_ecliptic_pulsar_direction` + rotate SSB vectors (**runs on TDB too**) |
| IFTE position scaling | `UNITS=TCB` | — | Multiply SSB/sun/planet vectors by `IFTE_K` |
| Phase mean subtraction | always | Weighted (`mean_mode="weighted"`) | Unweighted (`mean_mode="unweighted"`) |
| Binary dispatch | always | `_extract_binary_params(..., compatibility=...)` | **Same code path** — `compatibility` parameter is **unused** inside `_extract_binary_params` |

### TDB: effective difference between modes

| Layer | Pint | Tempo2 (today) |
|-------|------|----------------|
| Astropy SSB/planet geometry | yes | yes |
| JAX `compute_total_delay_jax` | yes | yes |
| TCB epoch map / IFTE | no | no |
| Native ecliptic Roemer | if `_ecliptic_coords` | same |
| Phase mean | weighted | unweighted |

### TZR / absolute phase (`_compute_tzr_phase`)

- `tzr_use_native_ecliptic = bool(params.get('_ecliptic_coords', False))` — **not**
  gated on `compatibility_mode`; pint and tempo2 share the same TZR geometry path.
- IFTE on TZR vectors: only when `UNITS=TCB`.
- Sun/planet at TZR: **Astropy** `get_body_barycentric_posvel`, not tempo2-native tables.

Case C (DD + DMX + `TZRMJD`, `TZRSITE=gbt`) — quantify TZR contribution in Phase A
after Roemer/ephemeris alignment (§8 priority 3).

### Shared PINT-family core (both modes on TDB)

Regardless of compatibility flag, JUG on TDB still uses:

- Astropy solar-system ephemeris for SSB–Earth–planet geometry (`jug/delays/barycentric.py`),
- JUG clock graph + `compute_tdb_standalone_vectorized` for UTC→TDB,
- JAX combined delay kernel (`jug/delays/combined.py`) for DM/binary/tropo/FD,
- PINT-like troposphere / FD / design-matrix conventions in the fitter.

Missing for TDB tempo2 mode:

- Tempo2-internal DE405 (or equivalent) table integration path,
- Tempo2 FB90 / IF99 time-ephemeris behavior where par omits keywords,
- Tempo2-native Roemer integration for `UNITS=TDB`.

### Session API

`TimingSession(..., compatibility="tempo2")` passes the flag through to
`compute_residuals_simple` (`jug/engine/session.py`). KIN/KOM conversion is
skipped in tempo2 mode (tempo2 IAU convention preserved). Plumbing is correct;
TDB tempo2 **delay** physics are not.

---

## 7. Gap breakdown — Phase A investigation order (TDB case B/C)

Priority-ordered **working hypotheses** for the ~6 ns raw JUG(tempo2) − libstempo
mismatch. Phase A measures per-term δ to confirm ordering; §3 fixes native-only
implementation and raw-residual acceptance.

1. **Ephemeris / Roemer backend (leading hypothesis; annual-shaped)**  
   tempo2 uses bundled DE405 + native delay geometry; JUG uses Astropy DE405 +
   explicit ecliptic rotation. Same `EPHEM` keyword does not guarantee identical
   Roemer/Shapiro at the ns level.

2. **TCB parity machinery not extended to TDB**  
   IF99 + IFTE + TCB epoch mapping explain case A success. TDB lacks explicit
   `TIMEEPH` / `DILATEFREQ` / TCB units on NG5 pars — follow tempo2 implicit
   defaults per libstempo oracle when implementing.

3. **TZR / absolute phase (DD + DMX + `TZRMJD`)**  
   Shared pint/tempo2 TZR geometry today; tempo2 AbsPhase on TDB may differ.
   Check after Roemer term alignment.

4. **Ecliptic convention**  
   Likely secondary once ephemeris backend is aligned: par `ECL IERS2003` vs
   tempo2 hard-coded obliquity vs JUG `_ecliptic_frame`.

5. **DD binary + DMX**  
   Lower priority for a clean annual term; case A binaries do not cover DD+DMX on TDB.

6. **Phase mean convention**  
   Implemented (unweighted in tempo2 mode). Remaining work is in **delay terms**,
   not post-hoc centering.

**Phase A method:** For each TOA, compare contributions (UTC→TDB, Roemer, Shapiro,
DM/DMX, binary, TZR). Use libstempo delay/plugin output **for diagnosis only**.
Port the dominant term natively first (§3).

---

## 8. Development goals

### Primary goal

`compute_residuals_simple(..., compatibility="tempo2")` raw pre-fit residuals
match libstempo at the same tolerance as case A:

- RMS δ **< 5 ns**, p99 **< 10 ns**, max **< 25 ns** (`tests/test_tempo2_residual_parity.py`)
- WRMS agreement within the same gate
- **No weighted-mean centering** in comparisons

Target datasets:

- Case B: equatorial NG5 J1600 TDB after `T2CMETHOD` removal
- Case C: ecliptic cross-engine par (625 TOAs, DD, AbsPhase, DMX, DE405, TDB)
- Eventually NG11-scale ecliptic TDB models

### Non-goals

- Closing the PINT vs tempo2 cross-engine floor **inside PINT**.
- Making `compatibility="pint"` match tempo2.

### Success criteria

1. Cases B and C: raw `JUG(tempo2) − libstempo` pass the `< 5 ns` RMS gate.
2. Raw `JUG(tempo2) − JUG(pint)` is **not** ~6 ns with identical shape vs libstempo —
   distinct backends (~61 ns mean offset alone is insufficient; delay physics must
   diverge correctly vs libstempo).
3. Case A TCB fixtures remain green.
4. Case B and/or C in `tests/data_tempo2/manifest.json` + pytest `@pytest.mark.tempo2`.

---

## 9. Work plan

### Phase A — Diagnose term-by-term (required before fixing)

- [x] Commit or symlink case B/C par/tim for CI and local runs
- [x] Per-TOA δ: UTC→TDB, Roemer, Shapiro, DM/DMX, binary, TZR vs libstempo oracle
- [x] Rank native ports; confirm ~6 ns is dominated by geometry vs TZR vs binary

**Phase A workflow (implemented):**

```bash
cd ref-packages/jug

# Provider + schema tests (no libstempo)
pytest tests/test_tempo2_phase_a_diagnostics.py -k "not tempo2" -q -o addopts=''

# Full Phase A on Case B/C (requires libstempo + TEMPO2)
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_phase_a_diagnostics.py -q -o addopts=''

# CLI report with JSON artifact
python tools/run_phase_a_diagnostics.py --output /tmp/phase_a_report.json
```

Key modules:

| Module | Role |
|--------|------|
| `jug/residuals/engine_conventions.py` | Runtime convention profile (Phase B) |
| `jug/residuals/diagnostic_conventions.py` | User-selectable diagnostic conventions |
| `jug/delays/tempo2_ephemeris.py` | jplephem DE405 SPK state vectors |
| `jug/delays/tempo2_geometry.py` | Tempo2 ecliptic / Roemer-Shapiro helpers |
| `jug/residuals/compatibility_providers.py` | Parallel pint/tempo2 delay providers |
| `jug/testing/tempo2_diagnostics.py` | libstempo oracle term extraction |
| `jug/testing/phase_a_comparison.py` | Term ranking and residual deltas |
| `tools/run_phase_a_diagnostics.py` | CLI runner for Case B/C |

Fixtures: `ng5_j1600_tdb_equatorial` (Case B), `ng5_j1600_tdb_ecliptic_cross_engine` (Case C) in `tests/data_tempo2/manifest.json` with `parity_status: green`.

### Phase B — TDB tempo2-native geometry (core fix; order from Phase A)

Native ports only — no libstempo/tempo2 in `compute_residuals_simple` tempo2 path:

- [x] Ephemeris / planetary positions: jplephem DE405 SPK (`jug/delays/tempo2_ephemeris.py`)
- [x] Roemer / Shapiro: separate `Tempo2DelayProvider` TDB path (`tempo2_tdb_native` backend)
- [x] Implicit par defaults on TDB via `EngineConventionProfile.from_params`
- [x] Remove shared `_compute_shared_geometry_terms` from tempo2 TDB path

**Architecture (Phase B):**

| Layer | Module | Role |
|-------|--------|------|
| Runtime conventions | `jug/residuals/engine_conventions.py` | `EngineConventionProfile` — physics defaults from par + tempo2 implicit rules |
| Diagnostic conventions | `jug/residuals/diagnostic_conventions.py` | Comparison/oracle knobs only (`residual_metric`, `term_set`, …) |
| Pint geometry | `PintDelayProvider` → `_compute_pint_geometry_terms` | Astropy JPL + PINT-family Roemer/Shapiro |
| Tempo2 TDB geometry | `Tempo2DelayProvider` → `_compute_tempo2_tdb_geometry_terms` | jplephem SPK sun/planet vectors + km delay kernels |
| Tempo2 TCB geometry | `Tempo2DelayProvider` → `_compute_tempo2_tcb_geometry_terms` | IFTE + epoch map (unchanged Case A path) |

Observatory ITRF→SSB positions still use Astropy on both paths (tempo2 `get_obsCoord`
port deferred). Planet Shapiro differs between pint and tempo2 paths (~20 ns peak) as
expected when ephemeris backends diverge; Roemer remains shared on equatorial B/C.

### Phase C — TZR / absolute phase

- [x] Gate TZR ecliptic handling on `compatibility="tempo2"`
- [x] Match tempo2 TZR clock/site for `TZRSITE=gbt`, `UNITS=TDB` (AUTO→TDB in tempo2 mode)
- [x] Validate on case C and TCB fixtures with TZR (e.g. `epta_j1909_t2`)

**Architecture (Phase C):**

| Layer | Module | Role |
|-------|--------|------|
| TZR dispatch | `jug/residuals/simple_calculator.py` | `_compute_tzr_phase` branches on provider/profile |
| Pint TZR geometry | `jug/residuals/tzr_geometry.py` | `compute_tzr_astrometry_pint` (Astropy, unchanged pint semantics) |
| Tempo2 TZR geometry | `jug/residuals/tzr_geometry.py` | `compute_tzr_astrometry_tempo2` (jplephem on TDB; IFTE on TCB) |
| TZRMJD scale | `resolve_tzrmjd_epochs` | tempo2 AUTO + UNITS=TDB → no UTC conversion |

### Phase D — Tests and regression fixtures

- [ ] Add case B and/or C to `tests/data_tempo2/manifest.json`
- [ ] `test_tempo2_mode_ng5_j1600_tdb_residual_parity` (xfail → pass), raw δ only
- [ ] README: libstempo + `TEMPO2` test requirements
- [ ] Demo notebook/helpers: raw δ for tempo2 panels; weighted center only for pint-family

### Phase E — Fitter / design matrix follow-through

After raw residuals match:

- [ ] `tests/test_tempo2_designmatrix_parity.py` on new fixtures
- [ ] `tests/test_tempo2_fit_parity.py`
- [ ] FD column scaling in tempo2 mode (`optimized_fitter.py`)

---

## 10. Related code and docs

| Resource | Path |
|----------|------|
| Residual engine | `jug/residuals/simple_calculator.py` |
| Phase residuals | `compute_phase_residuals` in same file |
| Barycentric delays | `jug/delays/barycentric.py` |
| Combined delay kernel | `jug/delays/combined.py` |
| Ecliptic par ingestion | `jug/io/par_reader.py` |
| Timescales / IFTE | `jug/utils/timescales.py` |
| libstempo test oracle | `jug/testing/tempo2_reference.py` |
| libstempo sandbox | `jug/testing/sandbox_tempo2.py` |
| Par fingerprint gate | `jug/testing/fingerprint.py` |
| Residual parity tests | `tests/test_tempo2_residual_parity.py` |
| Design matrix parity | `tests/test_tempo2_designmatrix_parity.py` |
| MetaPulsar demo helpers | `examples/notebooks-dev/demo_jug_like_libstempo_lib.py` |
| Cross-engine convention policy | `docs/METHOD_DESCRIPTION.md` (MetaPulsar repo) |
| PINT/tempo2 equivalence background | `pint-tempo2-checks/regression-check/equivalence-notes/README.md` |

### Quick verification commands (jug repo)

```bash
cd ref-packages/jug
git branch --show-current && git rev-parse --short HEAD

# Case A parity (requires libstempo + TEMPO2)
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_residual_parity.py -q -o addopts=''

# Ad-hoc pint vs tempo2 on a TCB fixture
PYTHONPATH=tests:. python -c "
from tempo2_fixtures import get_tempo2_fixture
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference
import numpy as np
fx = get_tempo2_fixture('epta_j0030_isolated')
ref = tempo2_reference(fx['par_path'], fx['tim_path'])
for mode in ('pint', 'tempo2'):
    j = compute_residuals_simple(fx['par_path'], fx['tim_path'], compatibility=mode, verbose=False)
    d = (j['residuals_us'] - ref.residuals_us) * 1000
    print(mode, 'RMS ns', float(np.sqrt(np.mean(d**2))))
"
```

---

## 11. FAQ

**Q: Should tempo2 parity tests subtract a weighted mean?**  
A: **No.** Compare raw residuals as returned. Existing tests already do this.

**Q: Is the ~6 ns pattern acceptable for MetaPulsar combination?**  
A: For cross-engine PINT vs tempo2 diagnostics, ~few ns can be an expected floor.
For **JUG(tempo2) vs libstempo**, it is not acceptable.

**Q: Why do TCB tests pass?**  
A: Case A activates IFTE, TCB epoch mapping, and unweighted phase mean. That
bundle is necessary but not sufficient for TDB.

**Q: Does native ecliptic Roemer on TDB matter?**  
A: It runs in tempo2 mode, but on shared Astropy geometry it does not separate
JUG(tempo2) from JUG(pint) on delays — only phase mean does.

**Q: Can we call libstempo or tempo2 inside `compatibility="tempo2"`?**  
A: **No** at runtime. Tests may use libstempo as oracle only.

**Q: Is Phase A allowed to change the implementation strategy?**  
A: **No.** Phase A only ranks which **native** ports to build first.

**Q: Why does weighted-centering make JUG(tempo2) look like JUG(pint)?**  
A: Delays are identical on TDB today; only the internal mean subtraction differs.
Centering removes the ~61 ns phase-offset signal and hides the libstempo gap.

---

## 12. Changelog

| Date | Summary |
|------|---------|
| 2026-06-01 | Initial investigation write-up; NG5 TDB gap; TCB vs TDB branch analysis |
| 2026-06-01 | Review pass: fixture matrix, raw vs centered metrics, decisions table |
| 2026-06-01 | Decisions locked: native-only, raw δ, unweighted phase; Phase plan |
| 2026-06-01 | **Phase A landed:** provider skeleton, diagnostic conventions, Case B/C fixtures, term diagnostics, oracle runner, comparison tests |
| 2026-06-01 | **Phase B landed:** `EngineConventionProfile`, split pint/tempo2 providers, jplephem TDB geometry, B/C parity green |
