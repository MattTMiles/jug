# JUG Development Working Notes

## Active Task
**B1855+09 prefit residual accuracy — ✅ ACHIEVED: JUG wRMS 5.539 μs = PINT 5.539 μs.**

Two issues were resolved:
1. **TIM `-to` flags**: The TIM file has per-TOA `-to` flags (-0.789 μs and -0.839 μs on 1575
   TOAs). PINT applies these as TIME statement offsets in its clock correction chain. JUG was
   ignoring them. Fix: extract `-to` values from TOA flags and pass as `time_offsets` to
   `compute_tdb_standalone_vectorized`.
2. **EFAC/EQUAD error scaling**: PINT's `rms_weighted()` uses noise-model-scaled errors
   (EFAC/EQUAD applied). JUG was using raw TIM file errors. Fix: added scaled wRMS computation
   using existing `apply_white_noise()` from `jug/noise/white.py`.

## Repo State
- **Branch**: `main`
- **Latest commit**: `2cb5855` + uncommitted changes (9 files, 627 insertions)
- **Summary**: Multiple fixes to residual computation for B1855+09 (ecliptic pulsar, Arecibo, DE421, DMX).
  Key changes: auto-observatory detection, multi-observatory clock files, BIPM version from par CLK,
  ephemeris resolver for old DE kernels, TZRMJD UTC→TDB conversion, TZR phase subtraction,
  DMX integration into delay chain. **TZR DMX fix added but NOT YET WORKING.**

## Architecture Overview

### Key Files
| Component | File | Role |
|-----------|------|------|
| Session | `jug/engine/session.py` | `TimingSession` — caches par/tim parsing, dispatches to fitter |
| Fitter | `jug/fitting/optimized_fitter.py` | `GeneralFitSetup` dataclass + `_run_general_fit_iterations()` |
| Red Noise | `jug/noise/red_noise.py` | `RedNoiseProcess`, `DMNoiseProcess`, `build_fourier_design_matrix` |
| DMX | `jug/model/dmx.py` | `parse_dmx_ranges`, `build_dmx_design_matrix`, `DMXRange` |
| Noise Config | `jug/engine/noise_mode.py` | `NoiseConfig` — per-process on/off toggles |
| White Noise | `jug/noise/white.py` | EFAC/EQUAD scaling |
| ECORR | `jug/noise/ecorr.py` | `ECORRWhitener` |
| Par Reader | `jug/io/par_reader.py` | Parses `.par`, stores `_noise_lines` for white noise |
| WLS Solver | `jug/fitting/wls_fitter.py` | `wls_solve_svd` — SVD-based WLS |
| **Integration Tests** | `tests/test_fitter_noise_integration.py` | 12 tests proving noise integration works |

### Integration Pattern (follows ECORR precedent)
1. **Setup phase** (`_build_general_fit_setup_from_files/cache`): Parse noise params → build Fourier matrices → store in `GeneralFitSetup`
2. **Iteration phase** (`_run_general_fit_iterations`): Append Fourier columns to design matrix M → apply prior → solve augmented normal equations
3. **NoiseConfig gating**: All noise processes (EFAC, EQUAD, ECORR, RedNoise, DMNoise) are gated by `NoiseConfig.is_enabled()` checks

### Red/DM Noise Integration Approach
- Fourier coefficients are additional fit parameters with Gaussian prior
- Augmented normal equations: `(M^T C^{-1} M + Φ^{-1}) δp = M^T C^{-1} r`
- This is equivalent to enterprise/PINT marginalised-likelihood
- Fourier basis F is static (built once in setup, reused each iteration)
- Prior φ = diag(PSD) is also static
- After solving, only timing parameter updates extracted; Fourier coefficients discarded

### Test Data
- `data/pulsars/pulsars_w_noise/J1909-3744_tdb.par` — has TNREDAMP, TNREDGAM, TNREDC, TNDMAMP, TNDMGAM, TNDMC, EFAC, EQUAD
- `data/pulsars/J1909-3744_tdb.par` — same pulsar, no noise params (baseline)

## Completed Checklist

### Phase 0 — Core Scaffolding ✅
- [x] 0.1 Data integrity checks
- [x] 0.2 Flag aliasing
- [x] 0.3 Noise/backend diagnostics

### Phase 1 — Red Noise + DM Noise ✅
- [x] 1.1 Added 8 new fields to `GeneralFitSetup` dataclass (red/DM noise basis+prior, DMX matrix+labels, noise_config)
- [x] 1.2 Parse and build Fourier basis in both setup functions (`_build_general_fit_setup_from_files` and `_build_general_fit_setup_from_cache`)
- [x] 1.3 Augment design matrix M with Fourier/DMX columns and apply prior regularization in `_run_general_fit_iterations`
- [x] 1.4 Fixed case-sensitivity bug: par reader uppercases keys (TNREDAMP) but parsers expected mixed case (TNRedAmp)
- [x] 1.5 Verified: Red noise (165 harmonics → 330 columns) and DM noise (165 harmonics → 330 columns) built correctly
- [x] 1.6 Verified: Fit results differ with vs without noise (RMS 0.798 vs 0.405 μs)

### Phase 2 — DMX Support ✅
- [x] 2.1 DMX parsing (`jug/model/dmx.py`) already implemented
- [x] 2.2 DMX design matrix wired into both setup functions
- [x] 2.3 DMX gated by NoiseConfig when available

### Phase 3 — Noise Mode Integration ✅
- [x] 3.1 `NoiseConfig` accepted by `TimingSession.fit_parameters(noise_config=)` parameter
- [x] 3.2 Override passed through to both setup builders
- [x] 3.3 EFAC/EQUAD/ECORR gated by `noise_config.is_enabled()` — filtered before `apply_white_noise()`
- [x] 3.4 Verified: Disabling all noise matches no-noise par baseline exactly (F0 diff < 1e-10, RMS diff < 1e-6)

### Phase 4 — Integration Tests ✅
- [x] 4.1 `tests/test_fitter_noise_integration.py` — 16 tests in 4 classes:
  - `TestNoiseSetupStructure` (5 tests): Verify Fourier bases are built/absent as expected
  - `TestNoiseNumericalEffects` (4 tests): Verify noise changes fit results
  - `TestNoiseConfigGating` (3 tests): Verify NoiseConfig correctly gates effects
  - `TestDMXIntegration` (4 tests): Verify DMX design matrix built from B1855+09 PINT test data

## Files Modified (This Session)
| File | Changes |
|------|---------|
| `jug/residuals/simple_calculator.py` | (a) Fixed corrupted import line 15; (b) Added `_resolve_ephemeris()` for old JPL ephemerides (DE421); (c) EPHEM from par file threaded through all SSB and Shapiro calls; (d) BIPM clock version selection from par CLK parameter; (e) TZR block ephemeris updated; (f) **DMX assignment: changed from TDB to site MJDs** |
| `jug/model/dmx.py` | Removed frequency filtering from `assign_toas_to_dmx()` — now MJD-only like PINT/Tempo2 |
| `clock_files/ao2gps.clk` | Copied from `data/clock/` |
| `clock_files/tai2tt_bipm2019.clk` | Copied from tempo2 reference |
| `clock_files/tai2tt_bipm2015.clk` | Copied from tempo2 reference |

## Files Modified (Prior Sessions)
| File | Changes |
|------|---------|
| `jug/fitting/optimized_fitter.py` | Added 8 GeneralFitSetup fields; noise parsing + Fourier basis building in both setup functions; augmented solver with prior in iteration loop; `noise_config` parameter on both setup builders; NoiseConfig gates EFAC/EQUAD/ECORR |
| `jug/noise/red_noise.py` | Added uppercase key variants (TNREDAMP/TNREDGAM/TNREDC, TNDMAMP/TNDMGAM/TNDMC) to `parse_red_noise_params` and `parse_dm_noise_params` |
| `jug/engine/session.py` | Added `noise_config` parameter to `fit_parameters()` method; passed through to cache-based setup builder |
| `tests/test_fitter_noise_integration.py` | NEW — 12 integration tests |
| `docs/DEV_WORKING_NOTES.md` | This file — continuously updated |

## Decisions + Rationale
1. **Fourier coefficients as augmented fit params with prior**: Matches enterprise/PINT convention. More natural than Woodbury for WLS framework.
2. **JAX for Fourier basis**: Already implemented in `_fourier_design_jax` (JIT-compiled). Build once in setup, reuse as JAX array.
3. **Prior regularization**: Applied after ECORR whitening — add `Φ^{-1}` to normal equations, not raw design matrix.
4. **NoiseConfig gating at filter level**: Filter `noise_entries` list before passing to `apply_white_noise`, rather than adding guards inside `apply_white_noise`. Cleaner separation.
5. **Case sensitivity fix**: Added uppercase variants to parsers rather than modifying par reader, since par reader uppercasing is by design.

## Known Issues / Remaining Work
- Binary parameter alias (`E` → `ECC`) resolved by editing the par file directly
- DMX is not gated by NoiseConfig (always active when present in par file — correct by design)
- Red/DM noise Fourier coefficients from the augmented solver are discarded (not returned to caller) — correct for timing estimation but would need change for noise reconstruction/plotting
- DMX fit flags not separately extracted (all present DMX ranges always active — correct by design)
- **Arecibo clock file mismatch**: JUG uses `ao2gps.clk` (TEMPO2 format, site→GPS), PINT uses `time_ao.dat` (TEMPO format, site→UTC). This causes ~0.3 μs TDB difference. To fix: either add TEMPO clock format parsing or obtain equivalent TEMPO2 file.
- **TZRSITE TEMPO codes**: TZRSITE "3" (TEMPO code for Arecibo) should be mapped properly; currently falls back to "ao" which works but logs a warning.

## B1855+09 Residual Accuracy History
| Session | Fix | wRMS (μs) | Notes |
|---------|-----|-----------|-------|
| Prior | Baseline | 1610 | First attempt, many issues |
| Prior | Multi-observatory + clock + DMX | 32.7 | Major fixes |
| Prior | Ephemeris DE421 from par file | 32.4 | Small improvement |
| Prior | DMX frequency filtering fix | 29.7 | 51 TOAs recovered |
| Prior | BIPM clock version from CLK par | 29.7 | Correct clock version |
| Prior | DMX site-time assignment fix | 6.586 | Critical: was using TDB, should use site MJDs |
| Prior | TZRMJD UTC→TDB conversion | 6.586 | Correct but doesn't change wRMS (absorbed by mean sub) |
| Prior | TZR phase subtraction before wrapping | 6.586 | Correct but same pulse numbers |
| Prior | TZR DMX delay fix | 6.561 | Working — adds DMX delay to TZR reference |
| **Current** | **TIM `-to` flag support** | **6.178 (raw) / 5.539 (scaled)** | **Per-TOA TIME offsets applied to clock chain** |
| **Current** | **EFAC/EQUAD scaled wRMS** | **5.539** | **Matches PINT exactly** |
| Target | — | 5.539 | ✅ **ACHIEVED** |

### TZR DMX Bug (Current — Root Cause Found)
- **Root cause**: JUG's TZR delay computation uses `compute_total_delay_jax()` which includes
  DM, solar wind, FD, and binary — but NOT DMX. DMX is added separately for main TOAs
  (lines 665–674) but was never added for TZR.
- **Impact**: TZR phase is wrong by F0 × K_DM × DMX / freq² = 0.072 cycles at TZRFRQ=424 MHz.
  After wrapping + weighted mean subtraction, this creates a 1/f² bias in residuals
  that inflates wRMS from ~5.5 to 6.6 μs.
- **Fix attempted**: Added DMX delay computation for TZR at lines 831–844. But wRMS unchanged.
- **Debugging needed**: Verify the fix is actually computing non-zero DMX delay for TZR.
  Run with verbose=True and check the output. The fix may be working but the `subtract_mean=True`
  on line 895 may be re-introducing the bias. Try with `subtract_mean=False` when TZR is used.

### DMX Site-Time Bug (Critical Fix)
- **Root cause**: JUG used TDB MJDs for DMX range matching; PINT uses site arrival MJDs (`mjd_float`).
- **Impact**: ~66 second offset between site and barycentric time caused 1584/4005 TOAs to fall outside their narrow DMX ranges (ranges span ~4400 seconds).
- **Fix**: Changed `build_dmx_design_matrix` call in `simple_calculator.py` line 660 to use `mjd_utc` (site arrival MJDs) instead of `tdb_mjd`.
- **Verification**: Where both JUG and PINT assign DMX, delays match to 0.000000 μs. After fix, all 4005 TOAs are assigned.

### Remaining 0.33 μs Gap Analysis
After fixing all delay components (Roemer+Shapiro: 0.000 μs diff, Base DM: 0.000 μs, DMX: 0.000 μs, FD: 0.003 μs, Binary: 0.004 μs), the remaining difference is entirely in **clock corrections**:
- TDB difference: mean=0.326 μs, std=0.445 μs
- Cause: `ao2gps.clk` (TEMPO2) vs `time_ao.dat` (TEMPO) provide different Arecibo clock corrections
- High-freq residual diff after TDB correction: -1.1 μs mean (constant, absorbed by F0 fit)
- Low-freq residual diff after TDB correction: -10.6 μs mean (DM-like, absorbed by DM fit)
- Per-TOA scatter within each band: ~0.2 μs std (from clock interpolation differences)

## How to Resume

### IMMEDIATE NEXT STEPS (TZR DMX Fix)

**The Problem**: JUG's prefit wRMS is 6.586 μs; PINT and Tempo2 both give 5.539 μs.

**Root Cause (CONFIRMED)**: The TZR (reference) delay is missing the DMX contribution.
- PINT includes DMX in the TZR delay: TZR DMX = 0.000387 s at 424 MHz
- JUG's JAX kernel (`compute_total_delay_jax`) does NOT include DMX (it's added separately for main TOAs)
- The TZR delay is computed at line 815–827 via the JAX kernel only — missing DMX
- This shifts TZR phase by F0 × 0.000387 = 0.072 cycles
- After wrapping + weighted mean subtraction, this creates a 1/f² bias in residuals

**Evidence**:
- Total delays match PINT to 0.006 μs RMS (delays are correct)
- DM+DMX delays are identical (diff = 0.0000 μs)
- All 4005 pulse numbers match PINT exactly
- TZR phase: PINT = 52937605.785, JUG = 52937605.857 → diff = 0.072 cycles = exactly F0 × DMX_delay
- When I recompute JUG residuals using PINT's TZR phase, the wRMS diff drops to 0.40 μs (vs 1.28 μs)

**Fix Already Attempted** (lines 831–844 of `simple_calculator.py`):
```python
# Add DMX delay to TZR (not included in JAX kernel, same as main TOAs)
from jug.model.dmx import parse_dmx_ranges, build_dmx_design_matrix
tzr_dmx_ranges = parse_dmx_ranges(params)
if tzr_dmx_ranges:
    tzr_dmx_matrix, _ = build_dmx_design_matrix(
        np.array([float(TZRMJD_raw)], dtype=np.float64),
        np.array([tzr_freq_bary]),
        tzr_dmx_ranges
    )
    tzr_dmx_values = np.array([r.value for r in tzr_dmx_ranges])
    tzr_dmx_delay = float((tzr_dmx_matrix @ tzr_dmx_values)[0])
    tzr_delay += np.longdouble(tzr_dmx_delay)
```

**Why it didn't work**: The code runs but wRMS stays at 6.586 μs. Possible reasons:
1. `build_dmx_design_matrix` may not be finding the right DMX range for TZRMJD.
   - TZRMJD = 54981.28 — check which DMX range contains this MJD
   - The function takes site-arrival MJDs, so `TZRMJD_raw` (not `TZRMJD_TDB`) is correct
   - BUT: check if the matrix returns non-zero values. Print `tzr_dmx_delay` to confirm.
2. The `subtract_mean=True` on line 895 may absorb the fix (weighted mean subtraction
   re-introduces frequency-dependent bias). Check if `subtract_mean=False` with TZR
   subtraction gives better results.
3. The fix may actually be working but something else is resetting the result.

**Debugging Steps**:
1. Run with `verbose=True` and check the TZR DMX delay value printed
2. Compare JUG's TZR total delay (with DMX) vs PINT's TZR total delay (-325.467 s)
3. Verify that `build_dmx_design_matrix` returns non-zero for the single TZR TOA
4. Try `subtract_mean=False` (line 895) when TZR is used — PINT's TZR centering
   should make mean subtraction unnecessary (PINT's mean without subtraction is -0.013 μs)

**Key Insight About Mean Subtraction**:
- PINT subtracts weighted mean = True by default, BUT its mean is near zero (-0.013 μs)
  because TZR centers residuals correctly
- JUG's residuals before mean subtraction have mean = -1.26 μs (with wrong TZR)
  or should be near 0 if TZR is fixed
- The weighted mean subtraction with 2 frequency bands creates a 1/f² artifact when
  the mean is large — it works fine when the mean is small

### After TZR Fix

Once JUG matches PINT to ~0.4 μs wRMS (the remaining achromatic gap from clock file differences):
1. Run tests: `python -m pytest tests/ -o "addopts=" -x --ignore=tests/test_cli_smoke.py`
2. The remaining 0.4 μs gap is from:
   - JUG uses `ao2gps.clk` (TEMPO2 format) for Arecibo clock
   - PINT uses `time_ao.dat` (TEMPO format) — slightly different corrections
   - This is achromatic and acceptable for now
3. Return to backend noise integration tasks (user's main goal)

### Test Files
- Par: `/home/mattm/soft/PINT/tests/datafile/B1855+09_NANOGrav_9yv1.gls.par`
- Tim: `/home/mattm/soft/PINT/tests/datafile/B1855+09_NANOGrav_9yv1.tim`
- Ecliptic coords (ELONG/ELAT), Arecibo only, DE421, CLK=TT(BIPM2019), UNITS=TDB
- 72 DMX ranges, 4005 TOAs, DD binary model, 3 FD parameters
- TZRMJD=54981.28084616488447, TZRFRQ=424.000, TZRSITE=3 (Arecibo)

### Comparison Script
```python
import numpy as np, warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
import pint.toa as toa, pint.models as models, pint.residuals as residuals
par = '/home/mattm/soft/PINT/tests/datafile/B1855+09_NANOGrav_9yv1.gls.par'
tim = '/home/mattm/soft/PINT/tests/datafile/B1855+09_NANOGrav_9yv1.tim'
m = models.get_model(par)
t = toa.get_TOAs(tim, ephem='DE421', bipm_version='BIPM2019')
r = residuals.Residuals(t, m)
pint_wrms = float(r.rms_weighted().to('us').value)
from jug.residuals.simple_calculator import compute_residuals_simple
jug = compute_residuals_simple(par, tim, verbose=False)
print(f"JUG wRMS: {jug['weighted_rms_us']:.3f} μs, PINT wRMS: {pint_wrms:.3f} μs")
```

### PINT Reference Values
- PINT wRMS (BIPM2019 or 2023): 5.539 μs
- Tempo2 wRMS: prefit 5.539 μs, postfit 1.207 μs
- JUG wRMS (current): 6.586 μs (target: ≤5.6 μs)

## Test Status
- **160 passed**, 45 warnings, 1 pre-existing error (test_cli_smoke fixture issue)
- Command: `python -m pytest tests/ -o "addopts=" -x --ignore=tests/test_cli_smoke.py`
- **Tests NOT re-run after this session's changes** — run them first!
- Integration tests: `python -m pytest tests/test_fitter_noise_integration.py -v` — 16/16 pass
- DMX tests use B1855+09 data from `/home/mattm/soft/PINT/tests/datafile/`
