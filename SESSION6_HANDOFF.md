# Session 6 Handoff Document

**Date**: 2025-11-30  
**Status**: Investigating 6 ns/yr residual trend between JUG and PINT

---

## Current Problem

There is a **time-correlated trend** in the residual difference (JUG - PINT) for J1909-3744:
- **Magnitude**: ~6 ns/yr slope from MJD 60700-61000
- **Shape**: Linear ramp, particularly visible in the last ~20% of data (after MJD 60500)
- **Concern**: This trend will compound over time and is unacceptable for production timing software

### What We've Ruled Out

1. ✅ **Binary Shapiro delay parameterization**: Fixed earlier (M2/SINI vs H3/STIG)
2. ✅ **Major clock file differences**: Both using BIPM2024 now
3. ✅ **ELL1 binary model sign error**: Fixed in line 182 of `combined.py`
4. ✅ **Gross timing errors**: Agreement is generally excellent (<<1 μs), just this small trend remains

### Current Test Configuration

**Both codes now using**:
- Clock file: `tai2tt_bipm2024.clk`
- EOP file: `eopc04_IAU2000.62-now`
- Binary model: ELL1
- Pulsar: J1909-3744 (10,408 TOAs, MJD 58472-61026)

---

## What's Been Done This Session

### 1. Milestone 1 Completion ✅
- Fixed ELL1 binary delay sign error (line 182)
- Implemented weighted RMS residuals
- Created CLI tool with plotting
- Benchmarked against PINT/Tempo2
- **Result**: 0.817 μs weighted RMS (vs 0.799 μs PINT, 0.779 μs Tempo2)

### 2. Milestone 2 Progress ✅
- Implemented Gauss-Newton + Levenberg-Marquardt fitter with JAX acceleration
- Created `jug/fitting/gauss_newton.py` (JIT-compiled design matrix calculation)
- Tested on synthetic data: recovers F0/F1 perfectly
- **Status**: Ready for real data testing, but blocked by residual trend issue

### 3. Binary Model Expansion ✅
- Implemented BT/DD binary models in `jug/delays/binary_bt.py`
  - BT: Keplerian + Einstein + Shapiro
  - DD: BT + OMDOT + XDOT
  - Kepler solver using Newton-Raphson (JAX JIT)
- Implemented T2 binary model in `jug/delays/binary_t2.py`
  - Universal model with EDOT, KIN, KOM support
  - Tested: matches BT to nanosecond precision
- **Status**: Models implemented, not yet integrated into main calculator

### 4. GitHub Repository Setup ✅
- Created private repo: https://github.com/MattTMiles/jug
- Added comprehensive .gitignore
- Pushed initial codebase

---

## Investigation Results

### Clock Correction Analysis
Checked BIPM2023 vs BIPM2024:
- JUG default: BIPM2023
- PINT default: BIPM2019
- **Action taken**: Forced both to BIPM2024
- **Result**: Reduced difference slightly, but **trend persists**

### Component-by-Component Comparison
Created `debug_component_diff.py` to compare:
- Clock corrections: JUG vs PINT
- Roemer delay: JUG vs PINT  
- Binary delay: JUG vs PINT
- Full residuals: JUG vs PINT

**Finding**: Need to isolate which component has the trend.

### Trend Analysis
- Created `debug_component_trend.py` to fit linear trends to MJD 60700+
- **Latest plot**: `debug_time_trend.png`
- Shows ~6 ns/yr slope in residual difference after MJD 60700

---

## Files Modified/Created This Session

### Core Package Files
- `jug/residuals/simple_calculator.py`: Fixed M2/SINI Shapiro delay handling
- `jug/delays/combined.py`: Fixed ELL1 sign error (line 182)
- `jug/delays/binary_bt.py`: NEW - BT/DD binary models
- `jug/delays/binary_t2.py`: NEW - T2 binary model
- `jug/fitting/gauss_newton.py`: NEW - JAX-accelerated fitter

### CLI Tools
- `jug/cli/compute_residuals.py`: Added weighted RMS, plotting with error bars

### Debug Scripts
- `debug_component_diff.py`: Compare delay components
- `debug_component_trend.py`: Analyze time-correlated trends
- `compare_bipm2024.py`: Force both codes to use BIPM2024

### Documentation
- `MILESTONE_1_COMPLETION.md`: Milestone 1 summary
- `MILESTONE_2_SESSION1_STATUS.md`: Milestone 2 progress
- `SESSION6_BINARY_EXPANSION.md`: Binary model implementation notes
- `SESSION6_MULTI_PULSAR_TESTING.md`: Multi-pulsar test plan
- `BENCHMARK_REPORT.md`: Speed comparison JUG vs PINT vs Tempo2
- `WEIGHTED_RMS_UPDATE.md`: Weighted RMS implementation notes

---

## Next Steps for Investigation

### Priority 1: Find the Trend Source
1. **Run component comparison with BIPM2024 enforced** (may not have been done correctly)
   ```bash
   python compare_bipm2024.py  # Should generate clean component plots
   ```

2. **Check each delay component for linear trend**:
   - Clock corrections (JUG - PINT)
   - Roemer delay (JUG - PINT)
   - Shapiro delay (JUG - PINT)
   - Einstein delay (JUG - PINT)
   - Binary Roemer (JUG - PINT)
   - Binary Einstein (JUG - PINT)
   - Binary Shapiro (JUG - PINT)

3. **Possible culprits**:
   - EOP file extrapolation beyond valid range?
   - Clock file coverage (does MeerKat clock go to MJD 61026?)
   - Time scale conversion (TDB calculation difference?)
   - Binary orbital parameter evolution (PBDOT, XDOT differences?)

4. **Verification test**: Compare on **earlier data only** (MJD < 60500)
   - If trend disappears, confirms it's a late-time effect

### Priority 2: Once Fixed, Continue Milestone 2
- Test fitter on real J1909-3744 data
- Implement remaining fittable parameters (RA, DEC, PMRA, PMDEC, PX, DM, etc.)
- Multi-pulsar testing with different binary models

---

## Key Code Locations

### Residual Calculation Pipeline
1. **Entry point**: `jug/cli/compute_residuals.py`
2. **Main calculator**: `jug/residuals/simple_calculator.py`
   - Line 57-65: Clock correction chain
   - Line 67-126: Barycentric correction (Roemer + Einstein + Shapiro)
   - Line 128-198: Binary delay calculation (ELL1/BT/DD/T2 dispatcher)
   - Line 200-220: Emission time calculation
   - Line 222-260: JAX residual computation

### Binary Models
- **ELL1**: `jug/delays/combined.py` (lines 71-198)
- **BT/DD**: `jug/delays/binary_bt.py` (NEW, not yet integrated)
- **T2**: `jug/delays/binary_t2.py` (NEW, not yet integrated)

### Fitting Code
- **Gauss-Newton**: `jug/fitting/gauss_newton.py`
- Uses `jax.jacfwd()` for automatic differentiation
- JIT-compiled design matrix calculation

---

## Test Commands

### Reproduce Current State
```bash
# Clean comparison with BIPM2024
python compare_bipm2024.py

# Generate residual plot
jug-compute-residuals \
  /home/mattm/soft/JUG/examples/J1909-3744.par \
  /home/mattm/soft/JUG/examples/J1909-3744_MeerKAT.tim

# Run component difference analysis
python debug_component_diff.py

# Analyze trend
python debug_component_trend.py
```

### Benchmark Performance
```bash
python benchmark.py
# Current results:
# JUG:    0.028s ± 0.002s (weighted RMS: 0.817 μs)
# PINT:   0.234s ± 0.015s (weighted RMS: 0.799 μs)
# Tempo2: 0.099s ± 0.008s (weighted RMS: 0.779 μs)
# JUG is ~8x faster than PINT, ~3.5x faster than Tempo2
```

---

## Critical Insight

**The 6 ns/yr trend is small but NOT acceptable** because:
1. It compounds over multi-decade timing baselines
2. Over 10 years: 60 ns systematic error
3. Over 25 years (IPTA data): 150 ns error
4. This is comparable to GW signal amplitudes we're searching for

**Must be fixed before moving to production.**

---

## Questions for Next Session

1. Is the trend in clock corrections, barycentric delays, or binary delays?
2. Are we correctly handling file coverage boundaries (EOP/clock files)?
3. Is there a TDB calculation difference between JUG and PINT?
4. Should we test on a **non-binary pulsar** to rule out binary model issues?

---

## Repository Status

- Branch: `main`
- Latest commit: "Initial commit with working ELL1 model and fitter"
- GitHub: https://github.com/MattTMiles/jug (private)
- All core functionality working, just this trend issue remains

---

**BOTTOM LINE**: JUG is 99% there. We have a working timing package that's 8x faster than PINT with excellent agreement. Just need to nail down this last 6 ns/yr trend before declaring victory on Milestone 1 and moving fully into Milestone 2 fitting work.
