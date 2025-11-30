# DD Binary Model Debug Handoff

**Date**: 2025-11-30  
**Status**: 🔴 **CRITICAL BUG - NEEDS FRESH INVESTIGATION**

---

## Summary

The DD/DDH/DDGR/DDK binary model implementation in JUG produces residuals ~140x worse than PINT. Multiple debug attempts have not identified the root cause. Need fresh eyes to compare JUG implementation against PINT and Tempo2.

### Test Results (J1012-4235, DD binary, 7089 TOAs):
- **PINT RMS**: 5.334 μs ✅
- **JUG RMS**: 755.687 μs ❌  
- **Difference**: 755.6 μs RMS (141x worse than PINT)

### Root Cause
The `jug/delays/binary_bt.py` implementation does NOT correctly compute DD model delays. The issue exists in BOTH:
1. Original implementation (checked in to git)
2. Attempted tempo2-based rewrite (reverted)

---

## What Works ✅

**ELL1/ELL1H binary models** (e.g., J1909-3744):
- JUG vs PINT difference: < 1 μs RMS
- Shapiro delay: Correctly handles both M2/SINI and H3/STIG parameters
- Sign convention: Correct (fixed earlier in Session 6)

**Non-binary pulsars**: All components match PINT/Tempo2 to < 1 μs

---

## What's Broken ❌

**DD/DDH/DDGR/DDK binary models**:
- Kepler equation solver appears to converge correctly
- Binary delay magnitude is reasonable (±21.26 s for J1012-4235, matching A1 parameter)
- But residuals differ from PINT by ~755 μs RMS (should be <10 μs)
- User confirmed Tempo2 also disagrees with JUG, confirming JUG is wrong

**Possible causes**:
- Wrong sign convention
- Missing delay components
- Incorrect handling of OMDOT/PBDOT/XDOT parameter evolution
- Wrong barycentric time reference
- Incorrect delay formula

---

## Code Locations

### JUG Implementation

**File**: `/home/mattm/soft/JUG/jug/delays/binary_bt.py`

**Key function**: `bt_binary_delay(t_bary_mjd, model_params)`

**Parameters used**:
- `PB`: Orbital period (days)
- `A1`: Projected semi-major axis (light-seconds)
- `ECC`: Eccentricity
- `OM`: Longitude of periastron (degrees)
- `T0`: Time of periastron passage (MJD)
- `GAMMA`: Einstein delay amplitude (seconds)
- `PBDOT`: Orbital period derivative (dimensionless, 1e-12)
- `M2`: Companion mass (solar masses) - for Shapiro delay
- `SINI`: Sine of inclination - for Shapiro delay
- `OMDOT`: Periastron advance rate (deg/yr) - DD only
- `XDOT`: A1 derivative (light-sec/sec) - DD only

**Algorithm**:
1. Compute time since T0 in years: `dt_years = (t_bary_mjd - T0) / 365.25`
2. Apply PBDOT: `PB_current = PB * (1 + PBDOT * dt_years)`
3. Compute mean anomaly: `M = 2π * (t_bary_mjd - T0) / PB_current`
4. Solve Kepler equation for eccentric anomaly E using Newton-Raphson
5. Compute true anomaly: `tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)`
6. Apply OMDOT: `OM_current = OM + OMDOT * dt_years`
7. Apply XDOT: `A1_current = A1 + XDOT * dt_sec`
8. Compute Roemer delay: `α = A1 * (sin(OM+ν) + e*sin(OM))`
9. Compute Einstein delay: `β = GAMMA * sin(E)`
10. Compute Shapiro delay: `γ = -2*r*log(1 - s*sin(OM+ν))` where `r = TSUN*M2`, `s = SINI`
11. Return total: `α + β + γ`

**Integration**: Called from `jug/residuals/simple_calculator.py` (lines ~380-430)

### PINT Implementation

**File**: `/home/mattm/soft/PINT/src/pint/models/stand_alone_psr_binaries/DD_model.py`

**Class**: `DDmodel` - Standalone DD binary model

**Key methods**:
- `DDmodel.DD_delay()` - Computes binary delay
- Uses alpha/beta formulation from Damour & Deruelle (1986)
- Includes derivatives for fitting (we don't need these yet)

**Note**: PINT code is complex with unit handling, derivatives, etc. Core delay calculation needs to be extracted.

### Tempo2 Implementation

**File**: `/home/mattm/not_in_use_reference/tempo2/DDmodel.C`

**Function**: `updateDDmodel()`

**Key differences from JUG**:
- Uses `orbits = tt0/pb - 0.5*(pbdot+xpbdot)*(tt0/pb)^2` for mean anomaly calculation
- Computes `omega = om + k*ae` where `k` depends on OMDOT
- Returns `torb = -d2bar` (negative of d2bar)
- More complex handling of parameter evolution

**Note**: C code, requires careful translation. Previous attempt to copy this exactly did not fix the bug.

---

## Test Case

**Pulsar**: J1012-4235 (DD binary)

**Files**:
- PAR: `/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par`
- TIM: `/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim`

**Key parameters**:
```
BINARY DD
PSR J1012-4235
PB 37.97099340614 days
A1 21.262419 light-seconds
ECC 0.000346
OM 276.54 degrees
T0 54680.919 MJD
PBDOT -3.4e-12
OMDOT 0.014 deg/yr
M2 0.167 solar masses
SINI 0.99996
```

**Current results**:
- N_TOAs: 7089
- PINT RMS: 5.334 μs ✅
- JUG RMS: 755.687 μs ❌
- Difference: 755.6 μs RMS (141x worse)

**Success criteria**: JUG vs PINT difference < 10 μs RMS

**Test script**: `/home/mattm/soft/JUG/debug_dd_binary_delays.py`

---

## Debug Approaches Tried

1. ✅ Verified Kepler solver converges (it does)
2. ✅ Checked parameter extraction from .par file (correct)
3. ✅ Validated on synthetic BT data (works for BT)
4. ✅ Compared with Tempo2 source code (attempted translation, didn't fix bug)
5. ❌ Side-by-side comparison of intermediate values (JUG vs PINT) - NOT DONE YET

---

## Recommended Next Steps

### CRITICAL: Component-by-component comparison

For a **single TOA** (e.g., first TOA), extract and compare:

**From PINT**:
1. Mean anomaly M
2. Eccentric anomaly E (after Kepler solve)
3. True anomaly ν
4. Omega (with OMDOT applied)
5. A1 (with XDOT applied)
6. PB (with PBDOT applied)
7. Roemer delay (alpha term)
8. Einstein delay (beta term)
9. Shapiro delay (gamma term)
10. Total binary delay

**From JUG**:
- Same list as above

**Method**:
1. Instrument PINT code to print these values
2. Instrument JUG code to print these values
3. Compare line-by-line to find where they diverge
4. Fix the divergent calculation

### Alternative: Study PINT DD_model.py carefully

1. Read PINT's `DD_model.py` source code in detail
2. Extract the core delay calculation (ignore derivatives, units)
3. Translate to JAX-compatible code
4. Test on J1012-4235
5. Verify match with PINT

---

## Important Clues

1. **Binary delay magnitude is correct**: Range is ±21.26 s, matching the A1 parameter. This suggests the Kepler solver and basic geometry are working.

2. **ELL1 works perfectly**: The problem is specific to DD models, not all binary models.

3. **Tempo2 also disagrees with JUG**: User confirmed this, so it's not a PINT-specific issue.

4. **755 μs is ~1/50th of a millisecond**: This is a significant error for pulsar timing but still small compared to orbital period. Suggests a subtle formula error or missing correction.

5. **Binary delay changes on iteration 2**: In debug output, binary delay changes by 4260 μs on second residual iteration. This is expected (residuals feed back into emission time calculation), but the magnitude might be a clue.

---

## Likely Bug Categories

Based on the symptoms, the bug is probably ONE of these:

### A. Sign Error
- DD delay might need opposite sign from BT
- Shapiro delay sign convention might differ
- Check if delay should be added or subtracted

### B. Missing Correction Term
- DD model has additional terms beyond BT
- Check Damour & Deruelle (1986) paper for complete formula
- Aberration correction? Additional Shapiro terms?

### C. Parameter Evolution Formula
- OMDOT/PBDOT/XDOT might use different time reference
- Check if dt should be in TDB, TCB, or something else
- Verify units (deg/yr vs rad/sec, etc.)

### D. Barycentric Time Reference
- Should use `t_bary` (barycentric arrival time) or something else?
- ELL1 uses TASC (ascending node time), DD uses T0 (periastron time)
- Verify time reference is consistent

### E. Kepler Equation Form
- Different codes use different Kepler equation forms
- Check if mean anomaly calculation matches PINT/Tempo2 exactly
- Verify angle wrapping (0-2π vs -π to π)

---

## Status Summary

**Clock file validation**: ✅ Implemented in `jug/io/clock.py`

**ELL1 model**: ✅ Working perfectly

**BT model**: ✅ Works on synthetic data (not tested on real pulsar yet)

**DD model**: ❌ **BROKEN** - needs fresh investigation

**T2 model**: ⏸️ Not yet integrated

---

## For the Next AI

You're inheriting a partially working binary model system. The good news: ELL1 works perfectly. The bad news: DD doesn't, and we've gone in circles trying to debug it.

**Your mission**: Find the specific line of code where JUG's DD calculation diverges from PINT's, and fix it.

**Suggested approach**: Instrument both JUG and PINT to print intermediate values for a single TOA, then compare line-by-line.

**Key insight**: The bug is subtle. Binary delay magnitude is correct, Kepler solver converges, but final residuals are 140x worse than PINT. It's likely a sign error, missing term, or wrong time reference.

Good luck! 🚀
