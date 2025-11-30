# JUG Binary Model Fix - Handoff Document

## Project Goal
Create a PINT-independent pulsar timing pipeline ("JUG") in the notebook `residual_maker_playground_MK2.ipynb` that matches PINT's accuracy.

## What Has Been Done

### 1. K_DM Constant Fix (COMPLETED ✓)
- **Problem**: JUG used `K_DM_SEC = 4148.808` but PINT/TEMPO2 use `1/2.41e-4 = 4149.377593`
- **Fix Applied**: Updated cell `#VSC-37db9bff` (lines ~60-78) to use `K_DM_SEC = 1.0/2.41e-4`
- **Result**: Reduced frequency-dependent error from ~7 µs at 900 MHz to ~0.6 µs

### 2. Barycentric Frequency Fix (COMPLETED ✓ - done in prior session)
- JUG now correctly computes `f_bary = f_topo × (1 - v_radial/c)` where `v_radial = ssb_vel · L_hat`
- This eliminated ~5.7 µs annual signal

### 3. Third-Order ELL1 Binary Model (COMPLETED ✓)
- Implemented Zhu et al. (2019), Fiore et al. (2023) third-order eccentricity terms
- Includes PBDOT, XDOT, GAMMA, binary Shapiro delay
- Function: `ell1_binary_delay_full()` in cell `#VSC-09e4100e` (lines ~285-398)

## Current Status

After K_DM fix:
- **JUG residuals RMS: 2.084 µs** (was 2.633 µs)
- **PINT residuals RMS: 0.817 µs**
- **JUG-PINT difference RMS: 1.929 µs** (was 2.489 µs)

### Remaining Issue: Binary Model 2.6 µs First-Harmonic Error

**Diagnostic Results:**
- Binary delay difference has **2.61 µs amplitude** at **1ω (first harmonic)**
- Pattern is almost pure **cosine** (B = 2.609 µs, A ≈ 0)
- After removing 1ω fit, residual RMS drops to **0.735 µs** (acceptable!)
- Correlation with cos(orbital_phase) = **0.94**

**Root Cause Analysis:**
- The 2.6 µs cosine pattern implies a **~29 ms time offset** in the binary calculation
- JUG currently uses `t_topo_tdb = tdbld - roemer_shapiro_sec/86400` as input to binary model
- The question is: **What time does PINT actually use for its binary model?**

**Key Insight:**
- Error = d(binary)/dt × Δt
- d(binary)/dt = a1 × cos(phase) × 2π/PB ≈ 9×10⁻⁵ s/s
- 2.6 µs error → Δt ≈ 29 ms (which matches typical Shapiro delay magnitude)

## What Still Needs To Be Done

### 1. Investigate PINT's Binary Time Input
Look at PINT source code to determine exactly what time PINT passes to the binary model:
- Does PINT use `tdbld` directly?
- Does PINT subtract Roemer delay before computing binary?
- Does PINT subtract Shapiro delay before computing binary?

Check these PINT files:
- `pint/models/binary_ell1.py`
- `pint/models/stand_alone_psr_binaries/ELL1_model.py`
- Look for how `tt2tb` or `tdbld` is transformed before binary computation

### 2. Test Different Time Inputs
Try computing JUG binary delay with:
1. `tdbld` directly (no correction)
2. `tdbld - roemer_sec/86400` (current approach)
3. `tdbld - binary_sec/86400` (iterative)

Compare which matches PINT best.

### 3. Apply the Fix
Once the correct time input is identified:
- Update the main computation cell `#VSC-1f8a2119` (lines ~518-692)
- Re-run and verify RMS difference drops to < 1 µs

## Key Files and Cells

| Cell ID | Lines | Purpose |
|---------|-------|---------|
| #VSC-37db9bff | 60-78 | Constants (K_DM_SEC fixed here) |
| #VSC-09e4100e | 285-398 | `ell1_binary_delay_full()` function |
| #VSC-1c8496cb | 401-422 | `dm_delay_vectorized()` function |
| #VSC-1f8a2119 | 518-692 | Main JUG computation pipeline |
| #VSC-e1f7c403 | 2701-2775 | Residual comparison plots |

## Key Variables in Kernel

- `jug_binary_sec`: JUG binary delay (current)
- `pint_bin_sec`: PINT binary delay for comparison
- `binary_diff`: JUG - PINT binary (in µs)
- `t_topo_tdb`: Current time input to binary (tdbld - roemer_shapiro)
- `tdbld`: TDB time in MJD (long double)
- `roemer_sec`, `shapiro_sec`: Astrometric delays

## Expected Outcome

After fixing binary time input:
- JUG-PINT difference RMS should drop from 1.93 µs to ~0.7 µs
- JUG residuals RMS should match PINT (~0.8 µs)
- First-harmonic orbital pattern should disappear
