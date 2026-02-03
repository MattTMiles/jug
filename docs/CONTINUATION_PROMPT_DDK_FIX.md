# JUG Continuation Prompt for Next Agent

**Date**: 2026-02-03
**Last Agent**: Claude (Copilot CLI Agent)
**Project**: JUG (JAX-based Pulsar Timing Software)
**Location**: `/home/mattm/soft/JUG`

## Current Status

JUG is a JAX-based pulsar timing software that computes timing residuals and fits timing model parameters. Most functionality is working well:

| Feature | Status | Notes |
|---------|--------|-------|
| Spin parameter fitting (F0, F1, ...) | ✅ Working | Matches PINT to 20 digits |
| DM parameter fitting (DM, DM1, ...) | ✅ Working | Full polynomial DM model |
| Astrometry fitting (RAJ, DECJ, PMRA, PMDEC, PX) | ✅ Working | PINT-style damped fitting |
| Binary fitting (ELL1 parameters) | ✅ Working | PB, A1, TASC, EPS1, EPS2, M2, SINI, PBDOT |
| ELL1/ELL1H binary model | ✅ Working | ~1 μs agreement with Tempo2 |
| **DD binary model** | ❌ **BUG** | Pre-fit residuals differ from Tempo2/PINT |
| DDK Kopeikin terms | ⏳ Not tested | Fix DD first |
| GUI | ✅ Working | PySide6 + pyqtgraph |

## Immediate Task: Fix DD Binary Model Pre-fit Residuals

The DD binary model (tested with J1713+0747) shows **pre-fit residuals that differ significantly from Tempo2/PINT**. This is NOT a fitting issue - the problem is in the residual computation itself, likely the binary delay calculation or how it's applied.

### Key Findings from Debugging

1. **ELL1 binary model works correctly** - J0613-0200 (ELL1H) shows JUG matching Tempo2:
   - JUG weighted RMS: 1.05 μs
   - Tempo2 weighted RMS: 0.97 μs
   - PINT weighted RMS: 2.99 μs (PINT is the outlier here)

2. **DD pre-fit residuals are wrong** - J1713+0747 (DD binary):
   - JUG post-fit weighted RMS: 2.77 μs
   - PINT post-fit weighted RMS: 0.17 μs
   - Tempo2 post-fit weighted RMS: 0.17 μs
   - **The pre-fit residuals differ, so this is NOT a fitting problem**

3. **Fitting parameters match** - All three software packages fit for the same parameters (verified by checking free parameters in each).

### The Problem

The bug is in the **residual computation pipeline for DD binary models**, specifically:
- Either the DD binary delay calculation itself
- Or how the binary delay is applied in the timing model
- Or the time passed to the binary delay function

The DD algorithm was previously thought correct (<1 ns when using PINT's exact times), but pre-fit residuals are wrong, meaning the issue is upstream - in how times or parameters are passed to the binary delay function.

### Files to Investigate

1. **`jug/residuals/simple_calculator.py`** - Main residual calculator
   - Lines 600-700: TZR calculation section
   - Check what time is passed to binary delay computation
   - Compare pre-fit residuals directly with Tempo2 output

2. **`jug/delays/combined.py`** - JAX-compiled delay kernel
   - Lines 90-95: `t_topo_tdb` calculation
   - How binary delay is incorporated

3. **`jug/delays/binary_dd.py`** - DD binary model implementation
   - Compare with PINT's `pint/models/binary_dd.py`
   - Check time units and parameter conversions

4. **`jug/delays/binary_dispatch.py`** - Binary model routing
   - Verify DD parameters are passed correctly

### Recommended Debugging Approach

1. **Compare pre-fit residuals directly**:
   ```bash
   cd /home/mattm/soft/JUG
   mamba activate discotech
   
   # Get Tempo2 pre-fit residuals
   tempo2 -f /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.par \
          /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.tim \
          -nofit -residuals
   
   # Compare with JUG pre-fit residuals (no fitting)
   ```

2. **Trace binary delay computation**:
   - Print the exact time passed to DD binary delay in JUG
   - Print the exact time used by PINT for the same TOA
   - Compare DD delay values for individual TOAs

3. **Check for unit/convention differences**:
   - Time: MJD vs seconds, TDB vs TCB
   - Angles: radians vs degrees for OM, OMDOT
   - Orbital period: days vs seconds

### Test Data

- **J1713+0747 (DD)**: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.par` and `.tim`
- **J0613-0200 (ELL1H)**: Same directory - use as working reference
- **J0437-4715 (DDK)**: Same directory - test after DD is fixed

### Environment Setup

```bash
# Activate environment with PINT, Tempo2, and JUG
mamba activate discotech

# JUG is installed in dev mode from /home/mattm/soft/JUG
# PINT available via: import pint
# Tempo2 available via: tempo2 command

# Run JUG GUI
jug-gui

# Run JUG fitting
jug-fit pulsar.par pulsar.tim --fit F0 F1
```

### Success Criteria

- DD model pre-fit residuals match Tempo2 to <100 ns RMS
- DD model post-fit weighted RMS matches Tempo2/PINT to <100 ns
- Update `docs/JUG_PROGRESS_TRACKER.md` with findings

### Key Documents

- `docs/JUG_PROGRESS_TRACKER.md` - Main progress tracker (2600+ lines)
- `docs/QUICK_REFERENCE.md` - User guide and API reference
- `docs/DDK_TESTING_STATUS.md` - DDK testing results
- `docs/MODEL_ARCHITECTURE.md` - How to add new parameters

## After Fixing DD

Once DD is fixed:
1. Test DDK model (adds Kopeikin annual orbital parallax terms)
2. Update progress tracker with validation results
3. Continue to M7 (White Noise Models - EFAC/EQUAD/ECORR)

## Important Notes

- **Use DE440 ephemeris** (more accurate than DE421)
- **Make minimal, surgical code changes** - don't refactor working code
- **ELL1 is the reference** - it works correctly, so compare DD implementation differences
- The user (Matt) needs JUG to match PINT/Tempo2 to nanosecond precision
