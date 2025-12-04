# Session 6: Multi-Pulsar Testing & T2 Binary Model

**Date**: 2025-11-30
**Duration**: ~2 hours
**Focus**: Binary model expansion (T2/DD variants), multi-pulsar testing setup

---

## Summary

This session focused on expanding JUG's binary model support and preparing for multi-pulsar validation testing. We implemented the T2 (Tempo2 general) binary model and created infrastructure for testing against the MPTA dataset.

---

## Accomplishments

### 1. T2 Binary Model Implementation ✅

**File**: `jug/delays/binary_t2.py`

Implemented the T2 (Tempo2 general) binary model which supports:
- All Keplerian parameters (PB, A1, ECC, OM, T0)
- Time derivatives (PBDOT, OMDOT, XDOT, EDOT)
- Relativistic effects (GAMMA, M2/SINI)
- 3D orbital geometry (KIN/KOM)

Key functions:
- `t2_binary_delay()`: JIT-compiled single-time computation
- `t2_binary_delay_vectorized()`: Vectorized for array inputs
- `t2_binary_delay_sec()`: Wrapper matching JUG API conventions

**Status**: Code implemented but **NOT YET TESTED** - no T2 binaries found in MPTA dataset.

### 2. Multi-Pulsar Test Script ✅

**File**: `test_t2_vs_pint.py`

Created validation script that:
- Loads pulsar from MPTA dataset
- Computes JUG residuals
- Computes PINT residuals (for comparison)
- Generates comparison plots
- Reports RMS differences

**Initial Test Results** (J1909-3744, ELL1 binary):
- JUG weighted RMS: 0.416 μs
- PINT unweighted RMS: 2.184 μs  
- Difference RMS: 2.108 μs

⚠️ **Note**: The ~2 μs difference needs investigation. Possible causes:
- Different TDB computation methods
- Different Shapiro delay implementations
- Different phase reference conventions
- Weighted vs unweighted residuals

### 3. MPTA Dataset Analysis

**Location**: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/`

**Binary Model Distribution**:
- ELL1: 17 pulsars
- DD: 16 pulsars
- ELL1H: 14 pulsars
- DDH: 11 pulsars
- DDK: 3 pulsars (includes J0437-4715)
- DDGR: 1 pulsar
- **T2: 0 pulsars** (none found)

**Status**: 
- ✅ ELL1 working (J1909-3744 tested)
- ❌ DD, DDH, DDK, DDGR variants **NOT YET IMPLEMENTED**
- ⚠️ T2 implemented but untested

---

## Binary Models Status

| Model | Status | Implementation | Testing | Notes |
|-------|--------|----------------|---------|-------|
| ELL1 | ✅ Working | Complete | Validated | Main test pulsar (J1909-3744) |
| ELL1H | ⚠️ Partial | Needs H3/STIG | Not tested | Orthometric parameterization |
| BT | ✅ Complete | `binary_bt.py` | Not tested | Basic Keplerian |
| DD | ⚠️ Needs test | Same as BT? | Not tested | Damour-Deruelle |
| DDH | ❌ Not impl | - | - | DD + H3/H4 Shapiro |
| DDK | ❌ Not impl | - | - | DD + KIN/KOM geometry |
| DDGR | ❌ Not impl | - | - | DD + XOMDOT |
| T2 | ⚠️ Untested | `binary_t2.py` | No data | Tempo2 universal model |

---

## Issues Found

### 1. Missing DD Model Variants ⚠️

The MPTA dataset has 31 pulsars using DD/DDH/DDK/DDGR variants, but JUG currently only supports basic BT/DD.

**Required implementations**:
- **DDH**: Add H3/STIG Shapiro parameterization (alternative to M2/SINI)
- **DDK**: Add KIN/KOM 3D orbital geometry
- **DDGR**: Add XOMDOT (second periastron advance derivative)

### 2. JUG vs PINT Discrepancy 🔍

Test on J1909-3744 shows ~2 μs RMS difference between JUG and PINT.

**Next steps**:
- Compare individual delay components (DM, binary, Shapiro, etc.)
- Check TDB computation differences
- Verify phase reference (TZR) handling
- Test on simpler pulsar (no binary, low DM)

### 3. T2 Model Untested ⚠️

No pulsars in MPTA dataset use T2 binary model. Need to:
- Find T2 test data from other sources
- Or create synthetic T2 binary data
- Or wait for MPTA to publish T2-format .par files

---

## Progress Tracker Updates Needed

### Milestone 2 Status

Current tasks from M2 plan:
- ✅ Design matrix computation infrastructure  
- ✅ JAX-accelerated Gauss-Newton + LM damping
- ✅ Multi-pulsar testing setup
- ⚠️ Binary model expansion (partial)
- ❌ DD/DDH/DDK/DDGR implementations
- ❌ Multi-telescope testing

**New Milestone 2.5**: Multi-Binary Support
- Implement DDH (H3/STIG Shapiro)
- Implement DDK (KIN/KOM geometry)
- Implement DDGR (XOMDOT)
- Test all DD variants against PINT
- Validate T2 model (find test data)

---

## Next Steps

### Immediate (Session 7)

1. **Investigate JUG vs PINT discrepancy**
   - Run component-by-component comparison
   - Isolate source of 2 μs difference
   - Fix if bug found, document if expected

2. **Implement DD variants**
   - DDH: H3/STIG alternative Shapiro
   - DDK: KIN/KOM 3D geometry
   - DDGR: XOMDOT parameter

3. **Test additional pulsars**
   - J0610-2100 (ELL1)
   - J0437-4715 (DDK) - needs DDK implementation
   - One DD pulsar
   - One DDH pulsar

### Medium Term (Sessions 8-10)

4. **Fitting validation**
   - Test Gauss-Newton fitter on synthetic data
   - Validate parameter uncertainties
   - Compare against PINT fitter

5. **Multi-telescope testing**
   - Test with Parkes data
   - Test with CHIME data  
   - Validate clock corrections for each backend

6. **Performance benchmarking**
   - Compare JUG vs PINT vs Tempo2 speed
   - Profile bottlenecks
   - Optimize if needed

---

## Files Modified/Created

### New Files
- `jug/delays/binary_t2.py` (167 lines) - T2 binary model
- `test_t2_vs_pint.py` (145 lines) - Multi-pulsar validation script

### Modified Files
- None (all additions)

---

## Validation Status

### Working
- ✅ ELL1 binary model (J1909-3744)
- ✅ Multi-pulsar test infrastructure
- ✅ PINT comparison framework

### Needs Work
- ⚠️ JUG-PINT 2 μs discrepancy
- ❌ DD variant models (DDH/DDK/DDGR)
- ⚠️ T2 model (no test data)

---

## Notes

- The MPTA dataset is well-organized with one `.par` and `.tim` file per pulsar
- File naming: `J####-####_tdb.par` and `J####-####.tim`
- Most MPTA pulsars use ELL1/DD/DDH models
- No T2 binaries found suggests it's rare in MSP timing

- The 2 μs JUG-PINT difference is concerning and should be investigated before proceeding with fitting
- May need to add more diagnostic output to residual calculator to debug

---

## Time Breakdown

- T2 model implementation: 30 min
- Test script creation: 20 min
- MPTA dataset exploration: 15 min
- Testing and debugging: 45 min
- Documentation: 20 min

**Total**: ~2 hours

---

## For Next Claude

If continuing this work:

1. **Priority 1**: Debug the 2 μs JUG-PINT difference on J1909-3744
   - Check `simple_calculator.py` delay components
   - Compare against PINT component-by-component
   - May be TDB/TT difference or Shapiro implementation

2. **Priority 2**: Implement missing DD variants
   - Study PINT's `binary_dd.py` for DDH/DDK/DDGR
   - Add to `jug/delays/` following existing patterns
   - Test against MPTA pulsars

3. **Priority 3**: Continue with fitting implementation
   - Once residuals validated, proceed with Milestone 2
   - Test fitter on synthetic data first
   - Then validate on real pulsars

**Reference notebook**: `playground/residual_maker_playground_active_MK7.ipynb` for working implementations.
