# Fix MK5_safe Accuracy Issue - AI Agent Instructions

## Problem Statement
`residual_maker_playground_active_MK5_safe.ipynb` currently has **180.8 ns RMS** accuracy error vs PINT.
The working reference `residual_maker_playground_active_MK4_ind_TZR_secondTry.ipynb` achieves **2.6 ns RMS** with full PINT independence.

## Root Cause
The issue is a **precision loss bug** in how TDB times are stored:
- MK5_safe pre-converts TDB from MJD to seconds during initialization
- This shifts 5 digits left (×86400), reducing fractional precision from 13 to 8 decimal places
- At nanosecond scale, this creates ~180 ns scatter in residuals

## Solution (Already Working in MK4_ind_TZR_secondTry)
Store TDB in MJD format during initialization, only convert to seconds at calculation time.

## Required Changes to MK5_safe

### Change 1: Store TDB in MJD (not seconds)
**File:** `residual_maker_playground_active_MK5_safe.ipynb`
**Location:** In the `JUGResidualCalculatorFinal` class, `_precompute_all` method

**Current code (around line 842-844):**
```python
# PRE-COMPUTE tdbld * SECS_PER_DAY (saves N multiplications per call!)
tdbld_ld = np.array(tdbld, dtype=np.longdouble)
self.tdbld_sec_ld = tdbld_ld * np.longdouble(SECS_PER_DAY)
```

**Change to:**
```python
# PRECISION FIX: Store TDB in MJD format (convert to seconds only at calculation time)
tdbld_ld = np.array(tdbld, dtype=np.longdouble)
self.tdbld_mjd_ld = tdbld_ld  # Store as MJD, don't pre-convert!
```

### Change 2: Convert TDB to seconds during residual calculation
**File:** `residual_maker_playground_active_MK5_safe.ipynb`
**Location:** In the `compute_residuals` method

**Current code (around line 899-901):**
```python
# Optimized phase: tdbld_sec_ld is pre-computed!
dt_sec = self.tdbld_sec_ld - self.PEPOCH_sec - delay_ld
```

**Change to:**
```python
# PRECISION FIX: Convert TDB to seconds here (not during init)
tdbld_sec = self.tdbld_mjd_ld * np.longdouble(SECS_PER_DAY)
dt_sec = tdbld_sec - self.PEPOCH_sec - delay_ld
```

### Change 3: Update dataclass field name
**File:** `residual_maker_playground_active_MK5_safe.ipynb`
**Location:** In the `JUGResidualCalculatorFinal` dataclass definition (around line 659)

**Current field:**
```python
tdbld_sec_ld: np.ndarray = None  # tdbld * SECS_PER_DAY as longdouble
```

**Change to:**
```python
tdbld_mjd_ld: np.ndarray = None  # tdbld in MJD as longdouble (convert to sec at calc time)
```

## Reference Implementation
The complete working implementation can be found in:
- **File:** `residual_maker_playground_active_MK4_ind_TZR_secondTry.ipynb`
- **Key cells:** Look for comments containing "PRECISION FIX" or "Store TDB in MJD"
- **Result:** Achieves 2.6 ns RMS accuracy with full PINT independence

## How MK4_ind_TZR_secondTry Works

### Independence Features (all in place):
1. **Standalone TDB calculation** - Uses clock files directly (BIPM2024, GPS, MeerKAT)
2. **Standalone TZR delay** - Computes barycentric frequency for DM delays independently
3. **Standalone frequency parsing** - Reads from TIM file
4. **Precision handling** - Stores TDB as MJD until calculation time

### Key Code Sections in MK4_ind_TZR_secondTry:

**TDB Storage (line ~868):**
```python
self.tdbld_mjd_ld = tdbld_ld  # Store as MJD
```

**TDB Conversion (line ~907-909):**
```python
# Convert TDB to seconds here (maintains precision)
tdbld_sec = self.tdbld_mjd_ld * np.longdouble(SECS_PER_DAY)
dt_sec = tdbld_sec - self.PEPOCH_sec - delay_ld
```

## Verification Steps

After making the changes, verify:

1. **Initialization should complete without errors**
   - Check for "Computing TDB standalone" message
   - Check for "TDB validation: max diff = 0.000 ns" (or similar)

2. **Run benchmark cell**
   - Should see timing comparison JUG vs PINT
   - Should see accuracy validation section

3. **Expected result:**
   - RMS accuracy should drop from **180.8 ns** to **~2-5 ns**
   - Mean offset should be near zero (< 10 ns)
   - Max difference should be < 20 ns

4. **Independence confirmed:**
   - No PINT data should be used in calculation (only for validation/comparison)
   - TDB comes from standalone clock chain
   - TZR delay computed independently
   - Frequencies parsed from TIM file

## Additional Context

### Why This Is Better Than Using PINT's TDB
- **Full independence:** No PINT dependency in calculation path
- **Validated accuracy:** Standalone TDB matches PINT to < 1 ns
- **Proper precision:** Storing as MJD preserves 13 fractional decimal places
- **Simple fix:** Only 3 lines of code need to change

### Precision Analysis
- **MJD format:** ~58526.214689902168... (13 fractional digits)
- **Seconds format:** 5056664949.207547... (only 6 fractional digits available)
- **Loss:** Pre-converting loses 7 decimal places of precision
- **Impact:** At nanosecond scale (1e-9 days = 86 ns), this creates 100-300 ns errors
- **Solution:** Delay conversion until after subtraction brings magnitude down

## Testing Notes

The fix has been validated in `MK4_ind_TZR_secondTry.ipynb`:
- 10,408 TOAs tested
- 2.6 ns RMS achieved
- Full independence verified (no PINT in calculation)
- 863x speedup maintained

## Questions for User (if needed)

If the fix doesn't work as expected:
1. Check if TDB is being computed standalone (should see validation message)
2. Check if frequencies are from TIM file (not PINT)
3. Check if TZR delay is being computed independently
4. Compare intermediate values with MK4_ind_TZR_secondTry

## Summary

**Change 3 lines of code in MK5_safe:**
1. Line ~844: Store `tdbld_mjd_ld` instead of `tdbld_sec_ld`
2. Line ~659: Update field name in dataclass
3. Line ~901: Convert MJD→seconds at calculation time

**Result:**
- 180.8 ns RMS → 2-5 ns RMS (70x improvement)
- Full PINT independence maintained
- Validated against working reference implementation
