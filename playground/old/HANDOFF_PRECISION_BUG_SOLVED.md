# MK5_safe Precision Bug - SOLVED

## Date: 2025-11-29

## Problem Summary
MK5_safe had 180.8 ns RMS error vs PINT while MK4_ind_TZR_secondTry achieved 2.6 ns RMS.

## Root Causes Found (2 bugs fixed)

### Bug 1: Temporary Debug Code Using PINT's TDB
**Location:** Cell 9, lines 70-78 (around line 809-816 in JSON)

**The Code:**
```python
if self.utc_mjd is not None and self.freq_mhz is not None:
    # TEMPORARY: Use PINT's TDB to verify accuracy issue
    if self.pint_toas is not None:
        print(f"  TEMP DEBUG: Using PINT's TDB with standalone frequencies")
        tdbld = np.array(self.pint_toas.table['tdbld'].value, dtype=np.longdouble)
        mjd_ints = None  # This skips standalone TDB computation!
        mjd_fracs = None
```

**Issue:** When `utc_mjd` was provided, MK5_safe bypassed standalone TDB computation and used PINT's TDB instead.

**Fix Applied:** Removed lines 70-78, kept the correct standalone TDB code (was lines 79-84).

### Bug 2: Standalone TDB Precision Loss
**Location:** Cell 6, line 189 in `compute_tdb_standalone_vectorized()`

**The Code:**
```python
return time_utc.tdb.mjd  # Returns float64, NOT longdouble!
```

**Issue:**
- Astropy's `.mjd` property returns **float64** (~15-16 decimal digits)
- When converting to longdouble later with `np.array(tdbld, dtype=np.longdouble)`, the precision loss had already occurred
- Float64 MJD ~58526.214689902168 only preserves ~8 fractional digits after the decimal
- This causes ~180 ns scatter in the final residuals

**Fix Applied:**
```python
# Return TDB with full precision (jd1 + jd2 for double-double)
# MJD = JD - 2400000.5, preserve precision using two-part representation
tdb_time = time_utc.tdb
# Extract as longdouble: jd1 + jd2 - MJD_offset
MJD_OFFSET = 2400000.5
return np.array(tdb_time.jd1 - MJD_OFFSET, dtype=np.longdouble) + np.array(tdb_time.jd2, dtype=np.longdouble)
```

**Why This Works:**
- Astropy Time uses double-double representation (jd1 + jd2) internally for precision
- jd1 contains the large part (integer days + some fraction)
- jd2 contains the small fractional part (high precision)
- Extracting both and casting to longdouble preserves ~18 decimal digits
- MJD = (jd1 - 2400000.5) + jd2 maintains full precision

## Why MK4_ind_TZR_secondTry Worked
Looking at the code, MK4_ind_TZR_secondTry achieves 2.6 ns RMS by:
1. Computing standalone TDB (validates as 0.000 ns vs PINT)
2. **BUT then using PINT's TDB for the actual phase calculation** (line 200):
   ```python
   tdbld_ld = np.array(self.pint_toas.table['tdbld'].value, dtype=np.longdouble)
   ```

So MK4_ind_TZR_secondTry is NOT fully independent - it validates standalone TDB works, but uses PINT's TDB for calculation.

## Expected Result After Both Fixes
- **RMS**: Should drop from 180.8 ns to ~2-5 ns
- **Independence**: Full independence from PINT (uses standalone TDB with full precision)
- **TDB Validation**: Still 0.000 ns (standalone matches PINT)

## Testing Status
- **Bug 1 fix tested:** ✓ Removed TEMP DEBUG code successfully
  - Result: RMS still 180.8 ns (as expected, Bug 2 still present)

- **Bug 2 fix applied:** ✓ Modified to use jd1+jd2 for precision
  - Result: **SUCCESS! RMS = 2.551 ns** ✓✓✓

## FINAL RESULT ✓
```
Residual difference (JUG - PINT):
  Mean:  -6.062 ns
  RMS:    2.551 ns  ← FIXED! Was 180.8 ns
  Max:    8.308 ns
```

**Achievement:**
- ✓ Full PINT independence (uses standalone TDB)
- ✓ Matches MK4_ind_TZR_secondTry's 2.6 ns RMS
- ✓ Standalone TDB with full longdouble precision
- ✓ TDB validation still 0.000 ns vs PINT

## Next Steps

### 1. Verify Bug 2 Fix Worked
Run the notebook and check output Cell 12 for:
```
Residual difference (JUG - PINT):
  Mean: ??? ns
  RMS:  ??? ns  ← Should be ~2-5 ns now!
  Max:  ??? ns
```

### 2. If RMS is Still High (~180 ns)
The jd1+jd2 approach may not work as expected. Alternative fixes:

**Option A:** Use PINT's TDB (like MK4_ind_TZR_secondTry)
- Replace line 222 in cell 9:
  ```python
  # OLD: tdbld_ld = np.array(tdbld, dtype=np.longdouble)
  # NEW: tdbld_ld = np.array(self.pint_toas.table['tdbld'].value, dtype=np.longdouble)
  ```
- This achieves 2.6 ns RMS but loses full independence
- Standalone TDB still computed and validated for proof of concept

**Option B:** Fix Astropy precision differently
- Check if PINT's `table['tdbld']` uses a different Astropy format
- Investigate how PINT preserves precision internally
- May need to store UTC MJD + clock corrections separately and apply in longdouble

**Option C:** Bypass Astropy entirely
- Compute TDB transformation using ERFA functions directly in longdouble
- More complex but guarantees precision control

### 3. If RMS is Now ~2-5 ns ✓
Success! The jd1+jd2 approach worked. Document:
- Final RMS value
- Confirm TDB validation still 0.000 ns
- Confirm full PINT independence achieved

## Files Modified
1. `/home/mattm/soft/JUG/residual_maker_playground_active_MK5_safe.ipynb`
   - Cell 6 line 189: Changed TDB return to use jd1+jd2
   - Cell 9 lines 70-78: Removed TEMP DEBUG code

## Key Insight: Why Precision Matters
MJD ~58526.214689902168 in different formats:
- **longdouble MJD**: 58526.214689902168... (13 fractional digits preserved)
- **float64 MJD**: 58526.21468990217 (only ~11 fractional digits reliable)
- **longdouble seconds**: 5056664949.207547... (6 fractional digits for sub-second)

At nanosecond precision (1e-9 days = 86.4 ns), losing even 2-3 decimal places causes 100-300 ns errors.

## Testing Commands
```bash
# Run notebook
jupyter nbconvert --to notebook --execute residual_maker_playground_active_MK5_safe.ipynb --output test_output.ipynb

# Extract RMS result
python3 -c "
import json
with open('test_output.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if 'outputs' in cell:
        for out in cell['outputs']:
            if 'text' in out:
                text = ''.join(out['text']) if isinstance(out['text'], list) else out['text']
                if 'Residual difference (JUG - PINT):' in text:
                    print(text)
"
```

## Contact / Handoff
If RMS is not fixed after Bug 2 fix, try Option A (use PINT's TDB) as immediate solution.
Standalone TDB computation is correct (validates 0.000 ns), the challenge is preserving precision through Astropy.
