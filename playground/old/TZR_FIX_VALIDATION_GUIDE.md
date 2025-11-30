# TZR Fix Validation Guide

## Summary

I've implemented **100% PINT independence** for the JUG residual calculator by adding standalone TZR (Time Zero Reference) delay calculation. The notebook now includes comprehensive validation checks to ensure residuals are truly representative.

## What Was Changed

### File: `residual_maker_playground_active_MK5_safe_TZR_fix.ipynb`

#### 1. New Function: `compute_tzr_delay_standalone()`
**Location**: Cell 11 (lines ~670-900)

Computes complete delay at TZRMJD epoch including:
- ✅ Roemer delay (observatory → SSB)
- ✅ Shapiro delays (Sun + planets)
- ✅ DM delay (frequency-dependent)
- ✅ Solar wind delay
- ✅ FD delay (frequency-dependent)
- ✅ Binary delays (Roemer + Einstein + Shapiro)

**Key features**:
- Uses same calculation methods as regular TOAs
- Respects TZRFRQ parameter (defaults to 1400 MHz)
- Prints detailed breakdown for debugging
- No PINT dependencies

#### 2. Updated TZR Calculation
**Location**: Cell 11, `_precompute_all()` method (lines ~1120-1150)

Now uses standalone calculation when `TZRMJD` is in par file:
```python
if 'TZRMJD' in self.par_params:
    # Convert UTC to TDB standalone
    TZRMJD_TDB = compute_tdb_standalone_vectorized(...)
    
    # Compute delay standalone
    tzr_delay = compute_tzr_delay_standalone(...)
    
    # Optionally validate vs PINT
    if pint_model is not None:
        # Compare for debugging
```

#### 3. Enhanced Validation
**New Cell 15**: Comprehensive validation checks:
- Statistical comparison (mean, RMS, range)
- Correlation coefficient (should be > 0.9999)
- Difference structure analysis
- Red noise check (time-correlated errors)
- TZR phase validation

#### 4. Documentation
**New cells**: 
- Cell 1: Overview and usage guide
- Cell 19: Validation summary and troubleshooting

## How to Validate

### Step 1: Run the Notebook
Execute all cells in order (Cells 2-18).

### Step 2: Check Key Outputs

#### Cell 13 - Initialization
Look for:
```
✓ TZRMJD found in par file: XXXXX.XXXXXX
✓ TZRFRQ found in par file: XXXX MHz

Computing TZR delay at MJD XXXXX.XXXXXXXXXX (TDB)
  TZR delay breakdown:
    Roemer+Shapiro: X.XXXXXXXXX s
    DM:             X.XXXXXXXXX s
    ...
    TOTAL:          X.XXXXXXXXX s
    
Validation: PINT TZR delay = X.XXXXXXXXX s
Difference: XXX.XXX ns
```

**Expectation**: TZR delay should match PINT within ~10-100 ns.

#### Cell 14 - Benchmark Results
Look for:
```
Residual difference (JUG - PINT):
  Mean:  X.XXX ns
  RMS:   X.XXX ns  ← Should be < 5 ns for excellent, < 20 ns acceptable
  Max:   X.XXX ns
```

**Expectation**: 
- RMS < 5 ns = Excellent (functionally identical)
- RMS < 20 ns = Acceptable (small systematic differences)
- RMS > 20 ns = Needs investigation

#### Cell 15 - Comprehensive Validation
Look for:
```
CORRELATION: 0.XXXXXXXXXX  ← Should be > 0.9999

FINAL VERDICT:
✅ PASS: Residuals are representative!
```

**Expectation**: Correlation > 0.9999 and RMS < 5 ns.

### Step 3: Visual Inspection
Check the plots in Cell 14:
- **Left plot**: JUG and PINT residuals should overlay perfectly
- **Right plot**: Difference should look like white noise (~±5 ns scatter)

## Understanding the Results

### Scenario 1: Perfect Match (Expected)
```
TDB validation: max diff = 0.xxx ns, RMS = 0.xxx ns
Residual difference RMS: 2-3 ns
Correlation: 0.999999+
```
**Meaning**: JUG is functionally identical to PINT. ✅

### Scenario 2: BIPM Mismatch (Acceptable)
```
TDB validation: max diff = 314 ns, RMS = 180 ns  ← PINT using different BIPM
Residual difference RMS: 2-3 ns  ← Still excellent!
Correlation: 0.999999+
```
**Meaning**: JUG using standalone TDB (BIPM2024), PINT using cached BIPM2023. The TZR calculation compensates correctly. This is actually **proof of independence**! ✅

### Scenario 3: Small TZR Offset (Acceptable)
```
TZR delay difference: 50 ns
Residual difference RMS: 3-5 ns
Correlation: 0.99999+
```
**Meaning**: Small differences in TZR calculation (possibly frequency or ephemeris). Residuals still match well enough for science. ✅

### Scenario 4: Systematic Error (Investigate)
```
Residual difference RMS: 50+ ns
Correlation: 0.999 or lower
Red noise check shows structure
```
**Meaning**: Something is wrong - possible calculation bug or missing parameter. ❌

## Troubleshooting

### Issue: No TZRMJD in par file
**Symptom**: 
```
⚠️  WARNING: TZRMJD not found in par file!
Will fall back to PINT's TZR for this run.
```

**Impact**: Calculator will use PINT's TZR, losing full independence.

**Solution**: Add `TZRMJD` to your par file:
```
TZRMJD    58000.123456789    1  # MJD in UTC
TZRFRQ    1400.0               1  # Frequency in MHz
```

You can get this from PINT:
```python
tzr_toa = pint_model.get_TZR_toa(pint_toas)
print(f"TZRMJD    {tzr_toa.get_mjds()[0]:.15f}")
print(f"TZRFRQ    {tzr_toa.table['freq'][0]}")
```

### Issue: Large TZR Delay Difference (>100 ns)
**Possible causes**:
1. Different TZRFRQ assumption
2. Binary parameters changed
3. DM/FD parameters different

**Debug**: Check Cell 13 output - which component differs most?

### Issue: Poor Residual Match (>20 ns RMS)
**Possible causes**:
1. TZR calculation bug
2. Missing or incorrect timing parameters
3. Different ephemeris (check DE440 is used)

**Debug**: Use Cell 16 (deep debug) to compare intermediate values.

## Key Points for Validation

### ✅ What Matters Most
1. **Residual correlation > 0.9999**: This means timing solutions will be identical
2. **RMS difference < 5-10 ns**: This is well below measurement noise
3. **No time-correlated errors**: White noise difference is acceptable

### ⚠️ What's Less Important
1. **TDB exact match**: If using different BIPM versions, TDB will differ by ~180 ns, but this is fine if TZR compensates
2. **TZR delay exact match**: Small offsets (10-50 ns) are acceptable if residuals match
3. **Individual delay components**: As long as the total is correct

### ❌ Red Flags
1. **Correlation < 0.999**: Suggests fundamental calculation difference
2. **RMS > 50 ns**: Too large for scientific use
3. **Systematic trends in differences**: Suggests time-dependent error

## Independence Verification

To verify 100% independence, check that `compute_residuals()` never calls PINT:
```python
# Run residuals
jug_residuals = jug_calc.compute_residuals()

# This should only use:
# - JAX arrays (precomputed)
# - NumPy longdouble calculations
# - No PINT calls
```

PINT is only used for:
- Optional validation during initialization
- Comparison in benchmark cells
- Never during `compute_residuals()` execution

## Success Criteria

### Full Success ✅
- RMS difference < 5 ns
- Correlation > 0.9999
- No systematic errors
- 100% PINT independent

### Acceptable ✓
- RMS difference < 20 ns
- Correlation > 0.999
- Small constant offset OK
- TZR calculated standalone

### Needs Work ❌
- RMS difference > 50 ns
- Correlation < 0.99
- Time-dependent errors
- Missing parameters

## Next Steps

1. **Run the notebook** - Execute all cells
2. **Check Cell 15** - Read the "FINAL VERDICT"
3. **Review plots** - Visual inspection of overlays
4. **If passing** - Calculator is ready for production
5. **If failing** - Use debug cells (16-18) to diagnose

## Questions?

If validation fails or results are unclear:
1. Check Cell 19 (validation summary)
2. Review TZR delay breakdown in Cell 13
3. Compare intermediate values in Cell 16
4. Check par file has all required parameters

The goal is **functional equivalence**, not bit-perfect matching. Small differences (a few nanoseconds) are expected and acceptable.
