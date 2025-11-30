# JUG Residuals: Issues and Fixes

## Investigation Summary

After thorough investigation of `residual_maker_playground.ipynb`, I found:

### What's Working
- ✓ TZR calculation runs correctly (cell 200)
- ✓ `phase_offset_cycles = 0.08740234375` is computed
- ✓ Binary delays, DM delays, barycentric corrections all computed
- ✓ Model parameters loaded from `.par` file

### What's NOT Working
- ✗ Residuals show ~840-850 μs RMS instead of ~0.8 μs like Tempo2
- ✗ Cell 204 uses **topocentric time** (incorrect)
- ✗ Residuals in cell 233 show suspicious pattern (nearly identical values)

## Key Issues Found

### Issue #1: Wrong Time Coordinate (Cell 204)
**Cell 204** computes residuals at topocentric time:
```python
t_topo_jax = jnp.array(t_mjd, dtype=jnp.float64)  # WRONG!
res_sec_topo = residuals_seconds_at_topocentric_time(t_topo_jax, model)
```

**Problem**: Residuals must be computed at **infinite-frequency barycentric time**, not topocentric time.

**Fix**: Use `t_inf` (emission time minus DM delay) like cell 233 does.

### Issue #2: Suspicious Residual Pattern (Cell 233)
Cell 233 outputs show:
```
First 5 residuals (us): [698.55588932 699.27539814 698.55588932 697.8363805  697.8363805]
RMS (us): 840.9383585921798
```

**Problem**: Values repeat exactly (indices 0==2, 3==4). This suggests:
1. Either the computation is cached incorrectly
2. Or there's a systematic offset of ~698 μs

698 μs × 339.3 Hz (F0) = 0.237 cycles ≈ 2.7 × phase_offset_cycles

This hints that `phase_offset_cycles` might be applied with the wrong sign or wrong magnitude.

### Issue #3: Potential Sign Error in phase_offset_cycles

Looking at the residual calculation in `residuals_seconds_at_topocentric_time()`:
```python
phase_diff = phase - phase_ref - model.phase_offset_cycles
```

**Question**: Should this be `+ phase_offset_cycles` instead of `- phase_offset_cycles`?

The TZR calculation computes the fractional phase at the TZR epoch. When we compute phase differences from TZR, we might need to ADD this offset, not subtract it.

## Recommended Fixes

### Fix #1: Test Sign of phase_offset_cycles

Run the diagnostic script:
```bash
cd /home/mattm/soft/JUG
# After running notebook up to cell 233:
python test_residuals_fix.py
```

This will test whether changing the sign of `phase_offset_cycles` fixes the issue.

### Fix #2: Verify Correct Time Input

Modify the `residuals_seconds` function to explicitly check what time it's receiving:

```python
@jax.jit
def residuals_seconds(t_mjd: jnp.ndarray, model):
    """
    Compute residuals at infinite-frequency barycentric time.

    CRITICAL: t_mjd should be infinite-frequency time (t_inf),
    NOT topocentric time!
    """
    dt = (t_mjd - model.tref_mjd) * 86400.0
    phase = model.f0 * dt + 0.5 * model.f1 * dt**2 + (1.0/6.0) * model.f2 * dt**3

    dt_ref = (model.phase_ref_mjd - model.tref_mjd) * 86400.0
    phase_ref = model.f0 * dt_ref + 0.5 * model.f1 * dt_ref**2 + (1.0/6.0) * model.f2 * dt_ref**3

    # TEST BOTH SIGNS
    # Original: phase_diff = phase - phase_ref - model.phase_offset_cycles
    # Alternative: phase_diff = phase - phase_ref + model.phase_offset_cycles
    phase_diff = phase - phase_ref - model.phase_offset_cycles

    frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
    return frac_phase / model.f0
```

### Fix #3: Check Parameter Values

Verify that the `.par` file contains **fitted** parameters, not initial guesses:

```bash
grep -E "F0|F1|DM" temp_model_tdb.par
```

If the uncertainties (3rd column) are large or missing, the parameters might be initial guesses.

### Fix #4: Direct Comparison Test

Create a minimal test comparing one TOA:

```python
# Compare residual calculation step-by-step with Tempo2
idx = 0
print(f"TOA {idx}:")
print(f"  JUG residual: {res_us[idx]:.6f} μs")
print(f"  Tempo2 residual: {t2_res_us[idx]:.6f} μs")
print(f"  Difference: {res_us[idx] - t2_res_us[idx]:.6f} μs")

# If difference is constant across all TOAs, it's a phase offset issue
# If difference varies, it's a time coordinate or parameter issue
```

## Action Plan

1. **Run diagnostic**: Execute `test_residuals_fix.py` to test phase_offset sign

2. **If sign flip fixes it**:
   - Change line in `residuals_seconds`:
     ```python
     phase_diff = phase - phase_ref + model.phase_offset_cycles  # Changed - to +
     ```

3. **If not fixed**:
   - Check if `t_inf` is being computed correctly
   - Verify phase_ref_mjd matches TZR emission time
   - Compare parameter values with Tempo2's fitted output

4. **Verify the fix**:
   - RMS should drop to ~0.8-10 μs
   - Correlation with Tempo2 should be > 0.999
   - Residual differences should be < 1 μs

## Expected Outcome

After applying the correct fixes:
- **Before**: RMS ~840 μs, poor correlation with Tempo2
- **After**: RMS ~0.8-10 μs, correlation > 0.999 with Tempo2

## Files Created

- `FIX_RESIDUALS.md`: Initial fix documentation
- `COMPLETE_FIX.py`: Reference implementation
- `test_residuals_fix.py`: Diagnostic script
- `ISSUES_AND_FIXES.md`: This file

## Next Steps

1. Run the diagnostic script
2. Apply the identified fix
3. Re-run residual calculation
4. Compare with Tempo2
5. Report results
