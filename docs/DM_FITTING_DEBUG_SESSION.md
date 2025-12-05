# DM Fitting Debug Session - December 4, 2025

## Current Status: PARTIALLY FIXED, ONE REMAINING BUG

### What's Fixed ✅
1. **Residual recomputation bug**: When DM parameters are fitted, the fitter now recomputes full timing residuals (including DM delays) in each iteration using `compute_residuals_simple()`.
2. **Convergence threshold**: Documented that `convergence_threshold=1e-10` should be used for DM fitting (not the default 1e-14).
3. **Parameter matching**: Fixed regex to handle multiple spaces in par files using `line.split()[0]`.

### What's Still Broken ❌
**DM parameters are NOT being updated during fitting!**

### Symptom
When fitting with the wrong par file (`J1909-3744_tdb_wrong.par`):
- Initial DM: **10.590712224111** pc/cm³ (WRONG)
- Expected DM: **10.390712224111** pc/cm³ (0.2 pc/cm³ difference)
- After 3 iterations: **10.590712224111** pc/cm³ (NO CHANGE!)
- F0 and F1 **do** update correctly
- RMS improves slightly: 206.828 → 206.626 μs (from F0/F1 changes only)

### What I've Verified

1. ✅ **DM derivatives are correct**:
   ```
   DM derivative range: 1.5e-3 to 5.0e-3 s/(pc/cm³)
   DM1 derivative range: -2.4 to 9.3 s/(pc/cm³/day)
   All non-zero, shape correct (10408,)
   ```

2. ✅ **Temp par file creation works**:
   ```python
   # Manually tested - temp file DOES contain updated DM value
   grep "^DM" temp_file → "DM  10.390712224111"  # Correct!
   ```

3. ✅ **Frequencies are passed correctly**:
   ```
   freq_mhz range: 907.7 - 1659.4 MHz
   ```

4. ✅ **Parameter update logic exists**:
   ```python
   # Line 493: Update params dict
   params[param] = param_values_curr[i]
   
   # Line 610: Update param_values_curr
   param_values_curr = [param_values_curr[i] + delta_params[i] for ...]
   ```

### Where I Am Now

I was about to check **what `delta_params` contains** after the WLS solve. The WLS solver (`np.linalg.lstsq`) should be computing:

```
M^T W M · Δp = M^T W · r
```

where:
- M = design matrix (derivatives)
- W = weight matrix (1/σ²)
- r = residuals
- Δp = parameter updates

**Hypothesis**: Either:
1. The WLS solve is producing `delta_params[2]` (for DM) ≈ 0, OR
2. The update is being applied but then immediately overwritten

### Next Steps

1. **Add debug output to see `delta_params`** after WLS solve:
   ```python
   # After line 598 in optimized_fitter.py
   if verbose and iteration == 0:
       print("\nFirst iteration delta_params:")
       for i, param in enumerate(fit_params):
           print(f"  Δ{param} = {delta_params[i]:.6e}")
   ```

2. **Check if design matrix M is singular** (rank-deficient):
   ```python
   print(f"Design matrix rank: {np.linalg.matrix_rank(M_weighted)}")
   print(f"Expected rank: {len(fit_params)}")
   ```

3. **Check if DM column is degenerate** with another column:
   ```python
   # Check correlation between DM and other parameters
   corr = np.corrcoef(M.T)
   print(f"DM-F0 correlation: {corr[2, 0]:.6f}")
   print(f"DM-F1 correlation: {corr[2, 1]:.6f}")
   ```

4. **If delta_params IS non-zero**, trace where the update gets lost:
   - Add print after line 610: `print(f"Updated params: {param_values_curr}")`
   - Add print after line 493: `print(f"params dict: {params}")`
   - Check if temp file is being read correctly by `compute_residuals_simple()`

### Key Code Locations

- **Main iteration loop**: `optimized_fitter.py` lines 488-640
- **Residual computation (DM path)**: lines 497-531
- **Design matrix construction**: lines 558-587
- **WLS solve**: lines 595-598
- **Parameter update**: line 610

### Expected Behavior

For the wrong par file, the **first iteration** should produce:
```
ΔDM ≈ -0.2 pc/cm³  (to correct 10.59 → 10.39)
```

This should reduce RMS from ~207 μs to ~0.4 μs (matching the correct par file).

### Files Modified So Far

1. ✅ `jug/fitting/optimized_fitter.py` - Added DM residual recomputation (lines 497-555)
2. ✅ `DM_FITTING_FIX.md` - Technical documentation
3. ✅ `DM_FITTING_FIX_SUMMARY.md` - User-facing summary
4. ✅ `DM_FITTING_COMPLETE.md` - Updated with convergence notes

### Test Command

```bash
cd /home/mattm/soft/JUG
python3 << 'EOF'
from jug.fitting.optimized_fitter import fit_parameters_optimized
from pathlib import Path

result = fit_parameters_optimized(
    par_file=Path("data/pulsars/J1909-3744_tdb_wrong.par"),
    tim_file=Path("data/pulsars/J1909-3744.tim"),
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=25,
    convergence_threshold=1e-10,
    verbose=True
)

print(f"\nInitial DM: 10.590712224111")
print(f"Expected:   10.390712224111")
print(f"Fitted DM:  {result['final_params']['DM']:.12f}")
print(f"Difference: {abs(result['final_params']['DM'] - 10.390712224111):.6f}")
EOF
```

### Resume Point

Start by adding this debug block after line 598 in `optimized_fitter.py`:

```python
# DEBUG: Check what WLS solver produced
if verbose and iteration == 0:
    print(f"\n=== DEBUG: First iteration WLS solve ===")
    print(f"Design matrix shape: {M.shape}, rank: {np.linalg.matrix_rank(M_weighted)}")
    print(f"Parameter updates (delta_params):")
    for i, param in enumerate(fit_params):
        print(f"  Δ{param:<4s} = {delta_params[i]:>15.6e}")
        print(f"    {param_values_curr[i]:.12f} → {param_values_curr[i] + delta_params[i]:.12f}")
    print(f"=====================================\n")
```

Then run the test command to see if `delta_params[2]` (DM) is zero or non-zero.

---

**Status**: In progress - need to trace WLS solver output  
**Time invested**: ~1.5 hours  
**Confidence**: 80% this is a simple bug in parameter update bookkeeping
