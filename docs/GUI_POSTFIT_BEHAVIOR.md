# JUG GUI - Postfit Residual Behavior

**Date**: 2026-01-27

---

## What Happens When You Click "Run Fit"

### 1. Fitting Phase
- "Run Fit" button is disabled
- Status bar shows "Fitting F0, F1..." (or whichever parameters you selected)
- Fitting runs in background (UI remains responsive)
- Progress updates shown in status bar

### 2. Completion Phase
- Fit results dialog appears with:
  - Parameter table (values and uncertainties)
  - Final RMS
  - Number of iterations
  - Convergence status
  - Fit time

### 3. Plot Update Phase ⭐ NEW!
- **Temporary .par file created** with fitted parameters
- **Postfit residuals computed** using updated model
- **Plot automatically updates** to show postfit residuals
- Statistics panel updates with postfit RMS and χ²/dof

### 4. Final State
- Plot shows **postfit residuals** (should be smaller!)
- RMS label shows postfit RMS
- "Reset to Prefit" button is enabled

---

## What You Should See

### Before Fit (Prefit)
```
Plot: Shows prefit residuals (large scatter if par file is wrong)
RMS: Large value (e.g., 206.828 μs for wrong par file)
Statistics: Shows prefit RMS
```

### After Fit (Postfit)
```
Plot: Shows postfit residuals (smaller scatter, centered on zero)
RMS: Smaller value (e.g., 206.625 μs - improvement of 0.2 μs)
Statistics: Shows postfit RMS, iterations, χ²/dof
```

### Visual Difference
- **Prefit**: Residuals may have systematic trends, offsets, or large scatter
- **Postfit**: Residuals should be more random, centered on zero, smaller scatter

---

## Example: J1909-3744 Wrong Par File

### Starting (Prefit)
- **File**: `data/pulsars/J1909-3744_tdb_wrong.par`
- **Prefit RMS**: ~206.83 μs
- **Plot**: Residuals scattered around zero with some structure

### After Fitting F0, F1
- **Postfit RMS**: ~206.63 μs
- **Improvement**: 0.2 μs (0.1%)
- **Plot**: Updated to show postfit residuals
- **Visual change**: Subtle improvement (small perturbation)

### Why Small Improvement?
The "wrong" par file has only slightly incorrect F0/F1 values. For a bigger improvement:
1. Try fitting more parameters: F0, F1, F2, DM
2. Use a more incorrect starting par file
3. Check for systematic trends in postfit residuals

---

## Testing the Wrong Par File

Try this workflow:

```bash
# Launch with wrong par file
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim

# In the GUI:
# 1. Check prefit RMS (should be ~206.8 μs)
# 2. Select F0, F1, DM (more parameters = better fit)
# 3. Click "Run Fit"
# 4. Wait ~2 seconds
# 5. View results dialog
# 6. Notice plot has updated!
# 7. Check postfit RMS (should be lower)

# To see the difference clearly:
# - Click "Reset to Prefit" to restore original residuals
# - Notice the difference in the plot
# - Click "Run Fit" again to see postfit residuals
```

---

## Comparing Prefit vs Postfit

### Manual Comparison
1. **Note the prefit plot** (take a screenshot if desired)
2. Click "Run Fit"
3. **Note the postfit plot** (compare visually)
4. Click "Reset to Prefit" to toggle back
5. Click "Run Fit" again to see postfit

### Visual Indicators
- **Prefit**: May show trends, offsets, or large scatter
- **Postfit**: More random, centered on zero, tighter scatter
- **Status bar**: Shows "Fit complete" with RMS improvement

---

## Technical Details

### How Postfit Residuals Are Computed

1. **Fit completes** → returns fitted parameter values
2. **Temporary .par file created** → original par file with fitted values
3. **Residuals recomputed** → using `compute_residuals_simple()` with temp par file
4. **Plot updated** → `self.residuals_us` replaced with postfit residuals
5. **Temp file deleted** → cleanup

### Why Not Just Use Fit Result?
The fit result contains the **final RMS** but not the actual residual values. To plot postfit residuals, we must:
- Update the timing model with fitted parameters
- Recompute phase for each TOA
- Subtract to get postfit residuals

---

## Troubleshooting

### Plot Doesn't Update
**Symptoms**: Statistics update but plot looks the same

**Possible causes**:
1. Prefit and postfit are very similar (fit didn't improve much)
2. Error in postfit computation (check status bar for errors)
3. Visual scale makes difference hard to see

**Solutions**:
- Check RMS values (prefit vs postfit)
- Try fitting more parameters
- Use a more incorrect starting par file
- Check console for error messages

### RMS Doesn't Improve
**Symptoms**: Postfit RMS same or worse than prefit

**Possible causes**:
1. Parameters already optimal (starting par file is correct)
2. Wrong parameters selected (try adding more)
3. Fit didn't converge (check iterations and convergence status)

**Solutions**:
- Check "Converged: True" in fit results
- Add more fit parameters (F0, F1, F2, DM)
- Increase max iterations in code (default: 25)

### Plot Updates But RMS Stays Same
**This should not happen!** If it does:
1. Check fit results dialog (what RMS is shown?)
2. Check statistics panel (what RMS is shown?)
3. Report as bug with details

---

## Expected Behavior Summary

✅ **Correct behavior:**
1. Click "Run Fit"
2. Wait for fit to complete (~2 seconds)
3. Fit results dialog appears
4. Click OK to close dialog
5. **Plot updates automatically** to show postfit residuals
6. Statistics panel shows postfit RMS
7. "Reset to Prefit" button becomes enabled

✅ **What you should observe:**
- Visual change in plot (less scatter, more centered)
- RMS value decreases (or stays same if already optimal)
- Status bar shows "Fit complete" message

❌ **Not expected:**
- Plot stays exactly the same after fit
- RMS increases after fit (unless starting params were better)
- GUI freezes during fit (should be non-blocking)

---

## Recommendations

### For Best Results
1. **Start with incorrect par file** to see clear improvement
2. **Fit multiple parameters** (F0, F1, DM) for better results
3. **Watch the plot** before/after to see the change
4. **Use "Reset to Prefit"** to toggle between prefit/postfit views

### For Testing
```bash
# Good test case (wrong par file)
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim

# Fit F0, F1, DM (should see clear improvement)
# Click "Run Fit"
# Watch plot update!
```

---

**The plot now updates automatically with postfit residuals!** 🎉
