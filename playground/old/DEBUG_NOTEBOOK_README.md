# JUG Debug Notebook - README

## File: `residual_maker_playground_claude_debug.ipynb`

This is a **clean, streamlined version** of the JUG residuals code with the bug fix applied.

## What's Included

### 1. Essential Setup
- Imports (JAX, NumPy, Astropy, etc.)
- Constants (speed of light, DM constant, etc.)
- Data file loading (observatories, ephemeris, IERS)

### 2. File Parsers
- `.par` file parser (timing parameters)
- `.tim` file parser (TOA data)
- Tempo2 output loaders (for comparison)

### 3. Core JUG Code
- `SpinDMModel` dataclass (registered as JAX pytree)
- `spin_phase()` - compute rotational phase
- `dm_delay_sec()` - DM delay calculation
- **`residuals_seconds()` - FIXED residual function**

### 4. TZR Phase Offset Calculation (NEW!)

**NEW ADDITION**: The notebook now includes a complete TZR calculation that:
1. Reads TZRMJD, TZRFRQ, TZRSITE from the .par file
2. Finds the closest TOA to the TZR epoch in Tempo2 data
3. Uses Tempo2's BAT (barycentric arrival time) at TZR
4. Computes binary delay at TZR using ELL1 model:
   - Roemer delay (orbital light travel time)
   - Einstein delay (γ term)
   - Shapiro delay (gravitational time delay from companion)
5. Subtracts binary delay to get emission time
6. Computes DM delay at TZR frequency
7. Subtracts DM delay to get infinite-frequency time
8. Computes spin phase at infinite-frequency time
9. Extracts fractional phase (-0.5 to 0.5 cycles)
10. Updates the model with correct `phase_offset_cycles`

**Expected TZR value**: ~0.087 cycles (based on J1909-3744 parameters)

### 5. The Residual Fix Applied

**Problem**: The residual function was subtracting `phase_offset_cycles` twice:
```python
# OLD - WRONG
phase_diff = phase - phase_ref - model.phase_offset_cycles
```

**Solution**: Removed the double-subtraction:
```python
# NEW - FIXED
phase_diff = phase - phase_ref
```

**Why**: `phase_ref` is already the absolute phase at the TZR epoch, which includes the phase offset. Subtracting `phase_offset_cycles` again was causing additional systematic offset.

### 6. Testing & Comparison
- Loads Tempo2 barycentric times and residuals
- Computes JUG residuals using the same times
- Compares results (RMS, correlation, differences)
- Generates diagnostic plots

### 7. Diagnostic Output
- Checks if `phase_offset_cycles` is set correctly
- Shows residual statistics
- Suggests corrections if needed

## How to Use

1. **Ensure data files are available**:
   - `temp_model_tdb.par` - timing parameters
   - `J1909-3744_NANOGrav_12yv3.gls.tim` - TOA data (or adjust path)
   - `temp_pre_general2.out` - Tempo2 residuals for comparison
   - `temp_pre_components_next.out` - Tempo2 BAT for testing
   - `data/observatory/`, `data/ephemeris/`, `data/earth/` - reference data

2. **Run the notebook**:
   ```bash
   jupyter notebook residual_maker_playground_claude_debug.ipynb
   ```

3. **Execute cells in order** - they build on each other

4. **Check results**:
   - Look for "SUCCESS" message if residuals match
   - Check the comparison plots
   - Review diagnostic output

## Expected Results

### If Fix Works
- **JUG RMS**: < 10 μs (similar to Tempo2)
- **Correlation**: > 0.999
- **RMS Difference**: < 1 μs
- **Message**: "✓✓✓ SUCCESS! JUG residuals match Tempo2!"

### If Fix Doesn't Work
Possible issues:
1. `phase_offset_cycles` not set (should be ~0.087 from TZR)
2. Using wrong time coordinate (should be infinite-frequency barycentric)
3. Parameter mismatch (initial vs fitted parameters)

## Files Created by This Notebook

- `jug_vs_tempo2_comparison.png` - Comparison plots

## Differences from Original Notebook

**Removed**:
- PINT integration (not needed for core functionality)
- Multiple experimental residual functions
- Extensive debugging cells
- Full barycentric correction code (uses Tempo2 BAT instead for main dataset)

**Added**:
- Standalone TZR phase offset calculation
- ELL1 binary delay model (for TZR computation only)
- Comprehensive step-by-step output showing all TZR calculations

**Kept**:
- Essential data loading
- Fixed residual calculation
- Tempo2 comparison
- Diagnostic output

**Focused on**:
- Computing correct TZR phase offset
- Testing the fix
- Comparing with Tempo2
- Clean, readable code with detailed comments

## Next Steps After Running

1. **Check TZR calculation output**: Look for the section "TZR PHASE OFFSET CALCULATION" in the output. It should show:
   - Binary delays at TZR (Roemer, Einstein, Shapiro)
   - TZR emission time
   - DM delay at TZR
   - TZR infinite-frequency time
   - **Phase offset: should be ~0.087 cycles** (not 0.0!)

2. **If residuals match Tempo2**: Success! The fix worked. Key indicators:
   - JUG RMS < 10 μs (similar to Tempo2's 0.8 μs)
   - Correlation > 0.999
   - RMS difference < 1 μs
   - Message: "✓✓✓ SUCCESS! JUG residuals match Tempo2!"

3. **If residuals still don't match**: Check the diagnostic output:
   - Is `phase_offset_cycles` still zero? (TZR calculation failed)
   - Is `phase_offset_cycles` set but residuals still wrong? (May need to check time coordinates)
   - Large mean offset? (Possible parameter mismatch)

4. **Integration into main code**: Once verified, the TZR calculation can be integrated into the main JUG pipeline. The key insight is that TZR must be computed using the infinite-frequency barycentric time after applying all delays (binary, DM, etc.).

## Questions?

Check the inline comments in the notebook cells - they explain what each section does.
