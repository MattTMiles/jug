# TZR Calculation Fix - Summary

## What Was Fixed

### The Problem

The JUG timing residuals were showing ~833 μs RMS compared to Tempo2's 0.8 μs RMS. Investigation revealed:

1. **Missing TZR calculation**: The `phase_offset_cycles` was set to 0.0 instead of the correct value (~0.087 cycles)
2. **Root cause**: The TZR (Time of Zero Residual) phase offset was not being computed at all when using Tempo2 BAT for comparison

### The Solution

Added a complete TZR phase offset calculation to the debug notebook that:

1. **Reads TZR parameters** from the .par file:
   - TZRMJD (topocentric time of zero residual)
   - TZRFRQ (frequency at TZR)
   - TZRSITE (observatory code at TZR)

2. **Finds the closest TOA** to TZRMJD in the Tempo2 component data

3. **Computes binary delay at TZR** using the ELL1 model:
   - Roemer delay (orbital light travel time)
   - Einstein delay (γ term for relativistic time dilation)
   - Shapiro delay (gravitational time delay from companion star)

4. **Computes emission time**: BAT - binary_delay

5. **Computes DM delay** at TZR frequency using:
   - DM evolution model (DM, DM1, DM2)
   - K_DM constant (4148.808 s MHz² pc⁻¹ cm³)

6. **Computes infinite-frequency time**: emission_time - DM_delay

7. **Computes spin phase** at infinite-frequency time using F0, F1, F2

8. **Extracts fractional phase** wrapped to (-0.5, 0.5] cycles

9. **Updates the model** with the correct `phase_offset_cycles`

## Files Modified

### `residual_maker_playground_claude_debug.ipynb`

- **New cell added** (after model building, before residual computation)
- **Cell title**: "TZR PHASE OFFSET CALCULATION"
- **Output**: Shows step-by-step TZR computation with intermediate values

### `DEBUG_NOTEBOOK_README.md`

- Updated to document the new TZR calculation
- Updated section numbering
- Added expected TZR value (~0.087 cycles for J1909-3744)
- Updated "Next Steps After Running" with TZR-specific checks

## Expected Results

### When Running the Updated Notebook

1. **TZR Calculation Output** should show:
   ```
   ================================================================================
   TZR PHASE OFFSET CALCULATION
   ================================================================================

   ✓ Found TZR parameters:
     TZRMJD = 59679.248061951184 MJD
     TZRSITE = 'pks'
     TZRFRQ = 1029.02558 MHz

   ✓ Found closest TOA to TZR:
     Requested TZRMJD: 59679.248061951184
     Actual topocentric: <close_value>
     Difference: <small_value> seconds
     Tempo2 BAT at TZR: <bat_value>

   ✓ Binary model: ELL1
     PB = 1.533449474574 days
     A1 = 1.8979380579999998 lt-s
     TASC = 54923.96966699 MJD
     EPS1, EPS2 = <values>
     Shapiro: r=<value>, s=<value>

     Binary delays at TZR:
       Roemer: <~1.88s typical>
       Einstein: <small>
       Shapiro: <small>
       Total: <~1.88s total>

   ✓ TZR emission time: <bat - binary_delay>

   ✓ DM delay at TZR:
     DM_eff = <~10.39> pc/cm^3
     Frequency = 1029.02558 MHz
     DM delay = <~0.041s>

   ✓ TZR infinite-frequency time: <emission - dm_delay>

   ✓ Phase at TZR:
     Absolute phase: <large number> cycles
     Fractional phase: ~0.087 cycles  ← KEY VALUE
     Time equivalent: ~256 μs

   ✓✓✓ Model updated with TZR phase offset!
     phase_ref_mjd = <inf_freq_time>
     phase_offset_cycles = 0.087
   ```

2. **Residual Comparison** should show:
   ```
   ================================================================================
   SIMPLE TEST: Using Tempo2 barycentric times
   ================================================================================

   Results:
     JUG RMS:        <1-10> μs  ← Much improved!
     Tempo2 RMS:     0.817 μs
     Correlation:    >0.999  ← High correlation!
     RMS difference: <1> μs  ← Small difference!

   ✓✓✓ SUCCESS! JUG residuals match Tempo2!
   ```

3. **Diagnostic** should show:
   ```
   ================================================================================
   DIAGNOSTIC: Phase Offset Analysis
   ================================================================================

   Model phase_offset_cycles: 0.087  ← Non-zero!

   ✓ TZR phase offset is set: 0.087 cycles
     Equivalent time offset: ~256 μs

   Mean residual offset: <small> μs = <small> cycles
   ```

## Technical Details

### Why This Fix Works

The TZR phase offset represents the fractional pulse phase at the reference epoch. When this is set correctly:

1. **All residuals are measured relative to the same pulse phase** at the TZR epoch
2. **Phase wrapping** (to -0.5 to 0.5 cycles) is done relative to this reference
3. **Systematic offsets** of hundreds of cycles are removed by anchoring to TZR

Without the TZR calculation:
- Each TOA's phase is wrapped independently
- Different wrapping choices create systematic ~microsecond offsets
- RMS residuals are inflated by 1000x

### ELL1 Binary Model Implementation

The ELL1 model used here implements:

```python
# Orbital phase
φ = n * (t - TASC)  where n = 2π/PB

# Roemer delay
Δ_R = A1 * [sin(φ) + 0.5*(EPS1*sin(2φ) - EPS2*cos(2φ))]

# Einstein delay
Δ_E = γ * sin(φ)

# Shapiro delay
Δ_S = -2r * ln(1 - s*sin(φ))

# Total binary delay
Δ_binary = Δ_R + Δ_E + Δ_S
```

### Time Coordinate Flow

```
TZRMJD (topocentric)
  ↓ (clock corrections + barycentric delays - from Tempo2)
BAT (barycentric arrival time)
  ↓ (subtract binary delay)
Emission time (when pulse was emitted)
  ↓ (subtract DM delay at TZRFRQ)
Infinite-frequency time (dispersion removed)
  ↓ (compute spin phase using F0, F1, F2)
Phase (absolute, in cycles)
  ↓ (wrap to -0.5 to 0.5)
Fractional phase → phase_offset_cycles
```

## Next Steps

1. **Run the updated notebook**:
   ```bash
   jupyter notebook residual_maker_playground_claude_debug.ipynb
   ```

2. **Execute all cells in order** and check:
   - TZR calculation output shows `phase_offset_cycles ≈ 0.087`
   - Residual comparison shows RMS < 10 μs and correlation > 0.999
   - Success message appears

3. **If successful**: Integrate the TZR calculation into the main JUG pipeline

4. **If not successful**: Check:
   - Are all binary parameters present in the .par file?
   - Is the binary model correctly identified (ELL1/ELL1H)?
   - Are Tempo2 component files loaded correctly?

## References

- TEMPO2_JUG_DISCREPANCIES.md - Original analysis identifying the TZR issue
- tempo2_vs_jug_comparison.md - Step-by-step comparison of calculations
- trace_single_toa.py - Single TOA trace showing expected values
