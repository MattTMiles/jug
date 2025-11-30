# JUG Residual Offset Diagnosis

## Current Status

**Problem**: JUG computes residuals with RMS of ~840 microseconds, while Tempo2 (treated as truth) has RMS of ~0.8 microseconds. The JUG residuals are **~1000x too large**.

**Symptoms**:
- JUG residuals: ~700 microseconds (all large positive values, 698-700 range)
- Tempo2 residuals: ±1 microsecond (zero-centered fluctuations)
- No obvious constant offset—the discrepancy is systematic

## Root Causes Identified & Fixed

### 1. ✅ **FIXED**: Phase offset wrapping (TZR anchor) 
- **What was wrong**: `phase_offset_cycles` was being set to an absolute phase value (~1.7 trillion cycles) instead of a fractional phase (-0.5 to 0.5 cycles)
- **Impact**: Created a theoretical offset of ~10^12 microseconds
- **Fix**: Changed line 961 to only store the fractional phase:
  ```python
  frac_phase_at_tzr = float(jnp.mod(phase_at_tzr + 0.5, 1.0) - 0.5)
  model.phase_offset_cycles = frac_phase_at_tzr
  ```
- **Result**: Phase offset now correctly bounded to (-0.5, 0.5), BUT residuals still wrong

### 2. ❓ **STILL INVESTIGATING**: Fundamental residual calculation

The residual calculation appears correct theoretically:
```python
phase_diff = phase - phase_ref + model.phase_offset_cycles
frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
spin_res = frac_phase / model.f0  # converts cycles to seconds
```

But the output magnitude is wrong. **Possible issues**:

#### a) Timing reference frame mismatch
- JUG uses `t_inf_jax` (dedispersed, FD-corrected emission time)
- But what should the residual be computed at?
- Is it at EMISSION time or ARRIVAL time at SSB?

#### b) Missing or double-counted delays
- Model parameters are in different scales (e.g., TDB vs TCB)
- Some delay components might not be properly accounted for
- Binary/Shapiro delays might be applied incorrectly

#### c) Incorrect time coordinate transformation
- Tempo2/PINT might compute residuals in a different time scale or reference frame
- JUG might need to account for additional corrections

## Latest Finding (Critical!)

Using **topocentric time** (t_mjd) instead of **emission time** (t_em_mjd) for phase calculation shows:
- ✅ Residuals now have correct **sign variation** like Tempo2
- ✅ RMS still ~850 us (vs Tempo2's 0.8 us)
- ❌ But residuals are NOT zero-centered—they're offset by ~1.4 milliseconds

This suggests:
1. The phase calculation is **conceptually correct**
2. But there's a **constant systematic delay** that's not being accounted for
3. Possibly the phase reference or TZR anchoring is wrong

## Next Steps to Investigate

### 0. **CRITICAL: Review how residuals should be computed**
Check PINT/Tempo2 docs on:
- Are residuals = (observed_time - predicted_time)?
- Or residuals = observed_phase (wrapped)?
- How does TZR/TZRMJD anchor factor in?

### 1. **Compare component-by-component with Tempo2**
Extract Tempo2's internal components from `temp_pre_components_next.out` and `temp_pre_general2.out`:
- Barycentric delay
- Shapiro delay  
- Binary delay
- DM delay
- Einstein delay
- TDB-TT correction

Compare each against JUG's calculations.

### 2. **Verify the observation semantics**
- Are `t_mjd` values from the `.tim` file:
  - Arrival times at the observatory (topocentric)?
  - Arrival times at the Solar System Barycenter (already corrected)?
  - The actual observed pulse arrival, or something else?

### 3. **Test with a simpler model**
Try computing residuals with:
- No binary model
- No DM delay
- Just F0, F1 spin
- At various reference epochs

See when the 1000x scale error appears.

### 4. **Check if it's a time coordinate issue**
The barycentric delay in `temp_pre_general2.out` column 3 might need to be SUBTRACTED from `t_mjd` differently. Perhaps:
```python
# Current approach:
t_em_mjd = t_mjd + bary_delay_sec / SECS_PER_DAY

# Maybe should be:
t_em_mjd = t_mjd - bary_delay_sec / SECS_PER_DAY
```

### 5. **Directly match Tempo2's residual definition**
Tempo2's documentation states residuals are calculated as:
$$\text{residual} = TOA_{observed} - \sum(\text{delays})$$

Verify that JUG is computing this exact quantity.

## Files to Check

- `/home/mattm/soft/JUG/temp_pre_general2.out` - Tempo2 components (format: MJD resid_sec error_us)
- `/home/mattm/soft/JUG/prefit.res` - Tempo2 prefit residuals (format: index resid_sec chi)
- `/home/mattm/soft/JUG/residual_maker_playground.ipynb` - JUG notebook
- `/home/mattm/projects/MPTA/partim/production/fifth_pass/J1909-3744.*` - Source data

## Key Model Parameters (J1909-3744)

```
F0 = 339.31568139672726 Hz
F1 = -1.8977e-15 Hz/s
DM = 10.39 pc/cm³
Binary: ELL1 model (TASC, PB, A1, EPS1, EPS2, etc.)
```

## Questions for User

1. When you run Tempo2 to generate `temp_pre_general2.out`, what flags/options do you use?
2. Is the `t_mjd` from the `.tim` file topocentric UTC or already barycentric?
3. What does the par file say about `UNITS` (TDB vs TCB)?
4. Have you verified that the clock corrections are being applied correctly?
5. Does Tempo2 produce intermediate files that show step-by-step residual calculation?

---

**Last Updated**: 2025-11-27  
**Status**: Phase wrapping fixed, magnitude error still present
