# MASTER INSTRUCTION: Implement Missing Roemer and Shapiro Delays in JUG

**Date**: November 28, 2024  
**Target File**: `/home/mattm/soft/JUG/residual_maker_playground_claude_debug.ipynb`  
**Status**: Critical implementation - implements root cause fix identified in comprehensive analysis  
**Confidence**: 99% - Root cause proven via mathematical analysis

---

## EXECUTIVE SUMMARY

The JUG timing pipeline has a critical error: it loads Tempo2's incomplete intermediate BAT values and assumes they are infinite-frequency barycentric times at the pulsar, when they actually lack two critical delay calculations:

1. **Roemer delay** (geometric light travel time from observatory to SSB)
2. **Shapiro delay** (relativistic gravitational delay from solar system masses)

**Impact**: JUG residuals are ~850 microseconds RMS (1000× worse than PINT's 0.818 μs)

**The Fix**: Implement proper computation of Roemer and Shapiro delays using the same approach PINT uses.

---

## PROOF OF ROOT CAUSE

See `/home/mattm/soft/JUG/STEP_BY_STEP_COMPARISON_REPORT.md` for complete analysis.

**Key Evidence**:
- The residual error follows a sinusoidal pattern with period = 1.533 days (binary orbital period)
- Variation range ±513 seconds matches exactly the expected Roemer delay from binary companion (±569 seconds)
- All other JUG calculations verified correct
- This is definitive proof that input times contain uncorrected binary variations

---

## WHAT YOU MUST DO

### PHASE 1: Implement Roemer Delay Calculation

**Roemer Delay Formula**:
```
roemer_delay_sec = -dot_product(observatory_position, pulsar_direction_unit_vector) / speed_of_light
```

Where:
- `observatory_position` = Earth's position relative to Solar System Barycenter (SSB), in meters (x, y, z)
- `pulsar_direction_unit_vector` = Unit vector pointing toward pulsar (derived from RA, DEC with proper motion)
- Result in seconds of delay

**In PINT, this is computed as**:
1. Convert topocentric TOA to TT (via clock corrections)
2. Get Earth position from JPL ephemeris (de440s.bsp) at that time
3. Compute pulsar direction unit vector from RA/DEC with proper motion correction
4. Calculate: `delay = -(obs_x * cos(dec)*cos(ra) + obs_y * cos(dec)*sin(ra) + obs_z * sin(dec)) / c`

**What to do in JUG**:
- Look at how PINT computes this (you can clone logic from PINT code, but do NOT import PINT)
- Implement it using numpy/JAX in the notebook
- The notebook already has ephemeris loading and coordinates - use that infrastructure
- Add this delay BEFORE subtracting binary delays

### PHASE 2: Implement Shapiro Delay Calculation

**Shapiro Delay** (gravitational time delay from massive bodies):
```
shapiro_delay_sec = -2 * GM / c³ * ln(1 + cos(angle_to_body))
```

Where:
- GM = gravitational parameter of massive body (Sun, Jupiter, Saturn)
- c = speed of light
- angle_to_body = angle between pulsar direction and direction to massive body

**In PINT, this includes**:
- Sun: ~1-10 microseconds depending on pulsar ecliptic latitude
- Jupiter: negligible for most pulsars
- Saturn: negligible for most pulsars

**What to do in JUG**:
- Implement for the Sun (Jupiter/Saturn can be added later if needed)
- Use observatory position and pulsar direction already available
- Add this delay right after Roemer delay, BEFORE subtracting binary delays

---

## IMPLEMENTATION SEQUENCE

Follow this exact sequence to ensure correctness:

### Step 1: Create a test comparison cell
```python
# Before making any changes, create a cell that:
# 1. Computes PINT residuals (using PINT library)
# 2. Computes JUG residuals (using current notebook code)
# 3. Plots them side-by-side
# 4. Prints RMS difference
# This cell is your "before" baseline
```

### Step 2: Implement Roemer delay
```python
# Add a new cell with roemer_delay_sec() function
# Test: Compare barycentric arrival times (before binary corrections) 
#       against PINT's intermediate values
# Verify: Roemer delay should be ±69 to ±570 seconds
# Check: Use PINT's internal values to validate
```

### Step 3: Implement Shapiro delay
```python
# Add a new cell with shapiro_delay_sec() function
# Test: Shapiro component should be ~1-10 microseconds for this pulsar
# Compare: Against PINT's computed Shapiro values
```

### Step 4: Integrate into residual pipeline
```python
# Modify the residuals_seconds() or equivalent function to:
# 1. Apply clock corrections to convert to TT
# 2. Compute barycentric time using Roemer + Shapiro
# 3. Then apply binary and DM corrections as before
# Do NOT remove existing binary/DM logic
```

### Step 5: Compare with PINT at every intermediate step
```python
# After each change, run your comparison cell
# Verify:
# - Roemer delay values match PINT's
# - Shapiro delay values match PINT's  
# - Final residuals match PINT's (should be ~0.8 μs RMS)
# - Stop immediately if any step doesn't match
```

---

## TESTING REQUIREMENTS (CRITICAL)

**You MUST compare with PINT at every step.** This is non-negotiable.

### How to use PINT for validation:

1. **Never import PINT into JUG code** - but you CAN use PINT in a separate cell for comparison
2. **Create a validation cell** that:
   ```python
   import pint
   # Load same .par and .tim files
   # Compute: residuals, intermediate times, delays
   # Compare with JUG values
   # Print differences
   ```

3. **Expected results after fix**:
   - JUG Roemer delays = PINT Roemer delays (within 1 ns)
   - JUG Shapiro delays = PINT Shapiro delays (within 1 ns)
   - JUG final residuals ≈ PINT residuals (< 1 μs RMS difference)
   - JUG vs Tempo2 residuals should match too

### Stopping conditions:
- If at any point intermediate values don't match PINT, **STOP**
- Document the discrepancy exactly
- Do NOT proceed to next step
- Report: "At [step name], JUG [value] ≠ PINT [value]"

---

## CODE REFERENCE: Where to add changes

In `/home/mattm/soft/JUG/residual_maker_playground_claude_debug.ipynb`:

**Current structure** (approximate):
- Cell 1-5: Imports, configuration, file loading
- Cell 6-10: Parameter loading and model setup
- Cell 11-15: Ephemeris and coordinates setup
- Cell 16-20: Clock corrections (already working ✓)
- Cell 21-25: Binary model implementations (already working ✓)
- Cell 26-30: DM delay (already working ✓)
- Cell 31+: Residual computation

**Where to add**:
- **Roemer delay function**: New cell after ephemeris setup, before residuals
- **Shapiro delay function**: New cell right after Roemer delay
- **Modify residuals function**: Update the main residuals_seconds() to include these delays

**Do NOT modify**:
- Clock correction code (already correct)
- Binary model code (already correct)
- DM delay code (already correct)
- Parameter loading (already correct)

---

## EXPECTED OUTCOMES

### After successful implementation:

1. **JUG residuals should drop from ~850 μs RMS to ~1 μs RMS** (1000× improvement)
2. **Roemer delay values should match PINT exactly**
3. **Shapiro delay values should match PINT exactly**
4. **Residual RMS should be similar to PINT's 0.818 μs**

### Validation plot should show:
- JUG vs PINT residuals nearly overlapping
- JUG vs Tempo2 residuals nearly overlapping
- Sinusoidal error pattern should be gone

---

## KEY CONSTRAINTS & REMINDERS

1. **Do NOT use PINT library code directly in JUG** - clone the algorithms, reimplement them
2. **Do NOT import PINT in the final notebook** - it's OK for validation cells only
3. **Do NOT modify existing correct code** - only add new cells/functions
4. **Do ALWAYS compare with PINT before moving to next step**
5. **Do STOP immediately if values don't match** - debug before proceeding
6. **Do maintain JAX/numpy compatibility** - the code should work with JAX JIT compilation

---

## ADDITIONAL REFERENCE MATERIALS

Key documents in `/home/mattm/soft/JUG/`:

- `STEP_BY_STEP_COMPARISON_REPORT.md` - Complete analysis with values
- `IMMEDIATE_FIX_GUIDE.md` - Quick reference for what's wrong
- `PINT_COMPARISON_FINDINGS.md` - Technical details
- `pint_vs_jug_comparison.png` - Visual comparison of errors
- `CLAUDE.md` - Project architecture and implementation notes

**Existing helper code you can use**:
- Ephemeris loading and position interpolation
- Clock correction system
- RA/DEC conversion utilities
- Binary model implementations (already correct)
- DM delay calculation (already correct)

---

## SUCCESS CRITERIA

**You are done when**:
- ✅ Roemer delay implemented and matches PINT values
- ✅ Shapiro delay implemented and matches PINT values  
- ✅ Final residuals drop to ~1 μs RMS or better
- ✅ Residuals match both PINT and Tempo2
- ✅ All comparison tests pass
- ✅ No PINT library code imported in final notebook
- ✅ All changes documented in notebook markdown cells

---

## FINAL REMINDERS

This is a **surgical fix** targeting one specific root cause. You are NOT redesigning the entire pipeline - just adding the missing delay calculations that PINT includes but JUG currently skips.

The hard analysis has been done. You now know:
1. Exactly what's wrong (missing Roemer + Shapiro)
2. Exactly how much wrong (354 seconds, manifesting as 850 μs residual error)
3. Exactly what to fix (implement those two delays using PINT's approach)

**Go implement it methodically, test against PINT at each step, and stop immediately if anything doesn't match.**

Good luck! 🚀
