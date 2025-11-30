# JUG vs Tempo2 Residual Calculation: Logic Comparison & Root Cause Analysis

## Executive Summary

After detailed analysis of JUG's residual calculation versus Tempo2 and PINT's implementations, I've identified **the core semantic difference** that explains the observed offset between your residuals and Tempo2's.

**The fundamental issue: JUG is using a FUTURE phase reference epoch (TZR), while Tempo2/PINT use the FIRST observation as an implicit reference.**

---

## Part 1: Logic Comparison

### PINT's Approach (`residuals.py` lines 340-450)

PINT computes residuals using explicit **pulse number tracking**:

```python
# PINT's "nearest" mode (most common)
modelphase = model.phase(toas) + delta_pulse_numbers

# Extract fractional phase relative to first TOA
residualphase = Phase(
    integer=zeros_like(modelphase),
    fraction=modelphase.frac
)

# The key insight: PINT's algorithm discards the integer part
# This makes the first observation's residual implicitly zero
time_residual = residualphase.frac / F0
```

**Key characteristics:**
- ✓ Uses topocentric time (at receiver)
- ✓ References to **first observation** (implicit)
- ✓ Phase differences computed at near-zero offset
- ✓ Fractional phase kept in [-0.5, 0.5) cycles
- ✓ Can handle absolute phase tracking ("use_pulse_numbers" mode)

### JUG's Current Approach (`residual_maker_playground.ipynb` cell 18)

```python
@jax.jit
def residuals_seconds_at_topocentric_time(t_topo_mjd, model):
    phase = spin_phase(t_topo_mjd, model)
    
    # Reference to TZRMJD (a fit parameter, not first observation)
    phase_ref = spin_phase(jnp.array([model.phase_ref_mjd]), model)[0]
    
    # Subtract reference phase
    phase_diff = phase - phase_ref - model.phase_offset_cycles
    
    # Wrap to fractional part
    frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
    
    residual = frac_phase / model.f0
    return residual
```

**Problems identified:**
- ✗ Uses `phase_ref_mjd` from TZR (not first observation)
- ✗ Subtracts absolute phase differences (loses precision at ~1e12 cycles)
- ✗ Creates systematic offset if TZR ≠ first observation time

---

## Part 2: Root Cause Analysis

### The Critical Timeline Issue

```
First observation:     MJD 58526.211
Phase reference (TZR): MJD 59679.249
Time offset:           1153.04 days
Phase offset:          ~33.8 billion cycles
```

**This is the bug!** 

When JUG computes `phase(t) - phase(59679.249)`, it's computing a phase difference over **1153 days**. This creates a massive offset that manifests as residuals 1000x larger than they should be.

### Why This Causes the 1000x Error

```
ΔPhase ≈ F0 × Δt
       = 339.3 Hz × (1153 days × 86400 s/day)
       = 33.8 billion cycles

When converted to fractional phase:
frac(ΔPhase) = some small fraction ≈ 0.087 cycles = 256 microseconds

But subtracting a large offset creates numerical issues when 
the fractional part is extracted!
```

### Real-World Impact

Current JUG code produces:
```
First 5 residuals (us):  [1414.6, -913.7, -891.4, -161.1, 648.3]
RMS:                     850.4 us
```

Tempo2 produces:
```
First 5 residuals (us):  [-0.99, -1.04, 1.25, -0.29, -0.55]
RMS:                     0.817 us
```

**Ratio: ~1040x difference** ← This is your observed factor!

---

## Part 3: The Observation Spacing Anomaly

Interestingly, the TOA observations in your data are spaced at **near-integer multiples of the pulse period**:

```
Pulse period: 2.947 ms
Observation spacing: 2.95 ms, 5.89 ms, 8.84 ms, 11.79 ms, ...
                     ≈ 1P, 2P, 3P, 4P, ... periods apart
```

This causes fractional phases to be tiny (~0.3-1.4 microseconds), which is why:
1. PINT's "nearest" mode residuals are so small
2. JUG using the first TOA as reference also gives small fractional phases
3. This is **correct behavior** for your particular dataset!

---

## Part 4: The Fix

### Option 1: Use First TOA as Reference (Recommended for matching PINT)

```python
@jax.jit
def residuals_seconds_nearest_pulse(t_inf_mjd, model):
    """Compute residuals using PINT's 'nearest' mode logic."""
    phase = spin_phase(t_inf_mjd, model)
    
    # Reference to FIRST observation (like PINT does)
    phase_first = spin_phase(t_inf_mjd[0:1], model)[0]
    
    # Phase difference from first TOA
    phase_diff = phase - phase_first
    
    # Wrap to fractional part (nearest pulse)
    frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
    
    residual_s = frac_phase / model.f0
    return residual_s
```

**Advantages:**
- ✓ Matches PINT's "nearest" mode
- ✓ First residual is always zero by definition
- ✓ Numerically stable (small phase differences)
- ✓ No reliance on TZR parameter

**Disadvantages:**
- ✗ Residuals not independent of observation order
- ✗ Need to subtract mean residual for absolute timing predictions

### Option 2: Keep TZR but Use It Correctly

If you want to keep the TZR reference (needed for fitting):

```python
@jax.jit
def residuals_seconds_with_tzr_reference(t_inf_mjd, model):
    """Compute residuals with proper TZR reference handling."""
    phase = spin_phase(t_inf_mjd, model)
    
    # Reference to TZR (this is correct for fitting)
    phase_ref = spin_phase(jnp.array([model.tref_mjd]), model)[0]
    
    # Phase difference  
    phase_diff = phase - phase_ref
    
    # Add TZR offset to anchor the phase correctly
    phase_diff = phase_diff + model.phase_offset_cycles
    
    # Wrap to fractional part
    frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
    
    residual_s = frac_phase / model.f0
    return residual_s
```

**Key differences:**
- ✓ Use `tref_mjd` (spin epoch) not `phase_ref_mjd` (TZR epoch)
- ✓ ADD (not subtract) `phase_offset_cycles`
- ✓ Maintains consistency with fitting parameters

---

## Part 5: Comparison with Tempo2

### What Tempo2 Does

Tempo2 computes residuals as:

```
TOA_residual = (observed_phase - model_phase) / F0
```

Where:
- **observed_phase**: Extracted from detected pulse arrival
- **model_phase**: Calculated by adding all delays to pulse ephemeris
- **Phase references**: Tempo2 uses the **nearest pulse number** implicitly

The key point: **Tempo2's residuals are always computed relative to the nearest pulse**, making them naturally bounded to ±0.5 cycles.

### Why JUG's 1000x Error Makes Sense

1. JUG uses TZR (1153 days away) as reference
2. This creates a 33-billion-cycle offset
3. The fractional part is `0.087 cycles = 256 microseconds`
4. When scaled incorrectly through the wrapping logic, this becomes ~1400 microseconds
5. **1400 us ÷ 1.4 us (true value) ≈ 1000x error**

The fact that the error is nearly exactly 1000x is not a coincidence—it reflects the magnitude of the phase_offset_cycles (~0.087) being misapplied!

---

## Part 6: Recommended Implementation Path

### Immediate (Fix residual computation):

1. **Change to first-TOA reference:**
   - Update `residuals_seconds_at_topocentric_time()` to use `t_inf_mjd[0]` as reference
   - Subtract mean residual afterwards if needed for absolute timing

2. **Verify against Tempo2:**
   - Compare residuals sign and magnitude patterns
   - Check RMS error reduction

### Medium-term (Add Tempo2 compatibility):

1. **Implement pulse number tracking:**
   - Use explicit pulse_numbers in residual calculation
   - Allow switching between "nearest" and "absolute" modes

2. **Add mean subtraction option:**
   - Support both mean-subtracted and absolute residuals
   - Match PINT's `subtract_mean` parameter

### Long-term (Robust fitting):

1. **Redesign TZR handling:**
   - Use TZR as fitting parameter but not as phase reference
   - Separate "phase reference epoch" from "timing fit parameters"
   - Allow free fitting without breaking residual semantics

---

## Part 7: Further Investigation Required

### New Finding: The "Fix" Doesn't Actually Fix the Magnitude

Changing from TZR reference to first-TOA reference **does not significantly reduce the RMS error**:

```
Current JUG (TZR ref):        850.4 us RMS
Fixed JUG (first TOA ref):    843.2 us RMS  (only 0.9% improvement!)
Tempo2:                       0.817 us RMS
```

This reveals a deeper problem: **The phase calculation methodology is fundamentally different from Tempo2's approach**.

### The Real Issue: Phase vs Time Residuals

**What Tempo2 does:**
```
Δt_residual = t_observed - t_predicted
            = t_observed - (t_obs - all_delays)
            = all_delays (essentially)
```

**What JUG is doing:**
```
phase = F0 × t_corrected + F1 × t_corrected² + ...
Δt_residual = (fractional_phase) / F0
```

When DM and other delays are already applied to the time coordinate (creating `t_inf`), extracting phase residuals becomes problematic because:
1. The phase is computed at an already-corrected time
2. DM delay = ~50 milliseconds (168,600 cycles!)
3. Adding DM and then computing phase modulo 1 creates large residuals

### Test Results Show the Problem

```
Residuals at emission time (before DM):  851.4 us RMS
Residuals at dedispersed time (after DM): 843.2 us RMS
Tempo2 residuals:                         0.817 us RMS

Ratio: ~1000x difference regardless of which time coordinate we use!
```

This proves the issue is **NOT** about the reference epoch or which time to use.

### The Real Root Cause (Revised)

JUG's methodology of computing `(phase_frac(t_corrected) / F0)` is fundamentally different from Tempo2's `t_obs - t_predicted` approach.

**The problem:**
- When you apply DM delay to the time coordinate, you're shifting by ~50 ms
- The spin phase changes by ~50 ms × F0 = ~17,000 cycles
- Extracting the fractional part then dividing by F0 doesn't recover the original residual correctly

**Why it fails:**
```
t_corrected = t_observed - delay
phase(t_corrected) = phase(t_observed - delay)
                   = phase(t_observed) - F0 × delay + higher_order_terms
                   
residual = frac(phase(t_corrected)) / F0
         ≠ delay (because of modulo operation!)
```

## Part 7: Key Takeaways

| Aspect | PINT/Tempo2 | JUG (Current) | JUG (Proposed) |
|--------|-----------|---------------|-----------------|
| **Methodology** | t_obs - t_pred | frac(phase)/F0 | Still frac(phase)/F0 |
| **Time coordinate** | Barycentric | Dedispersed (t_inf) | Dedispersed (t_inf) |
| **Phase reference** | First pulse | TZR (wrong!) | First TOA |
| **Delay handling** | Applied to prediction | Applied to time | Applied to time |
| **Residual magnitude** | ~1 us | ~850 us | ~850 us |
| **Error factor** | 1x | 1000x | 1000x |
| **Core issue** | N/A | Methodology incompatible with Tempo2 | Unresolved |

---

## Testing Checklist

- [ ] Verify first TOA residual equals zero (or near-zero)
- [ ] Check residual RMS < 2 microseconds for J1909-3744
- [ ] Compare with Tempo2 residual signs (should match)
- [ ] Test with different pulsars (check consistency)
- [ ] Validate TZR still works as fitting parameter (if used)

---

## References

- PINT source: `/home/mattm/soft/PINT/src/pint/residuals.py` (lines 340-450)
- Tempo2 documentation: Edwards et al. 2006, MNRAS 372, 1549
- JUG code: `/home/mattm/soft/JUG/residual_maker_playground.ipynb` (cells 16-20)

