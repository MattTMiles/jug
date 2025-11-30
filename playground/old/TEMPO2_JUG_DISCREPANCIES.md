# Tempo2 vs JUG: Identified Discrepancies

## Summary from Single TOA Trace

Traced TOA #1: t_topo = 58526.213889... MJD

### Tempo2 Result
- **Residual**: -2.016 μs

### JUG Results
- **Without TZR**: -1121.877 μs
- **With TZR** (simplified): -461.048 μs

### Discrepancy: ~460-1120 μs

---

## Key Findings

### 1. ✓ Binary Delay Calculation - SEEMS CORRECT

JUG computed: **1.882 seconds**

Formula used:
```python
delay_roemer = A1 * (sin(phi) + 0.5*(EPS1*sin(2*phi) - EPS2*cos(2*phi)))
delay_shapiro_binary = -2*r*log(1 - s*sin(phi))
```

This appears correct for ELL1 model. The magnitude (~1.9s) is reasonable for J1909-3744's binary orbit.

**Status**: Likely correct, but needs verification against Tempo2's exact formula.

---

### 2. ✓ DM Delay Calculation - SEEMS CORRECT

JUG computed: **0.0407 seconds** for 1029 MHz

Formula:
```python
K_DM = 4148.808 seconds
DM_eff = DM + DM1*(t-DMEPOCH) + DM2*(t-DMEPOCH)²
delay = K_DM * DM_eff / freq²
```

This matches standard DM delay formula.

**Status**: Likely correct.

---

### 3. ⚠️  ISSUE: TZR Phase Reference - DEFINITELY WRONG

Current JUG calculation uses **topocentric TZRMJD** without applying delays.

**What Tempo2 does:**
1. Start with TZRMJD (topocentric)
2. Apply ALL delays (clock, bary, binary, DM) to get **TZR emission time**
3. Compute phase at TZR emission time
4. Extract fractional part
5. Use this as phase reference

**What JUG is doing (wrong):**
- Using TZRMJD directly
- Not applying delays
- Getting wrong phase reference (0.388 cycles instead of ~0.087 cycles)

**Impact**: This creates a systematic offset of:
```
0.388 - 0.087 = 0.301 cycles
0.301 cycles / 339.3 Hz = 887 μs offset
```

This explains much of the ~460-1120 μs discrepancy!

---

### 4. ✗ CRITICAL ISSUE: What time to use for phase calculation?

**Question**: Should JUG compute phase at:
- A. `t_inf` (infinite-frequency barycentric time)?
- B. `t_emission` (emission time, before DM correction)?
- C. Something else?

From the trace:
- JUG used `t_inf` = 58526.21056990... MJD
- This gave phase = -14417729187.38... cycles

**Tempo2 convention**: Uses **infinite-frequency time** (after DM correction)

**Status**: JUG appears to be using the right time, but need to verify.

---

### 5. ⚠️  ISSUE: Phase wrapping method

JUG uses:
```python
frac_phase = mod(phase + 0.5, 1.0) - 0.5
```

This should give fractional phase in range `(-0.5, 0.5]` cycles.

**Tempo2**: Uses same method (confirmed from source code)

**Status**: Correct.

---

### 6. ✗ MAJOR ISSUE: Starting point

**Critical observation:**
- JUG is starting from Tempo2's **BAT** (Barycentric Arrival Time)
- BAT already has geometric delays applied!
- JUG should NOT recompute barycentric corrections

**Correct flow when using Tempo2 BAT:**
```
BAT (from Tempo2)
  → subtract binary_delay → emission time
  → subtract DM_delay     → infinite-frequency time
  → compute spin phase
  → apply TZR reference
  → convert to time residual
```

**Status**: JUG is doing this correctly in principle, but TZR is wrong.

---

## Specific Code Issues to Fix

### Issue #1: TZR Calculation in `residual_maker_playground.ipynb` Cell 200

**Current code** (around line 136):
```python
tzr_emission_mjd = float(tzr_em_time.mjd - tzr_dm_sec / SECS_PER_DAY + tzr_fd_sec / SECS_PER_DAY)
model.phase_ref_mjd = tzr_emission_mjd
phase_at_tzr = float(spin_phase(jnp.array([tzr_emission_mjd]), model)[0])
frac_phase_at_tzr = float(jnp.mod(phase_at_tzr + 0.5, 1.0) - 0.5)
model.phase_offset_cycles = frac_phase_at_tzr
```

**Problem**: The sign is wrong!

```python
# WRONG:
tzr_emission_mjd = tzr_em_time.mjd - tzr_dm_sec / SECS_PER_DAY + tzr_fd_sec / SECS_PER_DAY

# CORRECT should be:
tzr_emission_mjd = tzr_em_time.mjd - (tzr_dm_sec - tzr_fd_sec) / SECS_PER_DAY
```

Or more clearly:
```python
# Emission time (after binary delay)
t_em = t_bary - binary_delay/SECS_PER_DAY

# Infinite-frequency time (after DM delay)
t_inf = t_em - dm_delay/SECS_PER_DAY

# FD correction (profile evolution)
t_inf_corrected = t_inf + fd_delay/SECS_PER_DAY  # FD delays are added
```

Wait, I need to check FD delay sign...

---

### Issue #2: Residual function may be using wrong reference

In `residuals_seconds_at_topocentric_time()`:

**Current:**
```python
phase_diff = phase - phase_ref - model.phase_offset_cycles
```

**Analysis needed:**
- If `phase_ref` is the absolute phase at TZR, and
- `phase_offset_cycles` is the fractional part of that same phase
- Then yes, we're subtracting it twice!

**Fix:** Remove the `phase_offset_cycles` term:
```python
phase_diff = phase - phase_ref
```

But wait - this is what I already suggested and tested, and it didn't fix the issue. So maybe the problem is entirely in the TZR calculation.

---

### Issue #3: Frequency not loaded

The trace script assumed freq = 1029 MHz, but this should be loaded from the .tim file or Tempo2 output for each TOA.

---

## Recommended Fixes (Priority Order)

### 1. HIGH PRIORITY: Fix TZR Calculation

**File**: `residual_maker_playground.ipynb`, cell 200

**Current code** around the TZR calculation needs to:
1. Load TZRMJD, TZRFRQ, TZRSITE from .par
2. Compute barycentric time at TZR
3. Compute binary delay at TZR
4. Compute emission time = bary - binary
5. Compute DM delay at TZR
6. Compute t_inf = emission - DM_delay
7. Compute phase at t_inf
8. Extract fractional part
9. Set `phase_offset_cycles` to this value

**Expected result**: phase_offset_cycles ≈ 0.087 cycles (from .par file shows this in output)

---

### 2. MEDIUM PRIORITY: Verify Binary Delay Formula

Compare JUG's ELL1 implementation with Tempo2's source code:
- Check if PBDOT, XDOT corrections are applied
- Verify Shapiro delay formula (r, s parameters)
- Confirm Einstein delay (GAMMA parameter)

---

### 3. LOW PRIORITY: Load Frequency from Data

Parse the .tim file or Tempo2 output to get the actual frequency for each TOA, don't assume 1029 MHz.

---

## Testing Strategy

1. **Fix TZR calculation first**
   - Implement proper delay calculations at TZR epoch
   - Verify phase_offset_cycles ≈ 0.087

2. **Test on single TOA**
   - Should get residual within ~10 μs of Tempo2

3. **Test on all TOAs**
   - Should get RMS ≈ 0.8 μs
   - Correlation > 0.999

---

## Expected Outcome After Fixes

If TZR is calculated correctly:
- The 887 μs systematic offset should disappear
- Residuals should be within a few μs of Tempo2
- RMS should be ~0.8 μs (matching Tempo2)

The remaining small differences would be due to:
- Numerical precision
- Minor formula differences
- Parameter rounding
