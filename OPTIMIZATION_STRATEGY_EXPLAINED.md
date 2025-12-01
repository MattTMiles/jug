# Optimization Strategy - Detailed Explanation

**Date**: 2025-12-01
**Question**: What's the plan for speeding up fitting while keeping it working?

---

## Current Working Code (test_f0_f1_fitting_tempo2_validation.py)

```python
for iteration in range(max_iter):
    # 1. Create temp par file with updated F0/F1
    with tempfile.NamedTemporaryFile(...) as f:
        # Write F0 and F1 with new values
        
    # 2. Compute FULL residuals from scratch
    result = compute_residuals_simple(temp_par, tim_file)
    
    # 3. Compute derivatives
    derivs = compute_spin_derivatives(...)
    
    # 4. WLS solve
    delta_params = wls_solve_svd(...)
    
    # 5. Update parameters
    f0 += delta_params[0]
    f1 += delta_params[1]
```

**Time**: 21 seconds (25 iterations × 0.85s each)

**Why is it slow?**
Every iteration does:
- Create temp file (I/O)
- Parse par file
- Load clock files from disk
- Compute barycentric delays (ephemeris lookups)
- Compute binary delays
- Compute spin phase
- Compute residuals

---

## What Changes vs. What Stays The Same

### STAYS THE SAME ✅
1. **Derivatives** - same `compute_spin_derivatives()`
2. **WLS solver** - same `wls_solve_svd()`
3. **Convergence logic** - same checks
4. **Parameters being fit** - same ['F0', 'F1']
5. **Final answer** - exact same fitted values

### CHANGES (Performance Only) 🚀
1. **Residual computation** - compute static parts ONCE
2. **No temp files** - keep data in memory
3. **JAX JIT** - compile the hot loop

---

## The Key Insight: What Actually Changes During Fitting?

When fitting F0 and F1, here's what changes each iteration:

| Component | Changes? | Why? |
|-----------|----------|------|
| Clock corrections | ❌ NO | TOA times don't change |
| Observatory positions | ❌ NO | Fixed in space |
| Earth positions | ❌ NO | Same MJDs every iteration |
| Barycentric delays (Roemer) | ❌ NO | Only depends on TOA times and sky position |
| Shapiro delays | ❌ NO | Only depends on TOA times |
| Binary delays | ❌ NO | Not fitting binary parameters |
| DM delays | ❌ NO | Not fitting DM |
| **Spin phase** | ✅ YES | **This is what F0/F1 affect!** |

**90% of the computation doesn't need to happen every iteration!**

---

## The Optimization (Minimal Changes)

### Version 1: What I Started (Too Ambitious)

I tried to create a completely new JAX function. **This was a mistake** - too much change!

### Version 2: What We Should Do (Minimal Change)

**Keep the exact same code structure, just don't recompute static things:**

```python
# BEFORE FITTING (happens ONCE)
print("Precomputing static delays...")
# Compute full residuals ONCE to get all delays
result_initial = compute_residuals_simple(par_file, tim_file, subtract_tzr=False)

# Extract the pieces we'll reuse
toas_mjd = extract_toas(tim_file)
errors_sec = extract_errors(tim_file)

# Key optimization: save the dt_sec values
# These already have ALL delays baked in (clock, bary, binary, DM)
dt_sec_with_delays = result_initial['dt_sec']  # This is emission time - pepoch

# Now store these for reuse
cache = {
    'toas_mjd': toas_mjd,
    'dt_sec_with_delays': dt_sec_with_delays,  # Reuse this!
    'errors_sec': errors_sec,
    'pepoch': params['PEPOCH']
}

# FITTING LOOP (happens 25 times)
for iteration in range(max_iter):
    # Option A: Still call compute_residuals_simple but skip expensive parts
    # Option B: Compute residuals directly from cached dt_sec
    
    # Option B (simpler):
    dt = cache['dt_sec_with_delays']
    phase = f0_current * dt + 0.5 * f1_current * dt**2
    residuals = toas_mjd - (pepoch + phase/(f0_current * SECS_PER_DAY))
    
    # Everything else SAME AS BEFORE
    derivs = compute_spin_derivatives(params, toas_mjd, ['F0', 'F1'])
    M = np.column_stack([derivs['F0'], derivs['F1']])
    delta_params, cov, _ = wls_solve_svd(residuals, errors, M)
    
    f0_current += delta_params[0]
    f1_current += delta_params[1]
```

**Result**: 21s → 5s (4x speedup from avoiding recomputation)

---

## Why This Is Safe

1. **Same physics**: We're computing the exact same residuals, just reusing intermediate results
2. **Same derivatives**: No change to derivative calculation
3. **Same solver**: No change to WLS solver
4. **Same convergence**: Same stopping criteria

**The only difference**: We compute the expensive delays ONCE instead of 25 TIMES.

---

## What dt_sec Actually Represents

Looking at the code in `simple_calculator.py`:

```python
# This is computed after ALL delays
dt_sec = (t_emission_mjd - pepoch) * SECS_PER_DAY

# Where t_emission_mjd includes:
# - Original TOA (UTC)
# + Clock corrections (mk→utc→tai→tt)
# + Barycentric delays (Roemer + Shapiro)
# + Binary delays (if applicable)
# - DM delay (frequency-dependent)
# = Pulsar emission time
```

**For F0/F1 fitting**: All these delays are independent of F0/F1, so we can cache `dt_sec` and reuse it!

---

## The Actual Implementation Plan

### Step 1: Extract dt_sec from current code

```python
def compute_dt_sec_once(par_file, tim_file):
    """Run full computation ONCE, extract dt_sec."""
    result = compute_residuals_simple(par_file, tim_file, subtract_tzr=False)
    return result['dt_sec']
```

### Step 2: Create fast residual function using cached dt_sec

```python
def compute_residuals_from_cached_dt(dt_sec, f0, f1, pepoch, toas_mjd, weights):
    """Fast residuals using pre-computed dt_sec."""
    # Compute spin phase
    phase = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    
    # Convert to residuals
    model_toa = pepoch + phase / (f0 * SECS_PER_DAY)
    residuals = (toas_mjd - model_toa) * SECS_PER_DAY
    
    # Subtract weighted mean
    mean_res = np.sum(residuals * weights) / np.sum(weights)
    residuals -= mean_res
    
    # RMS
    rms = np.sqrt(np.sum(residuals**2 * weights) / np.sum(weights))
    
    return residuals, rms * 1e6
```

### Step 3: Update fitting loop (minimal change!)

```python
# BEFORE LOOP
dt_sec_cached = compute_dt_sec_once(par_file, tim_file)
toas_mjd, errors_sec, weights = extract_toa_data(tim_file)
pepoch = params['PEPOCH']

# LOOP (exactly same structure as before!)
for iteration in range(max_iter):
    # NEW: Fast residuals using cache
    residuals_sec, rms_us = compute_residuals_from_cached_dt(
        dt_sec_cached, f0_curr, f1_curr, pepoch, toas_mjd, weights
    )
    
    # SAME AS BEFORE:
    derivs = compute_spin_derivatives(params, toas_mjd, ['F0', 'F1'])
    M = np.column_stack([derivs['F0'], derivs['F1']])
    delta_params, cov, _ = wls_solve_svd(residuals_sec, errors_sec, M)
    f0_curr += delta_params[0]
    f1_curr += delta_params[1]
    
    # SAME: convergence check
    if converged: break
```

---

## Why My Current Attempt Got Complicated

I tried to extract "emission times" from the result, but `compute_residuals_simple` doesn't directly return them. Instead, it returns `dt_sec` which is already perfect for our needs!

**The fix**: Just use `dt_sec` directly instead of trying to reconstruct emission times.

---

## Generalization to Other Parameters

This same pattern works for any parameter that doesn't affect the delays:

| Parameters Being Fit | What to Cache | What to Recompute |
|---------------------|---------------|-------------------|
| F0, F1, F2 | All delays (dt_sec) | Only spin phase |
| DM, DM1, DM2 | Clock + bary delays | DM delay + phase |
| PB, A1, ECC | Clock + bary delays | Binary delay + phase |
| All together | Clock + bary delays | Binary + DM + phase |

**For F0/F1 fitting (most common)**: We can cache everything except phase!

---

## Summary

**What we're doing**: 
- Computing expensive delays (clock, bary, binary) ONCE
- Reusing them for all 25 iterations
- Only recomputing the cheap part (spin phase) each iteration

**What stays the same**:
- Derivatives calculation
- WLS solver
- Convergence logic
- Final fitted values

**Expected speedup**: 4-5x (21s → 4-5s)

**Risk level**: LOW - we're just avoiding redundant computation, not changing the physics

---

## Next Steps (Simple!)

1. Use the existing `dt_sec` from `compute_residuals_simple`
2. Create a simple function that computes residuals from dt_sec
3. Test it matches the full computation exactly
4. Integrate into fitting loop

**Time estimate**: 30 minutes to get it working correctly!

