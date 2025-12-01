# Milestone 2 Fitting Progress - Session 8

**Date**: 2025-11-30
**Status**: IN PROGRESS (90%)

## What Was Done

### 1. Created JAX Residual Core (`jug/residuals/core.py`)

**Goal**: Separate slow one-time operations from fast residual computation

**Approach**:
- `prepare_fixed_data()` - Runs once, computes TDB times, geometric delays, etc.
- `compute_residuals_jax()` - Fast JAX function that only updates spin/DM parameters
- Splits computation into:
  - **Slow (one-time)**: File I/O, TDB conversion, ephemeris lookups, geometric delays
  - **Fast (many times)**: Spin phase, DM delay (parameters being fitted)

**Implementation**:
- Pure JAX residual function with JIT compilation
- Pre-computation of all position-dependent delays
- Wrapper function to extract parameters from arrays

### 2. Created Test Script (`test_jax_fitting_integration.py`)

Tests correctness and speed of JAX residuals vs baseline.

## Current Status

### What Works ✅
- JAX phase computation
- DM delay computation
- Pre-computation infrastructure
- TZR phase handling

### What's Not Working ❌
- Binary delay integration
  - Simple_calculator includes binary delays
  - JAX version currently doesn't
  - Need to either:
    1. Pre-compute binary delays and add to geometric_delay
    2. Or re-implement binary models in pure JAX

### Root Cause of Test Failure

The residuals differ by ~10^12 μs because:
1. Baseline (simple_calculator) includes: geometric + DM + binary delays
2. JAX version only includes: geometric + DM delays
3. Missing binary delay ≈ 1.2 seconds for J1909-3744

## Solution Options

### Option A: Pre-compute Binary Delays ⭐ RECOMMENDED
```python
def prepare_fixed_data():
    # ...existing code...
    
    # Compute all delays using simple_calculator
    result = compute_residuals_simple(par_file, tim_file)
    
    # Extract total delay from result
    # total = geometric + DM + binary
    # We want: total - DM_at_ref + DM_fitted
    
    # Store (geometric + binary) as fixed
    # Let JAX recompute DM when params change
```

**Pros**: Simple, reuses existing code
**Cons**: Limits fitting to spin + DM parameters only

### Option B: Pure JAX Binary Models
Reimplement ELL1/DD/BT models as pure JAX functions.

**Pros**: Can fit binary parameters
**Cons**: More work, needs careful validation

### Option C: Test on Non-Binary Pulsar First
Use J0030+0451 (isolated MSP) to validate spin+DM fitting works.

**Pros**: Simpler test case
**Cons**: Delays full validation

## Recommended Next Steps

1. **Implement Option A** (30 min)
   - Extract total delay from simple_calculator
   - Subtract reference DM delay
   - Store as `total_delay_minus_dm` in fixed_data
   - In JAX: residual = phase(tdb - total_delay_minus_dm - dm_fitted)

2. **Validate on J1909-3744** (10 min)
   - Should match simple_calculator exactly
   - Check RMS < 0.001 μs

3. **Test Speed** (5 min)
   - First call: ~1 sec (JIT compilation)
   - Subsequent: <10 ms per call

4. **Integrate with Gauss-Newton** (1 hour)
   - Use `compute_residuals_jax` in design matrix
   - Test fitting on synthetic data
   - Then real data

## Time Estimate

- Fix binary delay issue: 30 min
- Validate correctness: 15 min
- Integrate with fitter: 1 hour
- **Total remaining**: ~2 hours

## Files Modified

- `jug/residuals/core.py` - New JAX residual functions
- `test_jax_fitting_integration.py` - Test script

## Files To Modify

- `jug/residuals/core.py` - Fix binary delay handling
- `jug/fitting/gauss_newton_jax.py` - Update to use new residual function
- `test_fitting_real.py` - Create real data fitting test

## Key Insight

The fundamental issue was trying to call a file-based function (`compute_residuals_simple`) inside the fitting loop. The solution is to:

1. **Call it once** to get all delays
2. **Extract delays** as numpy arrays
3. **Pass arrays** to JAX function
4. **Re-fit only** parameters that affect delays we can compute fast (spin, DM)

For binary parameters, we'd need pure JAX binary models (future work).
