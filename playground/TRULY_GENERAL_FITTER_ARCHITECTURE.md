# Truly General Fitter Architecture

**Date**: 2025-12-01  
**Status**: ✅ ARCHITECTURE COMPLETE, AWAITING IMPLEMENTATIONS

---

## The Problem You Identified

The Session 16 implementation could only fit spin parameters. If you wanted to fit:
```bash
jug-fit pulsar.par pulsar.tim --fit F0 F1 DM RAJ DECJ PB
```

It would **fail** because it couldn't handle mixed parameter types.

**You were right** - we needed a truly general architecture.

---

## The Solution: Modular Design Matrix Construction

### Core Concept

Instead of special-casing each parameter type, we build the design matrix **column-by-column** using modular derivative functions:

```python
# For fit_params = ['F0', 'F1', 'DM', 'RAJ', 'DECJ', 'PB']
M = np.zeros((n_toas, 6))  # 6 columns, one per parameter

for i, param in enumerate(fit_params):
    if param.startswith('F'):
        M[:, i] = compute_spin_derivative(param, dt_sec, f0)
    elif param.startswith('DM'):
        M[:, i] = compute_dm_derivative(param, dt_sec, freq_mhz)
    elif param in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
        M[:, i] = compute_astrometry_derivative(param, sky_pos, ...)
    elif param in binary_params:
        M[:, i] = compute_binary_derivative(param, orbital_phase, ...)
```

**Key insight**: Each derivative function is independent. You can mix and match any parameters!

---

## Architecture

### Main Entry Point

```python
def fit_parameters_optimized(par_file, tim_file, fit_params, ...):
    """Public API - handles ANY parameter combination."""
    return _fit_parameters_general(
        par_file, tim_file, fit_params, ...
    )
```

### General Fitter

```python
def _fit_parameters_general(par_file, tim_file, fit_params, ...):
    """Truly general fitter - routes to appropriate implementation."""
    
    # 1. Parse files, extract starting values
    params = parse_par_file(par_file)
    param_values = [params[p] for p in fit_params]
    
    # 2. Categorize parameters
    spin_params = [p for p in fit_params if p.startswith('F')]
    dm_params = [p for p in fit_params if p.startswith('DM')]
    astrometry_params = [p for p in fit_params if p in ['RAJ', 'DECJ', ...]]
    binary_params = [p for p in fit_params if p in ['PB', 'A1', ...]]
    
    # 3. Check what's implemented
    if not_implemented_params:
        raise NotImplementedError("Need to implement X, Y, Z derivatives")
    
    # 4. Optimization: Use specialized fitter if all params are same type
    if only_spin_params:
        return _fit_spin_params_general(...)  # Fast path
    
    # 5. General path: Mixed parameter fitting
    return _fit_mixed_params(par_file, tim_file, fit_params, ...)
```

### Mixed Parameter Fitter (TODO)

```python
def _fit_mixed_params(par_file, tim_file, fit_params, ...):
    """Fitting loop for mixed parameter types."""
    
    # Cache expensive computations once
    dt_sec = cache_delays(par_file, tim_file)
    
    for iteration in range(max_iter):
        # 1. Compute residuals with current parameters
        residuals = compute_residuals(param_values, dt_sec, ...)
        
        # 2. Build design matrix column-by-column
        M = np.zeros((n_toas, len(fit_params)))
        for i, param in enumerate(fit_params):
            M[:, i] = compute_derivative_for(param, ...)
        
        # 3. Solve WLS
        delta_params = wls_solve(residuals, errors, M)
        
        # 4. Update parameters
        param_values += delta_params
        
        # 5. Check convergence
        if converged: break
    
    return results
```

### Derivative Router

```python
def compute_derivative_for(param, dt_sec, freq_mhz, params, ...):
    """Route to appropriate derivative function based on parameter."""
    
    if param.startswith('F'):
        return compute_spin_derivative(param, dt_sec, params['F0'])
    
    elif param.startswith('DM'):
        return compute_dm_derivative(param, dt_sec, freq_mhz)
    
    elif param in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
        return compute_astrometry_derivative(param, sky_pos, ...)
    
    elif param in binary_params:
        return compute_binary_derivative(param, orbital_phase, ...)
    
    else:
        raise ValueError(f"Unknown parameter: {param}")
```

---

## What's Implemented vs TODO

### ✅ Implemented (Session 16)

1. **Spin derivatives** (`F0`, `F1`, `F2`, ...)
   - File: `jug/fitting/derivatives_spin.py`
   - Function: `compute_spin_derivatives()`
   - Status: ✅ Working, tested, validated

2. **Optimized spin-only fitter**
   - Uses JAX JIT compilation
   - Performance: ~3s for 10k TOAs
   - Status: ✅ Production-ready

3. **General architecture**
   - `_fit_parameters_general()` entry point
   - Parameter categorization
   - Error checking
   - Status: ✅ Framework complete

### 🔄 TODO (Milestone 3)

1. **DM derivatives** (Easy - ~2 hours)
   ```python
   def compute_dm_derivative(param, dt_sec, freq_mhz):
       """d(time_residual)/d(DM_n)"""
       K_DM = 4.148808e3  # MHz² pc⁻¹ cm³ s
       if param == 'DM':
           return -K_DM / freq_mhz**2
       elif param == 'DM1':
           return -K_DM * dt_sec / freq_mhz**2
       elif param == 'DM2':
           return -K_DM * dt_sec**2 / 2 / freq_mhz**2
   ```

2. **Astrometry derivatives** (Medium - ~4 hours)
   ```python
   def compute_astrometry_derivative(param, toas, sky_pos, earth_pos):
       """d(time_residual)/d(position/proper motion/parallax)"""
       # Need to recompute barycentric delays with perturbed position
       # Use JAX autodiff or finite differences
   ```

3. **Binary derivatives** (Hard - ~8 hours)
   ```python
   def compute_binary_derivative(param, toas, binary_params):
       """d(time_residual)/d(binary_param)"""
       # Need to recompute binary delays with perturbed parameter
       # Use JAX autodiff for orbital equations
   ```

4. **Mixed parameter fitter loop**
   - Implement `_fit_mixed_params()`
   - Build design matrix column-by-column
   - Status: Architecture designed, needs implementation

---

## Testing Strategy

### Phase 1: Single Parameter Types (✅ Done)
```bash
jug-fit pulsar.par pulsar.tim --fit F0 F1       # ✅ Works
jug-fit pulsar.par pulsar.tim --fit F0          # ✅ Works
jug-fit pulsar.par pulsar.tim --fit F0 F1 F2    # ✅ Works
```

### Phase 2: DM Parameters (Milestone 3)
```bash
jug-fit pulsar.par pulsar.tim --fit DM          # TODO
jug-fit pulsar.par pulsar.tim --fit DM DM1      # TODO
jug-fit pulsar.par pulsar.tim --fit F0 DM       # TODO (mixed!)
```

### Phase 3: Astrometry (Milestone 3)
```bash
jug-fit pulsar.par pulsar.tim --fit RAJ DECJ    # TODO
jug-fit pulsar.par pulsar.tim --fit F0 F1 RAJ   # TODO (mixed!)
```

### Phase 4: Binary (Milestone 3)
```bash
jug-fit pulsar.par pulsar.tim --fit PB A1       # TODO
jug-fit pulsar.par pulsar.tim --fit F0 PB       # TODO (mixed!)
```

### Phase 5: Kitchen Sink (Goal!)
```bash
jug-fit pulsar.par pulsar.tim --fit F0 F1 DM RAJ DECJ PB A1
# ↑ This is the ultimate goal: fit EVERYTHING simultaneously
```

---

## Performance Considerations

### Specialized vs General Fitters

**Specialized** (current spin-only):
- All derivatives computed in JAX
- Entire iteration JIT-compiled
- Performance: ~1ms per iteration

**General** (mixed parameters):
- Design matrix built column-by-column
- Each derivative may use different approach
- Performance: ~2-5ms per iteration (estimated)

**Strategy**: Keep specialized fitters for single-type fits:
```python
if all_spin_params:
    return _fit_spin_params_general(...)  # Fast path (1ms)
elif all_dm_params:
    return _fit_dm_params_general(...)    # Fast path
else:
    return _fit_mixed_params(...)          # General path (5ms)
```

**Tradeoff**: 5× slower for mixed fits, but still fast enough (0.05s vs 0.01s per iteration).

---

## API Stability

### User-Facing API (Stable)
```python
fit_parameters_optimized(par_file, tim_file, fit_params=['F0', 'F1', 'DM'])
```

**This will never change!** You can fit any combination, and the function routes internally.

### Internal Functions (May Change)
- `_fit_parameters_general()` - Router
- `_fit_spin_params_general()` - Specialized spin fitter
- `_fit_mixed_params()` - General fitter (TODO)

These are internal implementation details.

---

## Current Status

### What Works NOW
```bash
# Any combination of spin parameters
jug-fit pulsar.par pulsar.tim --fit F0
jug-fit pulsar.par pulsar.tim --fit F0 F1
jug-fit pulsar.par pulsar.tim --fit F0 F1 F2
jug-fit pulsar.par pulsar.tim --fit F1 F2  # Even without F0!
```

### What Gives Helpful Errors
```bash
jug-fit pulsar.par pulsar.tim --fit F0 DM
# Error: DM derivatives not yet implemented (Coming in Milestone 3)

jug-fit pulsar.par pulsar.tim --fit F0 RAJ DECJ
# Error: Astrometry derivatives not yet implemented (Coming in Milestone 3)
```

### What to Implement Next (Priority Order)
1. **DM derivatives** - Easiest (2 hours)
2. **Mixed fitter loop** - Medium (4 hours)
3. **Astrometry derivatives** - Harder (6 hours)
4. **Binary derivatives** - Hardest (8-10 hours)

**Total estimate**: ~20 hours for full Milestone 3

---

## Summary

### Before (Session 16)
- Could only fit spin parameters
- Hardcoded for F0+F1 (then generalized to any F)
- Would error on mixed parameter types

### Now (Post-Session 16)
- ✅ Architecture supports **any parameter combination**
- ✅ Spin parameters fully working
- ✅ Clear error messages for unimplemented parameters
- ✅ Design in place for mixed parameter fitting
- 🔄 Awaiting derivative implementations (Milestone 3)

### Goal (End of Milestone 3)
```bash
# This will work:
jug-fit pulsar.par pulsar.tim --fit F0 F1 DM DM1 RAJ DECJ PMRA PMDEC PB A1 ECC
# Fit 11 parameters simultaneously from different physical models!
```

**The architecture is ready. Now we just need to implement the derivative functions.**

---

## Files Modified

1. **`jug/fitting/optimized_fitter.py`**
   - Added `_fit_parameters_general()` - Main router
   - Modified `fit_parameters_optimized()` to use general fitter
   - Kept `_fit_spin_params_general()` for optimization

2. **Documentation**
   - `TRULY_GENERAL_FITTER_ARCHITECTURE.md` - This file

---

## Next Session

When you're ready to implement Milestone 3:
1. Start with DM derivatives (easiest)
2. Implement `_fit_mixed_params()` loop
3. Test `--fit F0 DM` (mixed parameters!)
4. Add astrometry and binary derivatives
5. Test kitchen sink: `--fit F0 F1 DM RAJ DECJ PB`

**Estimated time**: 20-25 hours total for full Milestone 3
