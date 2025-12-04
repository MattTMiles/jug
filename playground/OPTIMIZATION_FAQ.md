# Optimization FAQ

**Date**: 2025-12-01
**Questions**: Best speedup? Generalizable?

---

## Question 1: Is this the best speedup we can get?

**Short answer**: No, but it's the best **low-risk** speedup.

### Current Performance Breakdown (per iteration, 0.85s total)

| Component | Time | Cacheable? | Depends on F0/F1? |
|-----------|------|------------|-------------------|
| Clock corrections | ~0.30s | ✅ YES | ❌ NO |
| Barycentric delays (ephemeris) | ~0.20s | ✅ YES | ❌ NO |
| Binary delays | ~0.10s | ✅ YES | ❌ NO |
| DM delay | ~0.08s | ✅ YES | ❌ NO |
| **Phase calculation** | ~0.05s | ❌ NO | ✅ YES |
| Derivatives | ~0.03s | ⚠️ PARTIAL | ⚠️ PARTIAL |
| WLS solve | ~0.02s | ❌ NO | ✅ YES |
| File I/O overhead | ~0.07s | ✅ YES | ❌ NO |

### Speedup Levels

#### Level 1: Cache dt_sec (my current plan)
- **Eliminates**: Clock (0.30s) + Bary (0.20s) + Binary (0.10s) + DM (0.08s) + I/O (0.07s)
- **Keeps**: Phase (0.05s) + Derivatives (0.03s) + WLS (0.02s)
- **Per-iteration**: 0.85s → 0.10s
- **Total (25 iters)**: 21s → 2.5s + 3s initial cache = **5.5s**
- **Speedup**: **3.8x faster**
- **Risk**: LOW - just reusing existing computation

#### Level 2: JAX JIT compile residual+derivative loop
- **Eliminates**: Python overhead in hot loop
- **Per-iteration**: 0.10s → 0.05s
- **Total**: 5.5s → 3.5s
- **Speedup**: **6x faster**
- **Risk**: MEDIUM - need to JAX-ify derivative computation

#### Level 3: Tempo2-style linearization
- **Key insight**: Use design matrix to UPDATE model, not just fit
- **Eliminates**: Multiple iterations (25 → 5)
- **Total**: 3.5s → 2.0s
- **Speedup**: **10x faster (approaching Tempo2!)**
- **Risk**: HIGH - changes the iteration logic significantly

#### Level 4: Full C++ rewrite
- **Like Tempo2**: Pure compiled code
- **Total**: 2.0s → 0.5s
- **Speedup**: **40x faster**
- **Risk**: EXTREME - not Python anymore!

### Recommendation: Progressive Approach

**Phase 1 (tonight)**: Level 1 - Cache dt_sec
- Time: 30 minutes
- Speedup: 3.8x (21s → 5.5s)
- Risk: LOW
- **Do this first!**

**Phase 2 (session 15)**: Level 2 - JAX JIT
- Time: 2-3 hours
- Speedup: 6x total (21s → 3.5s)
- Risk: MEDIUM
- Worth doing after Level 1 validates

**Phase 3 (future)**: Level 3 - Linearization
- Time: 1-2 days
- Speedup: 10x (21s → 2s)
- Risk: HIGH
- Need to be very careful

---

## Question 2: Can we make this generalizable to any parameter?

**Short answer**: YES! The pattern naturally extends.

### The General Pattern

```python
def fit_parameters(par_file, tim_file, fit_params):
    """
    Generalized fitting for any parameter set.
    
    Parameters
    ----------
    fit_params : list
        Parameters to fit, e.g., ['F0', 'F1', 'DM', 'RAJ', 'DECJ']
    """
    
    # 1. Categorize parameters by what delays they affect
    spin_params = [p for p in fit_params if p in ['F0', 'F1', 'F2', ...]]
    dm_params = [p for p in fit_params if p in ['DM', 'DM1', 'DM2', ...]]
    astrometry_params = [p for p in fit_params if p in ['RAJ', 'DECJ', 'PMRA', ...]]
    binary_params = [p for p in fit_params if p in ['PB', 'A1', 'ECC', ...]]
    
    # 2. Determine what can be cached
    if binary_params:
        # Fitting binary: can only cache clock + barycentric
        cache_clock = True
        cache_bary = True
        cache_binary = False
        cache_dm = (not dm_params)  # Cache if not fitting
    elif dm_params:
        # Fitting DM: can cache clock + bary + binary
        cache_clock = True
        cache_bary = True
        cache_binary = True
        cache_dm = False
    else:
        # Fitting only spin: can cache everything!
        cache_clock = True
        cache_bary = True
        cache_binary = True
        cache_dm = True
    
    # 3. Compute cached components once
    cached_delays = compute_cacheable_delays(
        par_file, tim_file,
        cache_clock, cache_bary, cache_binary, cache_dm
    )
    
    # 4. Fitting loop
    for iteration in range(max_iter):
        # Recompute only non-cached delays
        residuals = compute_residuals_from_cache(
            cached_delays,
            current_params,
            recompute_binary=(not cache_binary),
            recompute_dm=(not cache_dm)
        )
        
        # Compute derivatives for ALL fit parameters
        derivs = compute_all_derivatives(current_params, fit_params)
        
        # Standard WLS
        M = np.column_stack([derivs[p] for p in fit_params])
        delta = wls_solve(residuals, errors, M)
        
        # Update all parameters
        for i, param in enumerate(fit_params):
            current_params[param] += delta[i]
```

### Caching Strategy by Parameter Type

| Parameters | Cache Clock? | Cache Bary? | Cache Binary? | Cache DM? |
|------------|--------------|-------------|---------------|-----------|
| **F0, F1 only** | ✅ YES | ✅ YES | ✅ YES | ✅ YES |
| **F0, F1, DM** | ✅ YES | ✅ YES | ✅ YES | ❌ NO |
| **F0, F1, PB, A1** | ✅ YES | ✅ YES | ❌ NO | ✅ YES |
| **RAJ, DECJ** | ✅ YES | ❌ NO | ❌ NO | ✅ YES |
| **All mixed** | ✅ YES | ❌ NO | ❌ NO | ❌ NO |

**Rule**: Cache a delay component if NO fitted parameter affects it.

### Implementation: Modular Delay Computation

```python
class CachedDelays:
    """Cache delay components based on what's being fitted."""
    
    def __init__(self, par_file, tim_file, fit_params):
        self.fit_params = fit_params
        
        # Determine what to cache
        self.cache_clock = True  # Always cache clock
        self.cache_bary = not self._affects_bary(fit_params)
        self.cache_binary = not self._affects_binary(fit_params)
        self.cache_dm = not self._affects_dm(fit_params)
        
        # Compute and cache
        if self.cache_clock:
            self.clock_corrections = self._compute_clock_once()
        if self.cache_bary:
            self.bary_delays = self._compute_bary_once()
        if self.cache_binary:
            self.binary_delays = self._compute_binary_once()
        if self.cache_dm:
            self.dm_delays = self._compute_dm_once()
    
    def _affects_bary(self, params):
        """Do any params affect barycentric delays?"""
        return any(p in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX'] for p in params)
    
    def _affects_binary(self, params):
        """Do any params affect binary delays?"""
        return any(p in ['PB', 'A1', 'ECC', 'OM', 'T0', 'TASC', ...] for p in params)
    
    def _affects_dm(self, params):
        """Do any params affect DM delays?"""
        return any(p in ['DM', 'DM1', 'DM2', ...] for p in params)
    
    def compute_residuals(self, current_params):
        """Compute residuals using cached + recomputed delays."""
        # Start with cached components
        total_delay = 0.0
        
        if self.cache_clock:
            total_delay += self.clock_corrections
        else:
            total_delay += compute_clock(current_params)
        
        if self.cache_bary:
            total_delay += self.bary_delays
        else:
            total_delay += compute_bary(current_params)
        
        if self.cache_binary:
            total_delay += self.binary_delays
        else:
            total_delay += compute_binary(current_params)
        
        if self.cache_dm:
            total_delay += self.dm_delays
        else:
            total_delay += compute_dm(current_params)
        
        # Compute phase (never cached)
        phase = compute_phase(current_params, total_delay)
        
        return phase_to_residuals(phase)
```

### Speedup by Parameter Combination

| Fit Parameters | Speedup | Why |
|----------------|---------|-----|
| F0, F1 | **4x** | Cache everything except phase |
| F0, F1, DM | **3x** | Must recompute DM delay |
| F0, F1, PB, A1 | **2.5x** | Must recompute binary delay |
| RAJ, DECJ | **2x** | Must recompute barycentric delays |
| All together | **1.5x** | Only cache clock corrections |

**Best case (spin only)**: 4x speedup
**Worst case (everything)**: Still 1.5x speedup from caching clocks!

---

## Implementation Plan for Generalization

### Step 1: Modular delay extraction (tonight)
```python
def extract_delay_components(par_file, tim_file):
    """Extract all delay components separately."""
    result = compute_residuals_simple(par_file, tim_file)
    
    return {
        'dt_sec': result['dt_sec'],  # Total for now
        # Future: Break down into components
        # 'clock_delay': ...,
        # 'bary_delay': ...,
        # 'binary_delay': ...,
        # 'dm_delay': ...,
    }
```

### Step 2: Component-wise caching (session 15)
```python
def compute_cacheable_components(par_file, tim_file, fit_params):
    """Compute and return only cacheable components."""
    # Analyze what can be cached
    # Compute and return those components
    # Leave rest for recomputation
```

### Step 3: Generalized fitter (session 16)
```python
def fit_any_parameters(par_file, tim_file, fit_params):
    """Fit any combination of parameters efficiently."""
    cache = build_smart_cache(par_file, tim_file, fit_params)
    
    for iteration in range(max_iter):
        residuals = cache.compute_residuals(current_params)
        derivs = compute_derivatives(current_params, fit_params)
        # ... standard fitting ...
```

---

## Summary

### Question 1: Best speedup?

**No**, Level 1 gives 4x. Potential levels:
- Level 1 (cache dt_sec): **4x** - LOW RISK ✅ Do tonight!
- Level 2 (JAX JIT): **6x** - MEDIUM RISK
- Level 3 (linearization): **10x** - HIGH RISK
- Level 4 (C++ rewrite): **40x** - Not Python anymore

**Recommendation**: Start with Level 1 (4x), proven safe and quick to implement.

### Question 2: Generalizable?

**YES!** The pattern extends naturally:

```python
# Works for ANY parameter combination
fit_any_parameters(par_file, tim_file, ['F0', 'F1'])           # 4x speedup
fit_any_parameters(par_file, tim_file, ['F0', 'F1', 'DM'])     # 3x speedup  
fit_any_parameters(par_file, tim_file, ['RAJ', 'DECJ', 'PX'])  # 2x speedup
fit_any_parameters(par_file, tim_file, ['PB', 'A1', 'ECC'])    # 2.5x speedup
```

**Key**: Cache only delays that don't depend on fitted parameters.

---

## Recommended Action

**Tonight**: Implement Level 1 for F0/F1 (the 80/20 case)
- Simple, safe, 4x speedup
- Validates the approach
- Foundation for generalization

**Next session**: Generalize to other parameter types
- Same pattern, more cases
- Still safe and predictable

**Future**: Consider Level 2/3 if needed
- Diminishing returns vs. complexity
- Maybe not worth it if 4-6x is good enough

