# JUG Performance Optimization Plan

**Current Performance**: 21.15s for F0+F1 fitting (10,408 TOAs)
**Target**: Approach Tempo2's 2.06s (10x speedup goal)

---

## Current Bottleneck Analysis

Per-iteration cost: 0.85s
- **Residual computation: 0.78s (92%)**
  - Clock corrections: ~0.3s
  - Ephemeris lookups: ~0.2s
  - Binary delays: ~0.1s
  - TZR handling: ~0.1s
  - Phase calculation: ~0.08s
- Derivatives + WLS: 0.07s (8%)

**Critical issue**: JUG recomputes EVERYTHING each iteration!

---

## Why Tempo2 Only Needs 1 Iteration

### Tempo2's Secret: Incremental Updates

Tempo2 doesn't recompute the full timing model each iteration!

**Tempo2's approach**:
1. Compute full model ONCE at start
2. Each iteration: Update ONLY the parameter-dependent parts
3. Uses **linearization** - assumes small parameter changes

**What Tempo2 caches between iterations**:
- ✅ Clock corrections (never change!)
- ✅ Observatory positions (fixed)
- ✅ Barycentric positions (fixed for given MJDs)
- ✅ Ephemeris delays (fixed)
- ✅ Binary delays (mostly fixed, only parameter-dependent parts update)
- ❌ Spin phase (updates with F0/F1 changes)

**JUG's current approach** (inefficient!):
1. Create NEW par file with updated F0/F1
2. Call `compute_residuals_simple()` which:
   - Reads par file from scratch
   - Reads clock files from scratch
   - Computes ephemeris from scratch
   - Computes binary delays from scratch
   - Computes everything from scratch!

**This is why JUG is 10x slower!** We're redoing work that doesn't need redoing.

---

## Optimization Strategy

### Phase 1: Cache Static Components (Easy - 5x speedup)

**Implementation**: Compute once, reuse for all iterations

```python
class CachedTimingData:
    """Cache components that don't change during fitting."""
    
    def __init__(self, par_file, tim_file):
        # Compute ONCE
        self.toas_mjd = parse_toas(tim_file)
        self.clock_corrections = compute_clock_corrections(self.toas_mjd)
        self.tdb_times = compute_tdb(self.toas_mjd, self.clock_corrections)
        self.barycentric_delays = compute_barycentric_delays(self.tdb_times)
        self.binary_delays = compute_binary_delays(self.tdb_times)  # Most of it
        self.dm_delays = compute_dm_delays(freqs)
        
    def compute_residuals(self, f0, f1):
        """Fast residual update - only recompute phase!"""
        # Use cached delays
        t_emission = self.tdb_times + self.barycentric_delays + self.binary_delays
        
        # ONLY recompute phase (fast!)
        phase = compute_phase(t_emission, f0, f1, pepoch)
        residuals = toas - phase / f0
        return residuals
```

**Expected speedup**: 0.85s → 0.15s per iteration (5-6x faster!)

**Impact on total time**: 21.15s → ~5s (with 25 iterations)

---

### Phase 2: Linearized Updates (Medium - 10x speedup)

**Key insight**: For small parameter changes, delays change linearly!

**Binary delay linearization**:
```python
# Compute once:
binary_delay_0 = compute_binary_delay(pb_0, a1_0, tasc_0, ...)
d_binary_d_pb = compute_binary_derivative_pb(...)

# Each iteration (fast!):
binary_delay = binary_delay_0 + d_binary_d_pb * (pb - pb_0)
```

**This is what Tempo2 does!** It uses the design matrix not just for the fit, but also to update the model.

**Expected speedup**: Another 2x on top of caching

**Impact on total time**: 21.15s → 2-3s

---

### Phase 3: JIT Compile Everything (Hard - 2x speedup)

**Current issue**: Even with caching, Python overhead is significant

**Solution**: Make `compute_residuals()` a pure JAX function

```python
@jax.jit
def compute_residuals_jit(t_emission, f0, f1, pepoch):
    """Fully JIT-compiled residual calculation."""
    dt = t_emission - pepoch
    phase = f0 * dt + 0.5 * f1 * dt**2
    # ... rest of calculation
    return residuals
```

**Expected speedup**: 2x on inner loop

**Impact on total time**: 2-3s → 1-1.5s

---

## Implementation Priority

### Quick Win #1: Cache Static Delays (1-2 hours work)

**File to modify**: `jug/residuals/simple_calculator.py`

Add caching layer:
```python
class FittingResiduals:
    def __init__(self, par_file, tim_file):
        # Parse once
        self.params = parse_par_file(par_file)
        self.toas = parse_tim_file(tim_file)
        
        # Compute static components ONCE
        self._compute_static_components()
        
    def _compute_static_components(self):
        """Compute delays that don't depend on fitted parameters."""
        # Clock corrections (never change)
        self.clock_corr = compute_clock_corrections(self.toas)
        
        # Barycentric delays (don't depend on F0/F1/DM)
        self.bary_delay = compute_barycentric_delays(...)
        
        # Binary delays (mostly static if not fitting binary params)
        self.binary_delay = compute_binary_delays(...)
        
    def residuals(self, f0, f1):
        """Fast residuals using cached components."""
        # Recompute ONLY phase
        t_emission = self.toas + self.clock_corr + self.bary_delay + self.binary_delay
        phase = f0 * (t_emission - pepoch) + 0.5 * f1 * (t_emission - pepoch)**2
        return self.toas - phase / f0
```

**Expected result**: 21s → 4-5s

---

### Quick Win #2: Avoid File I/O in Loop (30 minutes)

**Current problem**: Creating temp par file each iteration

**Solution**: Pass parameters directly to residual function

```python
# BEFORE (slow):
for iteration in range(max_iter):
    # Create temp par file
    with open(temp_par, 'w') as f:
        f.write(f"F0 {f0_curr}\n")
        f.write(f"F1 {f1_curr}\n")
    
    # Compute residuals (re-parses file!)
    result = compute_residuals_simple(temp_par, tim_file)

# AFTER (fast):
cached = FittingResiduals(par_file, tim_file)
for iteration in range(max_iter):
    # Direct parameter update (no I/O!)
    residuals = cached.residuals(f0_curr, f1_curr)
```

**Expected result**: Eliminate file I/O overhead (~0.1s per iteration)

---

### Medium Win: Linearized Binary Delays (2-3 hours)

**Implementation**: Add to derivatives module

```python
def compute_binary_delays_with_derivatives(params, toas):
    """Compute binary delays AND their derivatives."""
    delays = compute_binary_delays(params, toas)
    
    # Also compute derivatives
    d_delay_d_pb = ...
    d_delay_d_a1 = ...
    
    return delays, {
        'PB': d_delay_d_pb,
        'A1': d_delay_d_a1,
        ...
    }

def update_binary_delays_linearized(delays_0, derivatives, delta_params):
    """Fast update using linearization."""
    delays = delays_0.copy()
    for param, delta in delta_params.items():
        if param in derivatives:
            delays += derivatives[param] * delta
    return delays
```

**Expected result**: 4-5s → 2-3s

---

## Why Doesn't PINT Do This?

**PINT does cache!** But:
1. PINT recomputes TZR each iteration (for generality)
2. PINT's architecture is more general (harder to optimize)
3. PINT prioritizes correctness over speed

**Tempo2's advantage**:
- Written in C/C++ (10x faster baseline)
- Highly optimized over 20 years
- Uses every trick in the book

---

## Realistic Performance Targets

With optimizations:

| Optimization Level | Time | Speedup |
|-------------------|------|---------|
| **Current** | 21.15s | 1x |
| **+ Caching** | 4-5s | 4-5x |
| **+ Linearization** | 2-3s | 7-10x |
| **+ Full JIT** | 1-2s | 10-20x |

**Best case**: Approach Tempo2's 2s (within 2x is realistic for Python!)

---

## Action Items

### Immediate (Session 15)
1. ✅ Implement `FittingResiduals` class with caching
2. ✅ Eliminate temp file creation in fitting loop
3. ✅ Benchmark: Should see 4-5x speedup

### Short-term (Session 16-17)
4. Implement linearized binary delays
5. Cache ephemeris lookups
6. Benchmark: Should see 7-10x total speedup

### Long-term (Milestone 3)
7. Full JAX JIT compilation of residuals
8. Vectorize across multiple pulsars
9. GPU acceleration for batch processing

---

## Key Insight

**The real bottleneck isn't computation - it's redundant computation!**

Tempo2 is fast because it computes each thing ONCE.
JUG is slow because it computes each thing 25 TIMES.

Fix: Cache everything that doesn't depend on fitted parameters.

---

## Next Steps

Ready to implement caching layer?

Expected: 21s → 4-5s with ~1 hour of work!

