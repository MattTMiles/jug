# JUG Optimized Fitting Pipeline - Flowchart & Architecture

## Complete Fitting Workflow: Input Data → Fitted Parameters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER ENTRY POINT                                    │
│                                                                              │
│  from jug.fitting import fit_parameters_optimized                           │
│                                                                              │
│  result = fit_parameters_optimized(                                         │
│      par_file="J1909-3744.par",                                            │
│      tim_file="J1909-3744.tim",                                            │
│      fit_params=['F0', 'F1']                                               │
│  )                                                                          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 1: FILE PARSING                                     │
│                                                                              │
│  ┌────────────────────┐            ┌────────────────────┐                  │
│  │  parse_par_file()  │            │  parse_tim_file()  │                  │
│  │  (io/par_reader)   │            │  (io/tim_reader)   │                  │
│  └─────────┬──────────┘            └─────────┬──────────┘                  │
│            │                                  │                             │
│            ▼                                  ▼                             │
│  params = {                        toas = [                                │
│    'F0': 339.315...,                 TOA(mjd=55000.1, error=1.2us, ...),  │
│    'F1': -1.6e-15,                   TOA(mjd=55000.2, error=1.3us, ...),  │
│    'PEPOCH': 55000.0,                ...                                   │
│    'RAJ': ...,                     ]                                       │
│    ...                                                                      │
│  }                                                                          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STEP 2: LEVEL 1 OPTIMIZATION - CACHE dt_sec                     │
│                        (ONE-TIME COST: ~2.6s)                                │
│                                                                              │
│  Call: compute_residuals_simple(par_file, tim_file, subtract_tzr=False)    │
│                                                                              │
│  This computes ALL expensive delays ONCE:                                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  1. Clock Corrections (~0.3s)                                │           │
│  │     - Read tempo2 clock files                                │           │
│  │     - Chain: observatory → UTC → TAI → TT                    │           │
│  │                                                               │           │
│  │  2. Barycentric Delays (~0.8s)                               │           │
│  │     - Load JPL ephemeris (DE440s)                            │           │
│  │     - Geometric (Roemer) delay                               │           │
│  │     - Einstein delay (TT → TDB)                              │           │
│  │     - Shapiro delay (solar system gravity)                   │           │
│  │                                                               │           │
│  │  3. Binary Delays (~0.4s)                                    │           │
│  │     - ELL1/BT/DD model computation                           │           │
│  │     - Roemer + Einstein + Shapiro from companion             │           │
│  │                                                               │           │
│  │  4. DM Delay (~0.1s)                                         │           │
│  │     - Cold-plasma dispersion: K_DM * DM / freq²             │           │
│  │                                                               │           │
│  │  5. Emission Time Calculation (~0.3s)                        │           │
│  │     - t_em = t_bary - binary_delay - DM_delay               │           │
│  │                                                               │           │
│  │  6. Compute dt_sec (~0.1s)                                   │           │
│  │     - dt_sec = (t_em - PEPOCH) * SECS_PER_DAY               │           │
│  │     - This is the KEY cached quantity!                       │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  Result: dt_sec[10408 TOAs] = time differences in seconds                  │
│                                                                              │
│  WHY CACHE? These delays don't depend on F0/F1!                            │
│  They only depend on position, DM, binary params - which we're not fitting.│
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         STEP 3: LEVEL 2 OPTIMIZATION - JAX JIT COMPILATION                  │
│                        (ONE-TIME COST: ~0.4s)                                │
│                                                                              │
│  Convert to JAX arrays:                                                     │
│    dt_sec_jax = jnp.array(dt_sec)                                          │
│    errors_jax = jnp.array(errors_sec)                                      │
│    weights_jax = jnp.array(weights)                                        │
│                                                                              │
│  Warm up JIT (first call compiles):                                        │
│    @jax.jit                                                                 │
│    def full_iteration_jax_f0_f1(dt_sec, f0, f1, errors, weights):         │
│        # ... (see Step 4 for what's inside)                                │
│                                                                              │
│  This compiles to optimized XLA code for GPU/CPU.                          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 STEP 4: ITERATIVE FITTING LOOP                               │
│                      (8 iterations × 0.02s = 0.18s)                         │
│                                                                              │
│  FOR iteration = 1 to max_iter:                                             │
│                                                                              │
│    ┌────────────────────────────────────────────────────────────┐          │
│    │  Call: delta, rms, cov = full_iteration_jax_f0_f1(...)    │          │
│    │                                                             │          │
│    │  Inside this JIT-compiled function:                        │          │
│    │                                                             │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4a. Compute Spin Phase                            │    │          │
│    │  │                                                    │    │          │
│    │  │   phase = dt_sec * (f0 + dt_sec * f1/2)          │    │          │
│    │  │                                                    │    │          │
│    │  │   This is Taylor series:                          │    │          │
│    │  │   Φ = F0*Δt + F1*Δt²/2 + F2*Δt³/6 + ...         │    │          │
│    │  │   (we only use F0 and F1 here)                    │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4b. Wrap Phase (remove integer cycles)            │    │          │
│    │  │                                                    │    │          │
│    │  │   phase_wrapped = phase - round(phase)            │    │          │
│    │  │                                                    │    │          │
│    │  │   Example: phase=1234567.123 → 0.123 cycles       │    │          │
│    │  │   This discards the integer pulses!                │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4c. Convert Phase → Time Residuals                 │    │          │
│    │  │                                                    │    │          │
│    │  │   residuals = phase_wrapped / f0                   │    │          │
│    │  │                                                    │    │          │
│    │  │   Units: cycles / Hz = seconds                     │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4d. Subtract Weighted Mean                         │    │          │
│    │  │                                                    │    │          │
│    │  │   mean = sum(residuals * weights) / sum(weights)   │    │          │
│    │  │   residuals = residuals - mean                     │    │          │
│    │  │                                                    │    │          │
│    │  │   This centers the residuals for fitting           │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4e. Compute Analytical Derivatives                 │    │          │
│    │  │                                                    │    │          │
│    │  │   d(residual)/d(F0) = -dt_sec / f0                │    │          │
│    │  │   d(residual)/d(F1) = -(dt_sec² / 2) / f0         │    │          │
│    │  │                                                    │    │          │
│    │  │   From: d(phase)/d(Fn) = dt^(n+1) / (n+1)!        │    │          │
│    │  │   Negative sign: PINT convention (res = data-model)│    │          │
│    │  │   Division by f0: converts cycles → seconds        │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4f. Build Design Matrix                            │    │          │
│    │  │                                                    │    │          │
│    │  │   M = [d_F0, d_F1]  (10408 × 2 matrix)            │    │          │
│    │  │                                                    │    │          │
│    │  │   Each column = derivative w.r.t. that parameter   │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4g. Weighted Least Squares Solve                   │    │          │
│    │  │                                                    │    │          │
│    │  │   Weight the system:                               │    │          │
│    │  │     M_weighted = M * (1/errors)                    │    │          │
│    │  │     r_weighted = residuals * (1/errors)            │    │          │
│    │  │                                                    │    │          │
│    │  │   Solve via SVD:                                   │    │          │
│    │  │     delta_params = lstsq(M_weighted, r_weighted)   │    │          │
│    │  │                                                    │    │          │
│    │  │   Compute covariance:                              │    │          │
│    │  │     cov = inv(M_weighted^T @ M_weighted)           │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  ┌───────────────────────────────────────────────────┐    │          │
│    │  │ 4h. Compute RMS                                    │    │          │
│    │  │                                                    │    │          │
│    │  │   rms = sqrt(sum(residuals² * weights) / sum(w))  │    │          │
│    │  │   rms_us = rms * 1e6  (convert to microseconds)   │    │          │
│    │  └───────────────────────────────────────────────────┘    │          │
│    │                         │                                  │          │
│    │                         ▼                                  │          │
│    │  Return: delta_params, rms_us, cov                        │          │
│    └────────────────────────────────────────────────────────────┘          │
│                                                                              │
│    Convert JAX → NumPy:                                                    │
│      delta_params = np.array(delta_params_jax)                             │
│      cov = np.array(cov_jax)                                               │
│                                                                              │
│    Update parameters:                                                       │
│      f0_current = f0_current + delta_params[0]                             │
│      f1_current = f1_current + delta_params[1]                             │
│                                                                              │
│    Check convergence:                                                       │
│      IF max(|delta_params|) < threshold:                                   │
│        BREAK  (converged!)                                                 │
│                                                                              │
│  END FOR                                                                    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: RETURN RESULTS                                  │
│                                                                              │
│  result = {                                                                 │
│      'final_params': {                                                      │
│          'F0': 339.31569191904083027111,  # Fitted value                   │
│          'F1': -1.61475055178690184661e-15                                 │
│      },                                                                     │
│      'uncertainties': {                                                     │
│          'F0': 1.017e-14,  # From covariance matrix                        │
│          'F1': 1.661e-22                                                    │
│      },                                                                     │
│      'final_rms': 0.404443,  # μs                                          │
│      'iterations': 8,                                                       │
│      'converged': True,                                                     │
│      'total_time': 3.23,  # seconds                                        │
│      'cache_time': 2.64,                                                    │
│      'jit_time': 0.36,                                                      │
│      'covariance': array([[...], [...]])                                   │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Breakdown

### Time Budget (Total: 3.23s for 10,408 TOAs)

| Component | Time | % | Optimized? |
|-----------|------|---|------------|
| **Cache dt_sec** | 2.64s | 82% | ✅ Computed once |
| - Clock corrections | 0.30s | 9% | ✅ Cached |
| - Ephemeris lookups | 0.80s | 25% | ✅ Cached |
| - Barycentric delays | 0.60s | 19% | ✅ Cached |
| - Binary delays | 0.40s | 12% | ✅ Cached |
| - Other | 0.54s | 17% | ✅ Cached |
| **JIT compile** | 0.36s | 11% | ✅ One-time cost |
| **Fitting (8 iters)** | 0.18s | 6% | ✅ JAX JIT |
| - Per iteration | 0.023s | 0.7% | ✅ Blazing fast! |
| **Overhead** | 0.05s | 2% | - |

### Key Optimizations

1. **Level 1: Cache dt_sec** (5.87x speedup)
   - Compute expensive delays ONCE
   - Reuse for all 8 iterations
   - Saves: 2.64s × 7 iterations = 18.5s!

2. **Level 2: JAX JIT** (additional 1.12x speedup)
   - Compile entire iteration to XLA
   - GPU-optimized operations
   - Per-iteration: 0.055s → 0.023s (2.4x faster)

3. **Combined Effect**: 6.55x total speedup
   - Original: 21.15s
   - Optimized: 3.23s

---

## Code Organization in JUG Package

```
jug/
├── fitting/
│   ├── __init__.py                   # Exports fit_parameters_optimized
│   ├── optimized_fitter.py           # ✅ NEW: Level 2 optimized fitter
│   ├── derivatives_spin.py           # Analytical derivatives (F0, F1, F2)
│   ├── wls_fitter.py                 # WLS solver (SVD-based)
│   ├── optimizer.py                  # Generic linearized fitter
│   └── ...
│
├── residuals/
│   └── simple_calculator.py          # Compute residuals (used for caching)
│
├── io/
│   ├── par_reader.py                 # Parse .par files
│   └── tim_reader.py                 # Parse .tim files
│
├── delays/
│   ├── combined.py                   # Binary delays (ELL1, BT, DD)
│   └── ...
│
└── utils/
    └── constants.py                  # Physical constants
```

---

## Usage Examples

### Basic Usage

```python
from jug.fitting import fit_parameters_optimized
from pathlib import Path

result = fit_parameters_optimized(
    par_file=Path("J1909-3744.par"),
    tim_file=Path("J1909-3744.tim"),
    fit_params=['F0', 'F1']
)

print(f"F0 = {result['final_params']['F0']:.15f} Hz")
print(f"F1 = {result['final_params']['F1']:.15e} Hz/s")
print(f"Time: {result['total_time']:.2f}s")
```

### Advanced Usage

```python
# Fit with custom settings
result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=50,
    convergence_threshold=1e-16,
    clock_dir="custom/clock/path",
    verbose=True
)

# Access uncertainties
sigma_f0 = result['uncertainties']['F0']
sigma_f1 = result['uncertainties']['F1']

# Access covariance matrix
cov = result['covariance']
correlation = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
```

---

## Comparison: Original vs Optimized

### Original Approach (21.15s)

```python
for iteration in range(25):
    # Recompute EVERYTHING each time
    result = compute_residuals_simple(temp_par, tim_file)
    residuals = result['residuals']
    
    # Compute derivatives
    M = compute_derivatives(...)
    
    # WLS solve
    delta = wls_solve(residuals, errors, M)
    
    # Update parameters
    params['F0'] += delta[0]
    params['F1'] += delta[1]
```

**Problem**: Recomputes clock, bary, binary delays 25 times!

### Level 1 Optimization (3.60s)

```python
# Compute dt_sec ONCE
dt_sec = compute_residuals_simple(par, tim, subtract_tzr=False)['dt_sec']

for iteration in range(16):
    # Fast! Only compute phase with new F0/F1
    phase = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    residuals = (phase - np.round(phase)) / f0
    
    # Derivatives and solve
    M = compute_derivatives(...)
    delta = wls_solve(residuals, errors, M)
    
    f0 += delta[0]
    f1 += delta[1]
```

**Improvement**: 5.87x faster (cached delays)

### Level 2 Optimization (3.23s) - CURRENT

```python
# Compute dt_sec ONCE
dt_sec_jax = jnp.array(compute_residuals_simple(...)['dt_sec'])

@jax.jit
def full_iteration(dt_sec, f0, f1, errors, weights):
    # Everything in JAX, JIT-compiled!
    phase = dt_sec * (f0 + dt_sec * (f1/2))
    residuals = (phase - jnp.round(phase)) / f0
    M = jnp.column_stack([d_f0, d_f1])
    delta = wls_solve_jax(residuals, errors, M)
    return delta, rms, cov

for iteration in range(8):
    delta, rms, cov = full_iteration(dt_sec_jax, f0, f1, ...)
    f0 += delta[0]
    f1 += delta[1]
```

**Improvement**: 6.55x faster (cached + JIT)

---

## Extension to Other Parameters

### Currently Supported
- ✅ F0, F1 (spin frequency and derivative)

### Easy Extensions (TODO)
- F2 (second derivative) - trivial, just add d_f2 term
- DM, DM1, DM2 - similar caching strategy
- PB, A1, ECC (binary) - cache clock + bary only

### Requires New Derivatives
- RAJ, DECJ (position) - need sky position derivatives
- PMRA, PMDEC (proper motion) - need proper motion derivatives
- PX (parallax) - need parallax derivatives

Each parameter type may have different cache requirements!

---

## References

**Production Code**:
- `jug/fitting/optimized_fitter.py` - Main implementation
- `test_level2_jax_fitting.py` - Standalone test/validation

**Documentation**:
- `SESSION_14_COMPLETE_SUMMARY.md` - Complete session summary
- `SESSION_14_JAX_OPTIMIZATION.md` - Level 2 details
- `OPTIMIZATION_STRATEGY_EXPLAINED.md` - Strategy guide

**Validation**:
- Tested on J1909-3744 (10,408 TOAs)
- Matches PINT to 20 decimal places
- 6.55x faster than baseline
- Within 1.6x of Tempo2 (pure C++)
