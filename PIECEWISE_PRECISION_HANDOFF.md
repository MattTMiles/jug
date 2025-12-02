# Piecewise/Hybrid Precision Method - Handoff Document

## Background

Pulsar timing requires computing **phase residuals** - the difference between observed pulse arrival times and the predicted times from a timing model. The critical calculation is:

```
dt_sec = tdb_sec - PEPOCH_sec - total_delay_sec
phase = F0 * dt_sec + (F1/2) * dt_sec² + (F2/6) * dt_sec³
```

Where:
- `tdb_sec` = TDB time in seconds (~5.1e9 for MJD ~59000)
- `PEPOCH_sec` = Reference epoch in seconds (~5.1e9)
- `dt_sec` = Time since PEPOCH (~10⁸ seconds for multi-year datasets)
- `F0` ≈ 339 Hz (spin frequency)
- `F1` ≈ -1.6e-15 Hz/s (spin-down)

### The Precision Problem

When computing `phase = F0 * dt_sec`:
- `dt_sec` ≈ 10⁸ seconds (spanning years of data)
- `F0 * dt_sec` ≈ 3.4×10¹⁰ cycles
- We need the **fractional part** to ~10⁻⁶ cycle precision (sub-microsecond timing)
- float64 has ~16 digits → 3.4×10¹⁰ with 16 digits → only ~6 digits after decimal
- This is borderline for nanosecond precision

### The Hybrid Solution

Split `dt_sec` into chunks with local offsets:
1. Divide data into N chunks (e.g., 100 TOAs per chunk)
2. For each chunk, compute `t_offset = mean(dt_sec)` in that chunk
3. Compute `dt_local = dt_sec - t_offset` (now small, ~days not years)
4. Compute phase at offset: `phase_offset = F0*t_offset + (F1/2)*t_offset²`
5. Compute local phase: `phase_local = F0*dt_local + F1*t_offset*dt_local + (F1/2)*dt_local²`
6. Total phase = `phase_offset + phase_local`

The key insight: `dt_local` is small (~10⁵ seconds per chunk), so `F0*dt_local` preserves more precision.

## Your Task

Create a Jupyter notebook that compares three **FITTING** methods for J1909-3744:

**Goal**: Start with intentionally wrong F0/F1 parameters and fit them. Compare which method's fitted parameters are closest to the longdouble "ground truth".

### Method 1: Longdouble Reference (Ground Truth)
Replicate the `jug-fit` logic but replace JAX with numpy longdouble everywhere:
- Use `np.longdouble` for ALL calculations: dt_sec, phase, derivatives, WLS solve
- Compute design matrix with longdouble derivatives
- Solve WLS in longdouble precision
- This is the "ground truth" - slow but maximally precise

### Method 2: Current JUG Implementation  
Use the exact logic from `jug-fit` command (replicate what happens when you run):
```bash
jug-fit data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim --fit F0 F1
```
This uses JAX float64 throughout the fitting pipeline.

### Method 3: Hybrid Chunked Method
Use the same fitting logic as Method 2, but replace the phase calculation with the chunked approach:
```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def compute_phase_hybrid(dt_sec, F0, F1, chunk_size=100):
    """Compute phase using chunked hybrid method."""
    n_toas = len(dt_sec)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    
    phase = jnp.zeros(n_toas)
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_toas)
        
        dt_chunk = dt_sec[start:end]
        t_offset = jnp.mean(dt_chunk)  # Use longdouble for this!
        dt_local = dt_chunk - t_offset
        
        # Phase at offset (compute with high precision)
        phase_offset = F0 * t_offset + 0.5 * F1 * t_offset**2
        
        # Local phase (dt_local is small, float64 is fine)
        phase_local = F0 * dt_local + F1 * t_offset * dt_local + 0.5 * F1 * dt_local**2
        
        phase = phase.at[start:end].set(phase_offset + phase_local)
    
    return phase
```

**Critical fix**: The `t_offset` and `phase_offset` calculations should use `np.longdouble`, then convert to float64 for the JAX operations on `dt_local`.

## Data Files

- **Par file (WRONG initial values)**: `data/pulsars/J1909-3744_tdb_wrong.par`
- **Tim file**: `data/pulsars/J1909-3744.tim`
- **Clock files**: `data/clock/`

The par file has intentionally incorrect F0/F1 values for testing fitting convergence.
- ~10408 TOAs spanning MJD 58526 to 60837 (~6.3 years)

Expected post-fit weighted RMS residuals: **~0.4 microseconds** (when converged)

## Expected Output

### Plots
1. **Three residual time series** (3 panels or overlaid):
   - Longdouble residuals vs MJD
   - JUG residuals vs MJD  
   - Hybrid residuals vs MJD
   - All should look essentially identical, showing ~0.4 μs scatter

2. **Difference plots** (2 panels):
   - (Longdouble - JUG) vs MJD → shows JUG's precision loss
   - (Longdouble - Hybrid) vs MJD → should be much smaller

### Metrics to Report
- Weighted RMS of each method (should all be ~0.4 μs)
- RMS of (Longdouble - JUG) difference
- RMS of (Longdouble - Hybrid) difference
- Execution time for JUG vs Hybrid

### Expected Results
- JUG residuals should match longdouble to ~0.01-0.1 μs RMS
- Hybrid should match longdouble to ~0.001-0.01 μs RMS (10x better)
- Hybrid should be as fast or faster than JUG (JAX JIT)

## Key Code References

### How JUG computes residuals (simplified)
From `jug/residuals/simple_calculator.py` lines 478-500:
```python
# Spin parameters (high precision)
F0 = get_longdouble(params, 'F0')
F1 = get_longdouble(params, 'F1', default=0.0)
PEPOCH = get_longdouble(params, 'PEPOCH')

# Pre-compute coefficients
F1_half = F1 / np.longdouble(2.0)
PEPOCH_sec = PEPOCH * np.longdouble(SECS_PER_DAY)

# Time at emission (TDB - all delays)
tdb_mjd_ld = np.array(tdb_mjd, dtype=np.longdouble)
tdb_sec = tdb_mjd_ld * np.longdouble(SECS_PER_DAY)
dt_sec = tdb_sec - PEPOCH_sec - delay_sec

# Compute phase using Horner's method
phase = dt_sec * (F0 + dt_sec * (F1_half + dt_sec * F2_sixth))
```

### Getting dt_sec for your comparisons
```python
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble
from jug.utils.constants import SECS_PER_DAY
import numpy as np

# Get JUG result which includes dt_sec
result = compute_residuals_simple(par_file, tim_file, verbose=False)
dt_sec_f64 = result['dt_sec']  # float64 version
tdb_mjd = result['tdb_mjd']

# For longdouble reference, recompute dt_sec:
params = parse_par_file(par_file)
PEPOCH = get_longdouble(params, 'PEPOCH')
PEPOCH_sec = PEPOCH * np.longdouble(SECS_PER_DAY)
total_delay_ld = np.array(result['total_delay_sec'], dtype=np.longdouble)
tdb_mjd_ld = np.array(tdb_mjd, dtype=np.longdouble)
tdb_sec_ld = tdb_mjd_ld * np.longdouble(SECS_PER_DAY)
dt_sec_ld = tdb_sec_ld - PEPOCH_sec - total_delay_ld
```

## Mathematical Derivation of Hybrid Method

The Taylor expansion for phase is:
```
φ(t) = F0·t + (F1/2)·t² + (F2/6)·t³ + ...
```

For chunked computation with offset `t₀`:
```
t = t₀ + δt  (where δt is small)

φ(t) = F0·(t₀+δt) + (F1/2)·(t₀+δt)²
     = F0·t₀ + F0·δt + (F1/2)·(t₀² + 2t₀δt + δt²)
     = [F0·t₀ + (F1/2)·t₀²] + [F0·δt + F1·t₀·δt + (F1/2)·δt²]
     = φ_offset + φ_local
```

Where:
- `φ_offset = F0·t₀ + (F1/2)·t₀²` computed in longdouble (t₀ is large)
- `φ_local = F0·δt + F1·t₀·δt + (F1/2)·δt²` computed in float64 (δt is small)

The key is that `δt` is ~10⁵ seconds (within a chunk) vs `t₀` being ~10⁸ seconds.

## Important Notes

1. **Force JAX to use CPU**: Add at the start of your notebook:
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   os.environ['JAX_PLATFORMS'] = 'cpu'
   ```

2. **Ensure float64**: 
   ```python
   import jax
   jax.config.update("jax_enable_x64", True)
   ```

3. **The t_offset must be computed in longdouble** for the hybrid method to work correctly. Otherwise you lose precision when computing phase_offset.

4. **Verify the weighted RMS is ~0.4 μs** - if it's way off, something is wrong with the residual calculation.

5. **Use the WRONG par file**: `J1909-3744_tdb_wrong.par` - this has intentionally incorrect F0/F1 to test fitting convergence. The goal is to see which method fits to the correct values.
