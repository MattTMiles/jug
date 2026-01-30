# Prompt: Debug Astrometric + Binary Parameter Fitting Divergence in JUG

## Problem Statement

JUG (JAX-based pulsar timing software) has a critical bug where fitting astrometric parameters (RAJ, DECJ, PMRA, PMDEC, PX) together with binary parameters causes catastrophic divergence on the second fit iteration.

### Observed Behavior

1. **First fit**: Converges normally to RMS ~ 0.40 μs
2. **Second fit** (iteration 4 in test): RMS jumps to ~900 μs with massive parameter changes:
   - PMRA: -9.54 → -18.13 (90% change)
   - PX: 0.95 → 1.80 mas (89% increase)
   - M2: 0.21 M☉ → 631 M☉ (3000x increase - completely unphysical!)
   - SINI: 0.998 → 1.807 (>1, impossible!)
   - DM1: increases by 200x
   - EPS1/EPS2: blow up by orders of magnitude

3. **Third fit** (iteration 5): Complete NaN collapse

### Reproduction

Run the diagnostic script:
```bash
python diagnose_fit_divergence.py
```

This fits all parameters iteratively:
- Spin: F0, F1  
- Astrometry: RAJ, DECJ, PMRA, PMDEC, PX
- DM: DM, DM1, DM2
- Binary: PB, A1, TASC, EPS1, EPS2, M2, SINI, PBDOT

## Critical Context

### 1. JAX Requirements
- **MUST use JAX** for all timing computations
- Can only move away from JAX for:
  - External library interfaces (PINT/Tempo2 validation)
  - I/O operations
  - Simple utilities that don't need differentiation
- **Accuracy is paramount**: Must maintain nanosecond (1e-9 s) precision
- If float64 doesn't provide sufficient accuracy, **MUST NOT use JAX** for that computation

### 2. What Works
- **Binary-only fitting**: PB, A1, TASC, EPS1, EPS2, M2, SINI, PBDOT converge stably
- **Spin + DM fitting**: F0, F1, DM, DM1, DM2 are stable
- **Single fit** of all parameters works (first iteration succeeds)

### 3. What Fails
- **Iterative fitting** of astrometry + binary parameters together
- Parameters diverge specifically when astrometric and binary are fitted simultaneously across iterations

## Recent Changes

Astrometric derivatives were just implemented in `/home/mattm/soft/JUG/jug/fitting/optimized_fitter.py`:

```python
def _compute_astrometric_derivatives_jax(...)
```

This computes derivatives for RAJ, DECJ, PMRA, PMDEC, PX using:
- JAX automatic differentiation via `jax.grad()`
- SSB observer position (`ssb_obs_pos_ls`)
- Proper motion and parallax corrections

## Investigation Tasks

1. **Check design matrix conditioning**:
   - Are astrometric and binary columns nearly colinear?
   - What is the condition number when fitting all parameters?
   - Are there numerical precision issues in the matrix?

2. **Verify astrometric derivatives**:
   - Compare JUG's astrometric derivatives against PINT
   - Check if units are correct (radians vs degrees, etc.)
   - Verify the sign conventions
   - Are we properly handling the correlation between position and binary parameters?

3. **Examine parameter correlation**:
   - Which parameter pairs have high correlation?
   - Is there a degenerate parameter combination?
   - Check the covariance matrix structure

4. **Numerical stability**:
   - Is float64 sufficient for these derivatives?
   - Are there catastrophic cancellations?
   - Check if normalization/scaling would help

5. **Update mechanism**:
   - Is the parameter update step size appropriate?
   - Should we use damped least squares (Levenberg-Marquardt)?
   - Is the conversion between radians (fitting) and HMS/DMS (storage) correct?

## Files to Examine

- `/home/mattm/soft/JUG/jug/fitting/optimized_fitter.py` - Main fitter and derivative computation
- `/home/mattm/soft/JUG/jug/astrometry/astrometry.py` - Astrometric corrections
- `/home/mattm/soft/JUG/jug/io/par_reader.py` - RAJ/DECJ format conversion
- `/home/mattm/soft/JUG/diagnose_fit_divergence.py` - Reproduction script

## Expected Outcome

After fixing, iterative fitting of all parameters should:
- Converge to RMS ~ 0.40 μs (matching Tempo2/PINT)
- Maintain parameter stability across iterations
- Keep parameters physically reasonable (SINI ≤ 1, M2 ~ 0.2 M☉, etc.)
- Allow users to click "Fit" multiple times without divergence

## Test Data

- Par file: `/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par`
- Tim file: `/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim`
- Pulsar: J1909-3744 (millisecond pulsar in binary with white dwarf companion)

This is a well-characterized system with excellent data quality, so fitting should be very stable.
