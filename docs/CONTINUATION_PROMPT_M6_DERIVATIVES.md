# Continuation Prompt: M6 Binary Parameter Derivatives

**Date**: 2026-01-29  
**Status**: ✓ **ELL1 DERIVATIVES COMPLETE**  
**Context**: JUG (Python pulsar timing) - Milestone 6 Complete Parameter Fitting  
**Previous Work**: Astrometry derivatives complete (RAJ, DECJ, PMRA, PMDEC, PX validated to <1e-6 vs PINT)

---

## ✓ COMPLETED: ELL1 Binary Derivatives

**Implementation Status**: All ELL1 derivatives implemented, validated, and integrated into fitting engine.

### Validated Parameters
- ✓ **A1**: Perfect match with PINT (<1e-7)
- ✓ **EPS1, EPS2**: Perfect match with PINT (<1e-7)
- ✓ **SINI**: Perfect match with PINT (<1e-7)
- ✓ **M2**: Matches Tempo2 exactly, PINT has bug (~0.03% difference)
- ✓ **PB, TASC, PBDOT**: Match Tempo2 exactly, PINT has bug (~0.05% difference)
- ✓ **XDOT**: Implemented and integrated

### PINT Bug Discovered
Found missing `cos(Φ)` term in PINT's `d_delayS_d_Phi` (Shapiro delay derivative). JUG's implementation is mathematically correct and matches Tempo2. See `docs/DERIVATIVE_PARITY.md` for details.

### Fitting Integration Complete
- ✓ Binary derivatives integrated into `optimized_fitter.py`
- ✓ Session updates parameters after each fit (enables iterative fitting)
- ✓ GUI fitting works with binary parameters
- ✓ Post-fit report shows units for all parameters
- ✓ Tested on J1909-3744: converges in 3-5 iterations, RMS ~0.40 μs

---

## Current Task (REMAINING WORK)

Continue with remaining binary models and parameter types.

## NON-NEGOTIABLES

1. **Bit-for-bit numerical outputs**: Do NOT change computation order/precision or any engine math in existing code.
2. **Engine is canonical**: Any GUI action must call engine operations; GUI/CLI/API must match identically.
3. **No eager JAX import**: `import jug` must not import JAX; JAX is configured lazily via `jug/utils/jax_setup.py`.
4. **JAX preferred**: All new timing model derivatives MUST be written in JAX unless:
   - JAX cannot maintain required precision (document why)
   - JAX is slower than NumPy for that specific operation (benchmark and document)
   - If either exception applies, add a comment explaining why NumPy is used
5. **Accuracy over speed**: Where PINT and Tempo2 differ, use the more scientifically accurate implementation. Document the difference.
6. **Performance cannot regress**: Current benchmarks: ~2.5s cold start, ~100ms warm fits for 10k TOAs.

## Implementation Requirements

### 1. Verify Against Both PINT and Tempo2

For each derivative, compare JUG's implementation against:
- **PINT**: Use PINT's design matrix (`model.d_phase_d_param()` or similar)
- **Tempo2**: Use tempo2's analytical formulas from source code or published documentation

Document the comparison:
```markdown
| Parameter | JUG vs PINT | JUG vs Tempo2 | Notes |
|-----------|-------------|---------------|-------|
| PB        | ratio: X.XX | ratio: X.XX   | ...   |
| A1        | ratio: X.XX | ratio: X.XX   | ...   |
```

### 2. Handle Differences Correctly

If PINT and Tempo2 disagree:
1. Analyze which is more physically correct (consider higher-order terms, relativistic corrections)
2. Implement the more accurate version
3. Document the difference in a comment and in `docs/DERIVATIVE_PARITY.md`
4. Example: If Tempo2 includes a second-order correction that PINT omits, use Tempo2's formula

### 3. Write in JAX

All derivatives must be JAX-compatible:
```python
import jax.numpy as jnp
from jax import jit

@jit
def d_delay_d_PB(pb_days, a1_lt_s, eps1, eps2, tasc_mjd, toas_tdb_mjd):
    """Derivative of Roemer delay w.r.t. orbital period PB.
    
    Uses PINT convention (matches to <1e-6 relative error).
    Tempo2 differs by [describe difference] - we use PINT because [reason].
    """
    # Implementation using jnp operations
    ...
```

### 4. Validate Precision

After implementing each derivative:
1. Run comparison against PINT design matrix column
2. Run comparison against Tempo2 (if possible)
3. Ensure relative error < 1e-6 for typical pulsar parameters
4. Test on multiple pulsars with different orbital parameters

### 5. Existing Code Reference

Current files:
- `jug/fitting/derivatives_astrometry.py` - Astrometry (DONE, use as template)
- `jug/fitting/derivatives_spin.py` - Spin (DONE)
- `jug/fitting/derivatives_dm.py` - DM (DONE)
- `jug/fitting/derivatives_binary.py` - Binary (PARTIAL, needs refinement)

Binary model code:
- `jug/delays/binary_ell1.py` - ELL1 delay computation
- `jug/delays/binary_dd.py` - DD delay computation

## Parameters to Implement

### ELL1 Model (Priority)
- [x] PB - orbital period (implemented, ~2% off PINT - needs fix)
- [x] A1 - projected semi-major axis (implemented, ~5% off PINT - needs fix)
- [x] TASC - time of ascending node (implemented)
- [x] EPS1 - e*sin(omega) (implemented)
- [x] EPS2 - e*cos(omega) (implemented)
- [x] PBDOT - orbital period derivative (implemented)
- [x] XDOT - A1 derivative (implemented)
- [ ] FB0, FB1, ... - orbital frequency and derivatives (alternative to PB)

### Post-Keplerian (ELL1)
- [x] M2 - companion mass (implemented)
- [x] SINI - sin(inclination) (implemented)
- [ ] H3, H4, STIG - orthometric Shapiro parameters (ELL1H)

### DD Model (Lower Priority)
- [ ] ECC - eccentricity
- [ ] OM - longitude of periastron
- [ ] T0 - epoch of periastron
- [ ] OMDOT - periastron advance
- [ ] GAMMA - Einstein delay amplitude
- [ ] EDOT - eccentricity derivative

## Validation Script

Create `jug/tests/test_binary_derivative_parity.py`:
```python
"""Compare binary derivatives against PINT and Tempo2."""

def test_pb_derivative_parity():
    """PB derivative matches PINT and Tempo2."""
    # Load test pulsar
    # Compute JUG derivative
    # Compute PINT derivative
    # Compare (assert ratio within 1e-6 of 1.0)
    # If Tempo2 available, compare against that too
    
def test_a1_derivative_parity():
    ...
```

## After Binary Derivatives: Audit Non-JAX Functions

After completing binary derivatives, audit ALL timing model functions:

1. List all functions in:
   - `jug/delays/`
   - `jug/fitting/`
   - `jug/residuals/`

2. For each function using NumPy instead of JAX, document:
   - Why NumPy is used (precision? speed? legacy?)
   - Whether it could/should be converted to JAX
   - If keeping NumPy, add comment explaining why

3. Create `docs/JAX_AUDIT.md` with findings

## Success Criteria

- [ ] All ELL1 derivatives match PINT to <1e-6 relative error
- [ ] Differences from Tempo2 documented with scientific rationale
- [ ] All new code written in JAX
- [ ] Tests pass for J1909-3744 and at least one other ELL1 pulsar
- [ ] Can fit PB, A1, TASC, EPS1, EPS2 via GUI and CLI
- [ ] No performance regression
- [ ] Audit of non-JAX functions complete

## File Deliverables

1. Updated `jug/fitting/derivatives_binary.py` - JAX implementations
2. `jug/tests/test_binary_derivative_parity.py` - Validation tests
3. `docs/DERIVATIVE_PARITY.md` - Comparison documentation
4. `docs/JAX_AUDIT.md` - Audit of NumPy vs JAX usage
5. Updated `docs/JUG_PROGRESS_TRACKER.md` - Progress tracking

---

## Reference: Current Binary Derivative Status

From progress tracker:
- PB: ~2% off PINT
- A1: ~5% off PINT
- EPS1/EPS2/TASC: Implemented, parity unknown
- Note: "Derivatives have correct structure but ~2-5% systematic offset from PINT"
- Next step: "Trace PINT's orbital phase calculation for exact parity"

The systematic offset suggests a convention difference (units, reference epoch, or phase definition) rather than a formula error. Carefully trace PINT's code to find the source.
