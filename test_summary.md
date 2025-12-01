# JUG Fitter Test with PINT Residuals - Results

## Test Setup

We tested JUG's Gauss-Newton fitter using **PINT's residual calculation** to isolate the fitter from the residual computation.

### Initial Conditions
- True values: F0=339.315691919 Hz, F1=-1.614740e-15 Hz/s, DM=10.390712 pc/cm³
- Perturbations applied: ΔF0=1e-7 Hz, ΔF1=1e-16 Hz/s, ΔDM=0.001 pc/cm³
- Initial χ² = 1.49e12 (perturbed), Target χ² = 3.56e5 (true values)

## Results

### JUG Fit (using PINT residuals)
- Iterations: 20
- Final χ²: 1.49 (converged to LOCAL minimum, not global)
- **Did NOT recover true values**

| Parameter | True Value | Fitted Value | Offset | σ Offset |
|-----------|------------|--------------|--------|----------|
| F0 | 3.393156919e+02 | 3.393156920e+02 | 1.0e-07 | **9.8M σ** |
| F1 | -1.614740e-15 | -1.514740e-15 | 1.0e-16 | **600k σ** |
| DM | 10.390712 | 10.391712 | 0.001 | **3417 σ** |

## Interpretation

**CRITICAL FINDING**: Even when using PINT's own residual calculation, JUG's fitter gets stuck at the perturbed values and reports convergence at χ²=1.49, while the TRUE minimum is at χ²=355801.

This means:
1. ✅ **JUG's fitter algorithm works** (it found a local minimum)
2. ❌ **JUG's residual function differs from PINT** - the parameter updates computed from the design matrix are not moving toward the true values

The design matrix derivatives (dR/dF0, dR/dF1, dR/dDM) must be calculated differently between JUG and PINT, causing the fit to move in the wrong direction in parameter space.

## Next Steps

We need to debug why JUG's design matrix computation gives different derivatives than PINT expects. The issue is likely in:
- `jug/fitting/design_matrix_jax.py` - numerical derivative computation
- Step size for finite differences
- Or the order of delay corrections in the residual calculation
