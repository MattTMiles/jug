# Fitting Algorithm Analysis

## Problem Statement

When fitting pulsar timing parameters, JUG's Gauss-Newton fitter produces results that differ from PINT by ~3.5σ for F0 (and smaller amounts for other parameters). This is unacceptable for precision timing.

## Tests Performed

### Test 1: JAX vs Baseline Residuals
- **Result**: JAX initially had precision issues, fixed by ensuring float64 everywhere
- **Conclusion**: Residual computation in JAX now matches baseline (numpy) to high precision

### Test 2: Baseline vs PINT Residuals  
- **Result**: Very small differences (~0.02 μs RMS), essentially negligible
- **Conclusion**: JUG residuals match PINT residuals very closely

### Test 3: Numpy Gauss-Newton with PINT Residuals
- **Setup**: Used our numpy Gauss-Newton fitter but with PINT's residual calculation
- **Result**: Still got 3.5σ difference for F0, 0.5σ for others
- **Conclusion**: The problem is NOT the residual calculation - it's the fitting algorithm itself!

## Root Cause

Our Gauss-Newton implementation is too simple:
- No damping or trust regions → can overshoot and oscillate
- No parameter scaling → F0 (~100 Hz) and DM (~10 pc/cm³) have very different scales
- No step size limiting → unstable convergence
- No adaptive strategy → stuck in local minimum or diverges

## PINT's Approach

PINT uses **Powell's dogleg method** (`scipy.optimize.least_squares`) with:
1. **Trust region**: Limits step size to prevent overshooting
2. **Damping**: Interpolates between gradient descent (safe, slow) and Gauss-Newton (fast, risky)
3. **Parameter scaling**: Normalizes parameters to similar magnitudes
4. **Adaptive damping**: Increases damping if step is rejected, decreases if accepted
5. **Multiple termination criteria**: Chi-squared change, parameter change, gradient norm

## Solutions

### Option 1: Levenberg-Marquardt (Recommended)
- Gold standard for nonlinear least squares
- Adds damping parameter λ to normal equations: (J^T W J + λ diag(J^T W J)) δ = J^T W r
- When λ large → gradient descent (stable)
- When λ small → Gauss-Newton (fast)
- Adaptive λ based on whether step improves chi²

### Option 2: Use scipy.optimize.least_squares
- Wraps PINT's exact algorithm (dogleg/trust-region)
- Proven to work
- Less control, but reliable

### Option 3: Implement Powell's Dogleg
- More sophisticated than L-M
- What PINT uses
- Most work to implement

## Recommendation

1. **Implement Levenberg-Marquardt properly** with:
   - Adaptive damping (increase if step rejected, decrease if accepted)
   - Proper termination criteria
   - JAX-compatible implementation

2. **If L-M still doesn't match PINT exactly**, switch to `scipy.optimize.least_squares` to guarantee identical convergence

## Next Steps

1. Fix L-M implementation (currently has interface issues with JUG's data structures)
2. Test on synthetic data (known true parameters)
3. Compare L-M vs PINT on real data (J1909-3744)
4. If within 1σ → success!
5. If not → switch to scipy.optimize.least_squares

