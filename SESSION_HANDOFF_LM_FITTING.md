# Session Handoff: Levenberg-Marquardt Fitter Implementation

**Date**: 2025-12-01  
**Status**: Levenberg-Marquardt fitter implemented but not yet tested  
**Goal**: Complete implementation and validation of LM fitter for JAX-based pulsar timing

---

## Context & Background

### What We've Accomplished
1. ✅ Fixed JAX residual calculation precision issues (float64 enforcement)
2. ✅ Confirmed baseline (non-JAX) residuals match PINT to ~0.02 μs RMS
3. ✅ Identified that simple Gauss-Newton fitter has convergence issues (3.5 sigma off)
4. ✅ Implemented Levenberg-Marquardt fitter with damping and trust regions

### The Problem We're Solving
- Simple Gauss-Newton doesn't converge properly (gets stuck 3.5σ from reference values)
- Need robust nonlinear least squares that handles:
  - Parameter scaling (F0 ~ 100 Hz, DM ~ 10 pc/cm³)
  - Step size limiting (prevent overshooting)
  - Trust regions (adaptive damping)

### Why Levenberg-Marquardt?
- Gold standard for nonlinear least squares
- Interpolates between gradient descent (far from minimum) and Gauss-Newton (near minimum)
- Uses adaptive damping parameter λ
- Much more robust than simple Gauss-Newton

---

## Key Files & Locations

### Implementation Files
1. **`/home/mattm/soft/JUG/jug/fitting/levenberg_marquardt.py`**
   - Main LM fitter implementation
   - JAX-compatible with @jit decoration
   - Functions: `levenberg_marquardt_step()`, `fit_model()`

2. **`/home/mattm/soft/JUG/jug/residuals/jax_calculator.py`**
   - JAX residual calculation (corrected for precision)
   - Must maintain float64 throughout
   - Function: `residuals_jax()`

3. **`/home/mattm/soft/JUG/jug/residuals/simple_calculator.py`**
   - Baseline (non-JAX) residual calculator
   - Matches PINT to ~0.02 μs RMS
   - Function: `compute_residuals()`

### Test Files
1. **`/home/mattm/soft/JUG/test_jug_fitter_with_pint_residuals.py`**
   - Tests numpy Gauss-Newton with PINT residuals
   - Shows 3.5σ convergence issue exists even with PINT residuals

2. **`/home/mattm/soft/JUG/test_wls_vs_pint.py`**
   - Attempted PINT WLS clone (didn't work)

### Documentation Files
1. **`/home/mattm/soft/JUG/M2_JAX_FITTING_STATUS.md`**
   - Current status document
   - Details all precision fixes and convergence issues

2. **`/home/mattm/soft/JUG/JUG_PROGRESS_TRACKER.md`**
   - Overall milestone tracker
   - Currently on Milestone 2: Fitting algorithms

---

## What Needs to Be Done

### Immediate Next Steps

#### 1. Test the LM Fitter Implementation
Create test script: `/home/mattm/soft/JUG/test_lm_jug.py`

```python
# Test LM fitter with both PINT and JUG residuals
# Compare convergence to PINT's fit results
# Verify σ-level agreement
```

**Success criteria:**
- Converges within 1σ of PINT reference values
- Works with both PINT residuals and JUG residuals
- JAX version matches numpy version numerically

#### 2. Compare LM vs Gauss-Newton
- Show that LM converges where Gauss-Newton fails
- Document why (damping, trust regions, parameter scaling)

#### 3. Validate on Real Data
Use pulsar: **J1909-3744** (data in `/home/mattm/soft/JUG/data/J1909-3744.tim`)
- 10,408 TOAs
- Fit parameters: F0, F1, DM
- Compare final χ² and residual RMS to PINT

#### 4. Create Synthetic Data Test
- Generate fake TOAs with known parameters
- Perturb parameters slightly
- Verify both PINT and JUG recover true values

---

## Technical Details

### LM Algorithm Structure

```python
# Basic LM iteration
J = compute_jacobian(params)  # Design matrix
r = compute_residuals(params)  # Residuals
H = J.T @ W @ J + λ * diag(J.T @ W @ J)  # Damped Hessian
g = J.T @ W @ r  # Gradient
Δp = solve(H, -g)  # Parameter step

# Update damping based on improvement
if improvement_good:
    λ = λ / 10  # Decrease damping (move toward Gauss-Newton)
else:
    λ = λ * 10  # Increase damping (move toward gradient descent)
```

### Parameter Scaling
CRITICAL: Parameters have vastly different scales
- F0 ~ 339 Hz
- F1 ~ -1.6e-15 Hz/s  
- DM ~ 10.4 pc/cm³

**Solution**: Normalize by parameter magnitudes before solving

### Precision Requirements
- Must use `dtype=np.float64` everywhere in JAX
- JAX config: `jax.config.update('jax_enable_x64', True)`
- Timing precision: ~100 nanoseconds required

---

## Reference Data & Comparisons

### PINT Reference Values (J1909-3744)
From converged PINT fit:
```
F0: 339.31568730960866 Hz
F1: -1.6148278939812545e-15 Hz/s
DM: 10.392620781388889 pc/cm³
```

### Test Results So Far

| Fitter | Residuals | F0 Diff (σ) | F1 Diff (σ) | Status |
|--------|-----------|-------------|-------------|---------|
| Gauss-Newton (numpy) | PINT | 3.5 | 0.5 | ❌ Poor |
| Gauss-Newton (numpy) | JUG | 3.5 | 0.5 | ❌ Poor |
| LM (JAX) | JUG | ? | ? | ⏳ Untested |
| PINT WLS | PINT | 0 | 0 | ✅ Reference |

---

## Commands to Run

### 1. Test LM Fitter
```bash
cd /home/mattm/soft/JUG
python test_lm_jug.py
```

### 2. Compare Convergence
```bash
python test_jug_fitter_with_pint_residuals.py  # Gauss-Newton baseline
python test_lm_jug.py  # New LM results
```

### 3. Validate Precision
```bash
# Check residual calculation matches PINT
python compare_jug_pint_detailed.py
```

---

## Key Questions to Answer

1. **Does LM converge within 1σ?** (Required for success)
2. **Does JAX LM match numpy LM?** (Tests JAX precision)
3. **Why did Gauss-Newton fail?** (Document for future reference)
4. **Can we use simpler algorithm?** (Or is LM necessary?)

---

## Important Context from CLAUDE.md

From `/home/mattm/soft/JUG/CLAUDE.md`:
- JUG is PINT/Tempo2-free pipeline using JAX
- All timing implemented from scratch
- Precision requirement: ~100 nanosecond accuracy
- Current milestone: Fitting algorithms (Milestone 2)

---

## Links to Key Documentation

1. **Project overview**: `/home/mattm/soft/JUG/README.md`
2. **Implementation guide**: `/home/mattm/soft/JUG/JUG_implementation_guide.md`
3. **Progress tracker**: `/home/mattm/soft/JUG/JUG_PROGRESS_TRACKER.md`
4. **Current session status**: `/home/mattm/soft/JUG/M2_JAX_FITTING_STATUS.md`
5. **Design philosophy**: `/home/mattm/soft/JUG/JUG_master_design_philosophy.md`

---

## Success Criteria

✅ **Must achieve:**
- LM fitter converges within 1σ of PINT values
- Works with JAX residual calculation
- Passes on real pulsar data (J1909-3744)

🎯 **Nice to have:**
- Synthetic data recovery test
- Performance benchmarks (JAX speedup)
- Documentation of why LM > Gauss-Newton

---

## Next Session Start Command

```bash
cd /home/mattm/soft/JUG
# Read this file first:
cat SESSION_HANDOFF_LM_FITTING.md
# Then run first test:
python test_lm_jug.py
```

---

## Notes & Warnings

⚠️ **Precision is critical**: Always use float64, never float32  
⚠️ **Parameter scaling matters**: Normalize before solving  
⚠️ **Trust PINT as reference**: It's the gold standard  
⚠️ **Don't modify working baseline code**: Only touch fitting code

