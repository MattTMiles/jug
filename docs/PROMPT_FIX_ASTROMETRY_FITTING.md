# Prompt: Fix Astrometry Parameter Fitting Divergence in JUG

## Executive Summary

JUG (JAX-based pulsar timing software) has a critical bug where fitting astrometric parameters causes divergence when fitting multiple times. A previous Claude investigated and identified the root cause but did not implement a fix. This prompt provides the complete diagnosis and asks you to research how PINT and Tempo2 handle this correctly, then implement a proper solution.

## The Problem

When fitting all timing parameters (spin, DM, astrometry, binary) in JUG:
- **First fit**: Reports RMS = 0.40 μs (but TRUE RMS is 0.84 μs)
- **Second fit**: Reports RMS = 0.42 μs (but TRUE RMS is 27 μs!)
- **Third fit**: Complete NaN collapse

The fitter thinks it's improving but is actually making things worse with each fit.

## Root Cause Identified

The issue is a **fundamental mismatch between JUG's linear approximation and the true non-linear model**.

### How JUG Currently Works

1. **Caching**: At the start of fitting, JUG computes `dt_sec = tdb - PEPOCH - total_delay` and caches it
2. **Iterations**: During WLS iterations, JUG updates `dt_sec` for DM and binary delay changes, but **NOT** for astrometric delay changes
3. **Linear approximation**: The design matrix contains `d(delay)/d(param)`, and the WLS solver finds parameter updates assuming residuals change linearly with parameters
4. **Problem**: When astrometry parameters (RAJ, DECJ, PMRA, PMDEC, PX) change, the astrometric delay changes too, but this change is NOT reflected in the residuals used during iterations

### Why This Causes Divergence

The WLS solver optimizes against residuals computed with the **OLD** astrometric delay. When we update the par file and reload, the **NEW** astrometric delay is computed, giving completely different residuals. The "optimal" parameters found by WLS are actually sub-optimal for the true model.

### What Happens When We Enable Astrometric Delay Updates

When I tried updating the astrometric delay during iterations (like we do for DM and binary), the fit diverged even faster:
- Iteration 1: RMS = 0.40 μs
- Iteration 2: RMS = 0.41 μs  
- Iteration 3: RMS = 2.2 μs
- Iteration 4: RMS = 120 μs
- Iteration 5: RMS = 841 μs

This is because the astrometric delay change creates a feedback loop that amplifies errors. Small position changes cause large delay changes that aren't well-predicted by the first-order Taylor expansion (the design matrix).

## How PINT Handles This

PINT uses a fundamentally different approach that we need to understand and potentially adopt:

### PINT's ModelState Architecture

```python
class ModelState:
    @cached_property
    def resids(self):
        # Recomputes FULL model from scratch!
        return self.fitter.make_resids(self.model)
    
    @cached_property  
    def step(self):
        # Computes WLS solution using current residuals
        M, params, units = self.model.designmatrix(...)
        dpars, ... = fit_wls_svd(residuals, sigma, M, ...)
        return dpars

    def take_step(self, step, lambda_=1):
        # Creates NEW ModelState with updated model
        return WLSState(self.fitter, self.take_step_model(step, lambda_))
```

### PINT's Iteration Loop

```python
for i in range(maxiter):
    step = current_state.step  # WLS solution
    lambda_ = 1
    while True:
        # Create new state with updated parameters
        new_state = current_state.take_step(step, lambda_)
        
        # Check if chi2 ACTUALLY improved (using FULL model!)
        chi2_decrease = current_state.chi2 - new_state.chi2
        
        if chi2_decrease < -max_chi2_increase:
            # Step made things worse - reduce step size
            lambda_ /= 2
            continue
        
        # Accept the step
        current_state = new_state
        break
```

### Key Insight

PINT **recomputes the full model after each parameter update** to verify the chi2 actually improved. If it didn't, PINT reduces the step size (line search / trust region approach). This is more expensive but ensures stability.

## Your Task

### 1. Deep Research

Study how PINT and Tempo2 handle iterative fitting with astrometry parameters:

**PINT files to examine:**
- `src/pint/fitter.py` - `ModelState`, `WLSState`, `DownhillFitter._fit_toas()`
- `src/pint/models/timing_model.py` - `designmatrix()`, `d_phase_d_param()`
- `src/pint/models/astrometry.py` - `d_delay_astrometry_d_RAJ()`, etc.

**Questions to answer:**
1. Does PINT recompute the full model after each iteration? (I believe yes)
2. How does PINT's `d_phase_d_param` relate to the design matrix?
3. Does PINT use any regularization, damping, or trust region methods?
4. How does Tempo2 handle this? (Check Tempo2 source if accessible)

### 2. Implement a Fix

Options to consider:

**Option A: Full Model Recomputation (PINT-style)**
- After each WLS step, recompute residuals using the full model
- Verify chi2 improved; if not, reduce step size
- Pro: Guaranteed stability
- Con: Slower (need to recompute astrometric delay each iteration)

**Option B: Hybrid Approach**
- Use linear approximation for most iterations
- Periodically verify with full model
- Reduce step size if full model chi2 didn't improve

**Option C: Better Linear Approximation**
- Include second-order terms in the approximation
- Use the Hessian to capture curvature
- More complex but potentially faster

### 3. Requirements

- **MUST use JAX** for all timing computations where possible
- **Accuracy is paramount**: Maintain nanosecond (1e-9 s) precision
- The fix must allow users to click "Fit" multiple times without divergence
- Final RMS should match PINT/Tempo2 (~0.40 μs for this pulsar)

## Test Case

```bash
cd /home/mattm/soft/JUG
python -c "
from pathlib import Path
from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, format_ra, format_dec
import tempfile
import numpy as np

fit_params = ['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 
              'DM', 'DM1', 'DM2',
              'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT']

par_file = Path('data/pulsars/J1909-3744_tdb.par')
tim_file = Path('data/pulsars/J1909-3744.tim')
temp_dir = tempfile.mkdtemp()

for fit_num in range(5):
    result_pre = compute_residuals_simple(str(par_file), str(tim_file), verbose=False)
    initial_rms = np.sqrt(np.mean(result_pre['residuals_us']**2))
    
    result = fit_parameters_optimized(
        par_file=par_file, tim_file=tim_file,
        fit_params=fit_params, max_iter=10, verbose=False
    )
    
    # Write updated par file
    params = parse_par_file(par_file)
    for p, v in result['final_params'].items():
        if p == 'RAJ': params[p] = format_ra(v)
        elif p == 'DECJ': params[p] = format_dec(v)
        else: params[p] = v
    
    next_par = Path(temp_dir) / f'fit{fit_num+1}.par'
    with open(next_par, 'w') as f:
        for k, v in params.items(): f.write(f'{k} {v}\n')
    
    # Compute TRUE final RMS
    result_post = compute_residuals_simple(str(next_par), str(tim_file), verbose=False)
    true_final_rms = np.sqrt(np.mean(result_post['residuals_us']**2))
    
    print(f'Fit {fit_num+1}: initial={initial_rms:.3f}, reported={result[\"final_rms\"]:.3f}, TRUE={true_final_rms:.3f} us')
    
    if true_final_rms > initial_rms * 2:
        print('  DIVERGED!')
        break
    par_file = next_par

import shutil
shutil.rmtree(temp_dir)
"
```

**Expected output after fix:**
```
Fit 1: initial=0.820, reported=0.404, TRUE=0.404 us
Fit 2: initial=0.404, reported=0.404, TRUE=0.404 us
Fit 3: initial=0.404, reported=0.404, TRUE=0.404 us
Fit 4: initial=0.404, reported=0.404, TRUE=0.404 us
Fit 5: initial=0.404, reported=0.404, TRUE=0.404 us
```

## Key Files in JUG

- `/home/mattm/soft/JUG/jug/fitting/optimized_fitter.py` - Main fitter, `_run_general_fit_iterations()`
- `/home/mattm/soft/JUG/jug/fitting/derivatives_astrometry.py` - Astrometric derivatives
- `/home/mattm/soft/JUG/jug/fitting/wls_fitter.py` - WLS solver
- `/home/mattm/soft/JUG/jug/residuals/simple_calculator.py` - Full model residual computation

## Summary of Previous Investigation

1. **Derivatives are correct**: JUG's `d(delay)/d(RAJ)` matches PINT exactly (ratio = 1.000)
2. **Design matrix convention is correct**: Both use `+d(delay)/d(param)` 
3. **The issue is NOT in the derivatives themselves**
4. **The issue IS in not recomputing the full model after parameter updates**

The fix must ensure that after parameters are updated, the residuals reflect the TRUE model with the updated astrometric delay, not a stale cached version.
