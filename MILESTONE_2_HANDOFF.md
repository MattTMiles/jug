# Milestone 2 Handoff Document

**Date**: 2025-11-29
**Status**: Ready to Start
**Previous Milestone**: M1 Complete (100%)

---

## Overview

Milestone 2 implements **gradient-based fitting** using JAX autodiff and NumPyro optimization. This builds on M1's accurate residual computation to enable parameter optimization with Fisher matrix uncertainties.

## Prerequisites Met

✅ **Accurate residuals** - 0.003 μs precision vs PINT
✅ **JAX infrastructure** - All delays JIT-compiled
✅ **Fast computation** - Vectorized operations throughout
✅ **Validated implementation** - Tested on J1909-3744
✅ **CLI framework** - Ready to extend with fitting commands

## Goals for Milestone 2

### Primary Objectives

1. **Parameter Optimization**
   - Implement chi-squared minimization
   - Use JAX autodiff for gradients
   - Support parameter masking (FIT flags from .par file)

2. **Uncertainty Estimation**
   - Compute Fisher information matrix
   - Calculate parameter covariances
   - Report uncertainties on fitted parameters

3. **Fitting CLI**
   - Create `jug-fit` command
   - Support masking individual parameters
   - Output fitted .par file

### Success Criteria

- Fitted parameters match PINT within 1σ uncertainties
- Chi-squared matches PINT within 1%
- Fitting completes in <10 seconds for 10k TOAs
- CLI supports all M1 timing model parameters

---

## Implementation Plan

### Step 1: Extract Fitting Functions from Notebook

**Target Cells** (from MK7 notebook):
- Search for cells with "fisher", "optimize", "chi2"
- Look for NumPyro optimization code
- Extract parameter masking logic

**Create New Modules**:
```
jug/fitting/
  __init__.py
  chi2.py          # Chi-squared computation
  fisher.py        # Fisher matrix calculation
  optimizer.py     # NumPyro/Optax optimization
  masks.py         # Parameter masking from FIT flags
```

### Step 2: Implement Chi-Squared Function

**File**: `jug/fitting/chi2.py`

```python
import jax
import jax.numpy as jnp

@jax.jit
def chi2_residuals(params_dict, toas_data, model_template):
    """Compute chi-squared from residuals.
    
    Parameters
    ----------
    params_dict : dict
        Parameter values to fit
    toas_data : dict
        TOA data (times, freqs, errors, etc.)
    model_template : dict
        Fixed model parameters
        
    Returns
    -------
    float
        Chi-squared value
    """
    # Compute residuals with new parameters
    residuals = compute_residuals_jax(params_dict, toas_data, model_template)
    
    # Chi-squared with TOA uncertainties
    errors = toas_data['errors_us']
    chi2 = jnp.sum((residuals / errors) ** 2)
    
    return chi2
```

### Step 3: Implement Fisher Matrix

**File**: `jug/fitting/fisher.py`

```python
import jax

def compute_fisher_matrix(params_dict, toas_data, model_template):
    """Compute Fisher information matrix using JAX autodiff.
    
    The Fisher matrix is the Hessian of -log(likelihood) = χ²/2.
    
    Returns
    -------
    fisher : ndarray
        Fisher information matrix (n_params × n_params)
    param_names : list
        Parameter names in order
    """
    # Get gradient function
    grad_fn = jax.grad(chi2_residuals)
    
    # Get Hessian (Fisher matrix)
    hessian_fn = jax.jacfwd(grad_fn)
    fisher = hessian_fn(params_dict, toas_data, model_template)
    
    return fisher
```

### Step 4: Implement Optimizer

**File**: `jug/fitting/optimizer.py`

```python
import jax
import optax

def fit_parameters(initial_params, toas_data, model_template, 
                   fit_mask=None, max_iter=100):
    """Fit timing model parameters using gradient descent.
    
    Parameters
    ----------
    initial_params : dict
        Starting parameter values
    toas_data : dict
        TOA data
    model_template : dict
        Fixed parameters
    fit_mask : dict, optional
        Boolean mask for which parameters to fit
    max_iter : int
        Maximum optimization iterations
        
    Returns
    -------
    fitted_params : dict
        Optimized parameter values
    uncertainties : dict
        Parameter uncertainties from Fisher matrix
    chi2_final : float
        Final chi-squared value
    """
    # Setup optimizer (Adam or LBFGS)
    optimizer = optax.adam(learning_rate=1e-3)
    
    # Apply parameter mask
    if fit_mask is not None:
        params_to_fit = {k: v for k, v in initial_params.items() 
                        if fit_mask.get(k, True)}
    else:
        params_to_fit = initial_params
    
    # Optimization loop
    opt_state = optimizer.init(params_to_fit)
    
    for i in range(max_iter):
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(chi2_residuals)(
            params_to_fit, toas_data, model_template
        )
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params_to_fit = optax.apply_updates(params_to_fit, updates)
        
        # Check convergence
        if i > 0 and abs(loss - prev_loss) < 1e-6:
            break
        prev_loss = loss
    
    # Compute uncertainties
    fisher = compute_fisher_matrix(params_to_fit, toas_data, model_template)
    covariance = jnp.linalg.inv(fisher)
    uncertainties = {k: jnp.sqrt(covariance[i, i]) 
                    for i, k in enumerate(params_to_fit.keys())}
    
    return params_to_fit, uncertainties, loss
```

### Step 5: Create Fitting CLI

**File**: `jug/scripts/fit.py`

```python
#!/usr/bin/env python3
"""Command-line interface for fitting pulsar timing models."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Fit pulsar timing model parameters using JUG"
    )
    
    parser.add_argument("par_file", type=Path)
    parser.add_argument("tim_file", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                       help="Output .par file with fitted parameters")
    parser.add_argument("--max-iter", type=int, default=100)
    
    args = parser.parse_args()
    
    # Load data
    params = parse_par_file(args.par_file)
    toas = parse_tim_file(args.tim_file)
    
    # Extract FIT flags
    fit_mask = {k: v.get('fit', True) for k, v in params.items()}
    
    # Run fitting
    fitted_params, uncertainties, chi2 = fit_parameters(
        params, toas, fit_mask=fit_mask, max_iter=args.max_iter
    )
    
    # Print results
    print(f"Chi-squared: {chi2:.2f}")
    print("\nFitted Parameters:")
    for name, value in fitted_params.items():
        unc = uncertainties[name]
        print(f"  {name:10s} = {value:20.15f} ± {unc:.15e}")
    
    # Save output
    if args.output:
        write_par_file(args.output, fitted_params, uncertainties)
```

**Add to pyproject.toml**:
```toml
[project.scripts]
jug-fit = "jug.scripts.fit:main"
```

---

## Testing Strategy

### Test Case 1: J1909-3744 (Full Fit)

```bash
jug-fit J1909-3744.par J1909-3744.tim --output J1909-3744_fitted.par
```

**Expected**:
- Chi-squared should match PINT within 1%
- Fitted F0, F1 should match within 1σ
- Binary parameters should improve fit

### Test Case 2: Parameter Masking

Edit .par file to add FIT flags:
```
F0    173.6879458... 0 1.0e-12
F1    -1.7297e-15    1 1.0e-21
```

**Expected**:
- F0 should not change (FIT flag = 0)
- F1 should be fitted (FIT flag = 1)

### Test Case 3: Uncertainty Validation

Compare Fisher matrix uncertainties with PINT:
```bash
# PINT uncertainties
tempo2 -f J1909-3744.par J1909-3744.tim -fit

# JUG uncertainties
jug-fit J1909-3744.par J1909-3744.tim
```

**Expected**:
- Uncertainties should agree within 10%

---

## Reference Materials

### Notebook Cells to Extract

Look for these patterns in MK7 notebook:
- `def fit_timing_model(...)`
- `fisher_matrix = ...`
- `optax.adam(...)`
- `jax.grad(chi2_fn)`

### Key JAX Features to Use

1. **Automatic Differentiation**:
   ```python
   grad_fn = jax.grad(chi2_function)
   grads = grad_fn(params)
   ```

2. **Hessian (Fisher Matrix)**:
   ```python
   hessian_fn = jax.hessian(chi2_function)
   fisher = hessian_fn(params)
   ```

3. **JIT Compilation**:
   ```python
   @jax.jit
   def chi2_jitted(params):
       return compute_chi2(params)
   ```

### Optimization Libraries

**Optax** (recommended):
- `optax.adam()` - Good default optimizer
- `optax.lbfgs()` - For small parameter sets
- Supports learning rate schedules

**NumPyro** (alternative):
- `numpyro.optim.Adam()`
- Integrates with MCMC (future milestone)

---

## Common Pitfalls

### 1. Parameter Scaling

**Problem**: F0 ~ 10² Hz, F1 ~ 10⁻¹⁵ Hz/s → bad conditioning

**Solution**: Scale parameters before optimization
```python
scales = {'F0': 1e2, 'F1': 1e-15, 'F2': 1e-25}
params_scaled = {k: v / scales[k] for k, v in params.items()}
```

### 2. JAX Immutability

**Problem**: Can't modify dicts in-place inside JAX functions

**Solution**: Use functional updates
```python
# Bad
params['F0'] = new_value

# Good
params = {**params, 'F0': new_value}
```

### 3. Gradient NaNs

**Problem**: Derivatives undefined at certain points

**Solution**: Add small regularization
```python
chi2 = jnp.sum((residuals / errors) ** 2) + 1e-10
```

---

## Deliverables

When Milestone 2 is complete, you should have:

- [ ] `jug/fitting/` module with 4 files
- [ ] `jug-fit` CLI command working
- [ ] Fitted parameters match PINT within 1σ
- [ ] Fisher matrix uncertainties computed
- [ ] Tests validating on J1909-3744
- [ ] Documentation in `MILESTONE_2_COMPLETION.md`

---

## Time Estimate

- **Code extraction**: 2-3 hours
- **Testing & debugging**: 2-3 hours
- **Documentation**: 1 hour
- **Total**: 5-7 hours (~1 week part-time)

---

## Questions to Answer Before Starting

1. Does the notebook have a working fitting function?
2. Are parameter uncertainties computed with Fisher matrix?
3. Which optimizer is used (Optax, NumPyro, scipy)?
4. How are FIT flags handled in the notebook?

**Action**: Review notebook to answer these questions before coding.

---

**Ready to Start**: Yes! M1 provides solid foundation for fitting.

**Next Session**: Extract fitting code from notebook → Create `jug/fitting/` module → Test on J1909-3744

Good luck! 🚀
