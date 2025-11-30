# Copy-Paste Ready: Time-Domain Residual Functions

Use these two functions as drop-in replacements for your residual calculation.

## Import Section

```python
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

# These constants should already be defined in your code
# SECS_PER_DAY = 86400.0
# K_DM_SEC = 4.148808e-3  # DM constant in ms / (pc/cm³)
```

## Function 1: Time-Domain Residuals

```python
@jax.jit
def residuals_time_domain(
    t_bary_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    model,  # SpinDMModel
) -> jnp.ndarray:
    """
    Compute timing residuals in the time domain (Tempo2 style).
    
    This correctly implements the Tempo2/PINT residual calculation:
    1. Compute spin phase at barycentric arrival time
    2. Extract fractional phase (nearest pulse)
    3. Convert to time residual
    
    Args:
        t_bary_mjd: Barycentric arrival times (MJD)
        freq_mhz: Observation frequencies (MHz)
        model: SpinDMModel with F0, F1, F2, PEPOCH, etc.
    
    Returns:
        Residuals in seconds (time domain)
    
    Usage:
        res_sec = residuals_time_domain(t_bary_jax, freq_jax, model)
        res_us = np.array(res_sec) * 1e6  # Convert to microseconds
        res_centered = res_us - np.mean(res_us)  # Remove offset
    """
    
    # Time since reference epoch (PEPOCH) in seconds
    dt_sec = (t_bary_mjd - model.tref_mjd) * SECS_PER_DAY
    
    # Compute spin phase at barycentric arrival time
    # phase = F0*t + (F1/2)*t^2 + (F2/6)*t^3
    phase = (model.f0 * dt_sec + 
             0.5 * model.f1 * dt_sec**2 + 
             (1.0/6.0) * model.f2 * dt_sec**3)
    
    # Extract fractional phase (position relative to nearest pulse)
    # Wrap to (-0.5, +0.5) to represent phase offset from nearest integer
    fractional_phase = jnp.mod(phase + 0.5, 1.0) - 0.5
    
    # Convert fractional phase to time residual
    # residual [seconds] = fractional_phase [cycles] / F0 [cycles/second]
    residuals_sec = fractional_phase / model.f0
    
    return residuals_sec
```

## Function 2: DM-Corrected Time-Domain Residuals

```python
@jax.jit
def residuals_time_domain_dm_corrected(
    t_bary_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    model,  # SpinDMModel
    dm_value: jnp.ndarray,  # Scalar DM value
) -> jnp.ndarray:
    """
    Compute DM-corrected timing residuals (frequency-independent).
    
    Removes the frequency dependence from dispersion measure (DM).
    Lower frequencies are delayed more by DM; this function corrects for that.
    
    Args:
        t_bary_mjd: Barycentric arrival times (MJD)
        freq_mhz: Observation frequencies (MHz)
        model: SpinDMModel
        dm_value: DM value to use for correction
    
    Returns:
        DM-corrected residuals in seconds
    
    Usage:
        dm_jax = jnp.array([10.39])
        res_sec = residuals_time_domain_dm_corrected(
            t_bary_jax, freq_jax, model, dm_jax[0]
        )
    """
    
    # Get base time-domain residuals
    residuals_sec = residuals_time_domain(t_bary_mjd, freq_mhz, model)
    
    # DM dispersion delay correction
    # DM delay = DM * (1/f^2) * K_DM
    # K_DM ≈ 4.148808 ms / (pc/cm³)
    dm_correction_sec = dm_value * (freq_mhz**(-2)) * K_DM_SEC
    
    # Subtract DM correction to get frequency-independent residuals
    residuals_corrected_sec = residuals_sec - dm_correction_sec
    
    return residuals_corrected_sec
```

## Complete Minimal Example

```python
import jax
import jax.numpy as jnp
import numpy as np

# Constants
SECS_PER_DAY = 86400.0
K_DM_SEC = 4.148808e-3

# [Include function definitions above here]

# Your data
t_bary_mjd = np.array([59000.0, 59001.0, 59002.0, ...])  # Tempo2 barycentric times
freq_mhz = np.array([1400.0, 1400.0, 1400.0, ...])       # Frequencies
model = ...  # Your SpinDMModel instance

# Convert to JAX (outside the loop)
t_bary_jax = jnp.array(t_bary_mjd)
freq_jax = jnp.array(freq_mhz)

# Compute residuals
residuals_sec = residuals_time_domain(t_bary_jax, freq_jax, model)
residuals_us = np.array(residuals_sec) * 1e6

# Remove mean
residuals_centered = residuals_us - np.mean(residuals_us)

print(f"RMS: {np.sqrt(np.mean(residuals_centered**2)):.6f} μs")
print(f"Min: {np.min(residuals_centered):.6f} μs")
print(f"Max: {np.max(residuals_centered):.6f} μs")
```

## For Parameter Fitting

```python
from scipy.optimize import minimize

def likelihood_function(params, t_bary_jax, freq_jax, model_template):
    """
    Likelihood function for parameter fitting.
    
    params: [f0_adjustment, f1_adjustment, dm_adjustment]
    """
    df0, df1, dm = params
    
    # Update model with new parameters
    model_new = SpinDMModel(
        f0=model_template.f0 + df0,
        f1=model_template.f1 + df1,
        f2=model_template.f2,
        dm=model_template.dm + dm,
        dm1=model_template.dm1,
        dm2=model_template.dm2,
        dm_coeffs=model_template.dm_coeffs,
        dm_factorials=model_template.dm_factorials,
        dm_epoch_mjd=model_template.dm_epoch_mjd,
        tref_mjd=model_template.tref_mjd,
        phase_ref_mjd=model_template.phase_ref_mjd,
        phase_offset_cycles=model_template.phase_offset_cycles,
    )
    
    # Compute residuals
    res_sec = residuals_time_domain(t_bary_jax, freq_jax, model_new)
    res_us = jnp.array(res_sec) * 1e6
    
    # Chi-square (minimize this)
    chi2 = jnp.sum((res_us - jnp.mean(res_us))**2)
    
    return chi2

# Initial guess (usually zeros, indicating no adjustment needed)
x0 = [0.0, 0.0, 0.0]

# Fit
result = minimize(
    likelihood_function,
    x0=x0,
    args=(t_bary_jax, freq_jax, model),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

# Fitted parameters
f0_fitted = model.f0 + result.x[0]
f1_fitted = model.f1 + result.x[1]
dm_fitted = model.dm + result.x[2]

print(f"Fitted F0: {f0_fitted:.15f}")
print(f"Fitted F1: {f1_fitted:.15e}")
print(f"Fitted DM: {dm_fitted:.6f}")
print(f"Chi-square: {result.fun:.6e}")
```

## JAX Gradient Support

These functions are fully differentiable. You can compute gradients:

```python
# Gradient with respect to model parameters
grad_fn = jax.grad(likelihood_function, argnums=0)
gradients = grad_fn([0.0, 0.0, 0.0], t_bary_jax, freq_jax, model)

print(f"Gradient w.r.t. F0: {gradients[0]}")
print(f"Gradient w.r.t. F1: {gradients[1]}")
print(f"Gradient w.r.t. DM: {gradients[2]}")
```

## No Dependencies

These functions require only:
- `jax` and `jax.numpy`
- Your `SpinDMModel` class (already in your code)
- Constants `SECS_PER_DAY` and `K_DM_SEC` (already defined)

**No PINT, no external dependencies.**

---

**For complete documentation and examples, see:**
- `TIME_DOMAIN_RESIDUALS.md` - Detailed guide
- `FINAL_DIAGNOSIS.md` - Background and explanation
- Notebook cells `#VSC-44c0f7ea` and `#VSC-43727334` - Full implementation
