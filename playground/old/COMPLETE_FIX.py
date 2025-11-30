"""
Complete fix for JUG residuals to match Tempo2

Based on investigation of residual_maker_playground.ipynb
"""

import jax
import jax.numpy as jnp
import numpy as np

# ==================================================
# FIX 1: Ensure residuals are computed at the correct time
# ==================================================

# The residual function should use INFINITE-FREQUENCY barycentric time
# This is t_em_mjd (emission time) MINUS the DM delay

# In cell 233, there's code to compute t_inf:
# t_inf = t_em - dm_delay/SECS_PER_DAY

# Make sure this is used consistently

# ==================================================
# FIX 2: Verify phase_offset_cycles is being used
# ==================================================

# The residual function residuals_seconds_at_topocentric_time() should be:

@jax.jit
def residuals_seconds(t_mjd: jnp.ndarray, model):
    """
    Compute timing residuals in seconds.

    Args:
        t_mjd: Times at which to evaluate residuals (should be infinite-frequency times)
        model: SpinDMModel with all parameters including phase_offset_cycles

    Returns:
        Residuals in seconds
    """
    # Compute spin phase at the given times
    dt = (t_mjd - model.tref_mjd) * 86400.0  # SECS_PER_DAY = 86400
    phase = model.f0 * dt + 0.5 * model.f1 * dt**2 + (1.0/6.0) * model.f2 * dt**3

    # Compute phase at the reference epoch
    dt_ref = (model.phase_ref_mjd - model.tref_mjd) * 86400.0
    phase_ref = model.f0 * dt_ref + 0.5 * model.f1 * dt_ref**2 + (1.0/6.0) * model.f2 * dt_ref**3

    # Phase difference from reference (accounting for TZR offset)
    phase_diff = phase - phase_ref - model.phase_offset_cycles

    # Wrap to (-0.5, 0.5] cycles
    frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5

    # Convert to time
    residual = frac_phase / model.f0

    return residual


# ==================================================
# FIX 3: Key insight from the notebook
# ==================================================

# The notebook concludes that the 1000x discrepancy comes from comparing:
# - JUG with INITIAL parameters → ~850 μs RMS
# - Tempo2 with FITTED parameters → ~0.8 μs RMS

# To fix this, we need to either:
# 1. Fit the parameters in JUG to minimize residuals
# 2. Or use Tempo2's FITTED parameter values (not the initial .par values)

# The .par file (temp_model_tdb.par) likely contains FITTED values
# But check if there's a pre-fit .par file being used instead

# ==================================================
# FIX 4: Check the time input
# ==================================================

# Critical fix from investigation:
# Cell 204 uses t_mjd (topocentric time) - THIS IS WRONG
# Should use t_inf_mjd (infinite-frequency time) instead

# The correct usage in cell 204 should be:

"""
# Compute infinite-frequency times first
t_inf_mjd = t_em_mjd - dm_delay_sec / 86400.0 + fd_delay_sec / 86400.0

# Then compute residuals at infinite-frequency time
t_inf_jax = jnp.array(t_inf_mjd, dtype=jnp.float64)
res_sec = residuals_seconds(t_inf_jax, model)
res_sec = res_sec - jnp.mean(res_sec)  # Remove mean
res_us = np.array(res_sec) * 1e6
"""

# ==================================================
# VERIFICATION
# ==================================================

def verify_residuals(res_jug_us, res_tempo2_us):
    """
    Verify that JUG residuals match Tempo2
    """
    # Calculate RMS
    rms_jug = np.sqrt(np.mean(res_jug_us**2))
    rms_t2 = np.sqrt(np.mean(res_tempo2_us**2))

    # Calculate correlation
    corr = np.corrcoef(res_jug_us, res_tempo2_us)[0, 1]

    # Calculate difference
    diff = res_jug_us - res_tempo2_us
    rms_diff = np.sqrt(np.mean(diff**2))

    print(f"JUG RMS: {rms_jug:.3f} μs")
    print(f"Tempo2 RMS: {rms_t2:.3f} μs")
    print(f"Correlation: {corr:.6f}")
    print(f"RMS difference: {rms_diff:.3f} μs")

    if corr > 0.999 and rms_diff < 1.0:
        print("✓ MATCH! Residuals agree with Tempo2")
        return True
    elif abs(rms_jug - rms_t2) > 100:
        print("✗ Large RMS discrepancy - likely using wrong parameters or wrong time")
        return False
    else:
        print("⚠ Close but not perfect - may need fine-tuning")
        return False


# ==================================================
# SUMMARY OF FIXES TO APPLY
# ==================================================

print("""
FIXES TO APPLY TO residual_maker_playground.ipynb:

1. In cell 233 (or wherever residuals are computed):
   - Ensure t_inf_mjd is computed from t_em_mjd minus DM delay
   - Use t_inf_jax for residual calculation, NOT t_mjd (topocentric)

2. Verify model.phase_offset_cycles is non-zero:
   - Should be ~0.08740234375 from TZR calculation
   - If zero, TZR calculation didn't run

3. Check parameter source:
   - temp_model_tdb.par should have FITTED parameters
   - If using initial/pre-fit parameters, that explains the discrepancy

4. After fixes, residuals should:
   - Have RMS ~0.8-10 μs (similar to Tempo2)
   - Correlate > 0.999 with Tempo2 residuals
""")
