
import jax
import jax.numpy as jnp
import numpy as np
from jug.fitting.derivatives_astrometry import (
    compute_astrometric_delay,
    d_delay_d_RAJ,
    d_delay_d_DECJ,
    d_delay_d_PMRA,
    d_delay_d_PMDEC,
    d_delay_d_PX,
    HOURANGLE_PER_RAD,
    DEG_PER_RAD,
    MAS_PER_RAD
)

# Enable x64 for precision
jax.config.update("jax_enable_x64", True)

def test_derivatives(n_toas=10):
    print(f"Testing astrometry derivatives with {n_toas} random TOAs...")
    
    # 1. Setup Mock Data
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Random observer positions (roughly 1 AU scale)
    ssb_obs_pos_ls = jax.random.normal(k1, (n_toas, 3)) * 500.0  # light-seconds
    
    # Random TOAs (spanning ~10 years)
    toas_mjd = 55000.0 + jax.random.uniform(k2, (n_toas,)) * 365.25 * 10
    
    # Pulsar parameters
    # J1909-like position
    psr_ra_val = 19.16 * (jnp.pi / 12.0)  # ~19h
    psr_dec_val = -37.7 * (jnp.pi / 180.0) # ~-37deg
    posepoch = 55000.0
    pmra_val = -3.7  # mas/yr
    pmdec_val = -9.5 # mas/yr
    px_val = 0.8     # mas
    
    params = {
        'RAJ': psr_ra_val,
        'DECJ': psr_dec_val,
        'POSEPOCH': posepoch,
        'PMRA': pmra_val,
        'PMDEC': pmdec_val,
        'PX': px_val,
    }
    
    # 2. Wrapper for JAX autodiff
    # We need a function delay(param_value) -> scalar delay (summed or per-toa)
    
    # Helper to update params and compute
    def delay_wrt_param(val, param_name, base_params):
        p = base_params.copy()
        p[param_name] = val
        # Return sum of delays to get gradient per toa (vector-jacobian product effectively)
        # Actually easier: just grad sum, then we check individual if needed. 
        # But JAX grad of scalar output is standard.
        # Let's verify element-wise by taking jacobian.
        d = compute_astrometric_delay(p, toas_mjd, ssb_obs_pos_ls)
        return d

    # 3. Compute and Compare
    
    # RAJ
    print("\n--- RAJ ---")
    jac_fn = jax.jacobian(lambda x: delay_wrt_param(x, 'RAJ', params))
    grad_jax = jac_fn(psr_ra_val) # shape (n_toas,)
    grad_ana_rad = d_delay_d_RAJ(psr_ra_val, psr_dec_val, ssb_obs_pos_ls)
    
    # Analytical is seconds/radian. 
    err = jnp.max(jnp.abs(grad_jax - grad_ana_rad))
    print(f"Max error (rad): {err:.2e}")
    assert err < 1e-12
    
    # DECJ
    print("\n--- DECJ ---")
    jac_fn = jax.jacobian(lambda x: delay_wrt_param(x, 'DECJ', params))
    grad_jax = jac_fn(psr_dec_val)
    grad_ana_rad = d_delay_d_DECJ(psr_ra_val, psr_dec_val, ssb_obs_pos_ls)
    err = jnp.max(jnp.abs(grad_jax - grad_ana_rad))
    print(f"Max error (rad): {err:.2e}")
    assert err < 1e-12

    # PX
    print("\n--- PX ---")
    # For PX, the input to compute_astrometric_delay is in mas
    # But d_delay_d_PX output is seconds/radian?? 
    # Let's check the docstring of d_delay_d_PX in the file.
    # It says: "@(delay)/@(PX) in units of seconds/radian"
    # But compute_astrometric_delay takes 'PX' in mas.
    # So jax.grad will be d(delay)/d(PX_mas).
    # d_delay_d_PX returns d(delay)/d(PX_rad).
    # We must convert.
    
    jac_fn = jax.jacobian(lambda x: delay_wrt_param(x, 'PX', params))
    grad_jax_mas = jac_fn(px_val) # seconds/mas
    
    grad_ana_rad = d_delay_d_PX(
        psr_ra_val, psr_dec_val, ssb_obs_pos_ls,
        toas_mjd=toas_mjd, posepoch_mjd=posepoch,
        pmra_rad_yr=pmra_val * (jnp.pi/180/3600000),
        pmdec_rad_yr=pmdec_val * (jnp.pi/180/3600000)
    )
    # Convert ana to mas
    grad_ana_mas = grad_ana_rad / MAS_PER_RAD
    
    err = jnp.max(jnp.abs(grad_jax_mas - grad_ana_mas))
    print(f"Max error (mas): {err:.2e}")
    assert err < 1e-12

    print("\nSUCCESS: All derivatives match JAX autodiff!")

if __name__ == "__main__":
    test_derivatives()
