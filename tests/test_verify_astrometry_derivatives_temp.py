
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
    """Verify analytical astrometry derivatives match JAX autodiff.

    The analytical d_delay_d_RAJ/DECJ compute the Roemer-delay derivative only
    (no PM or PX coupling).  To match the forward model exactly we therefore
    set PMRA=PMDEC=0 **and** PX=0 when testing RAJ/DECJ.

    PMRA/PMDEC derivatives use PM≠0 but PX=0 (Roemer+PM, no parallax).
    PX derivative uses the full parameter set (PM+PX).

    Tolerances: RAJ/DECJ use 1e-12 (pure geometry, no unit conversions).
    PMRA/PMDEC/PX use 1e-10 (mas↔rad conversions accumulate ~1e-12 FP noise
    that is amplified by the derivative magnitude).
    """
    print(f"Testing astrometry derivatives with {n_toas} random TOAs...")

    # 1. Setup Mock Data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    # Random observer positions (roughly 1 AU scale)
    ssb_obs_pos_ls = jax.random.normal(k1, (n_toas, 3)) * 500.0  # light-seconds

    # Random TOAs (spanning ~10 years)
    toas_mjd = 55000.0 + jax.random.uniform(k2, (n_toas,)) * 365.25 * 10

    # Pulsar parameters
    psr_ra_val = 19.16 * (jnp.pi / 12.0)   # ~19h
    psr_dec_val = -37.7 * (jnp.pi / 180.0)  # ~-37deg
    posepoch = 55000.0
    pmra_val = -3.7   # mas/yr
    pmdec_val = -9.5   # mas/yr
    px_val = 0.8       # mas

    def delay_wrt_param(val, param_name, base_params):
        p = base_params.copy()
        p[param_name] = val
        return compute_astrometric_delay(p, toas_mjd, ssb_obs_pos_ls)

    # ------------------------------------------------------------------
    # RAJ / DECJ — pure Roemer (PM=0, PX=0)
    # ------------------------------------------------------------------
    params_roemer = {
        'RAJ': psr_ra_val, 'DECJ': psr_dec_val,
        'POSEPOCH': posepoch,
        'PMRA': 0.0, 'PMDEC': 0.0, 'PX': 0.0,
    }

    # RAJ
    print("\n--- RAJ ---")
    grad_jax = jax.jacobian(
        lambda x: delay_wrt_param(x, 'RAJ', params_roemer)
    )(psr_ra_val)
    grad_ana = d_delay_d_RAJ(psr_ra_val, psr_dec_val, ssb_obs_pos_ls)
    err = jnp.max(jnp.abs(grad_jax - grad_ana))
    print(f"Max error (rad): {err:.2e}")
    assert err < 1e-12, f"RAJ derivative mismatch: {err:.2e}"

    # DECJ
    print("\n--- DECJ ---")
    grad_jax = jax.jacobian(
        lambda x: delay_wrt_param(x, 'DECJ', params_roemer)
    )(psr_dec_val)
    grad_ana = d_delay_d_DECJ(psr_ra_val, psr_dec_val, ssb_obs_pos_ls)
    err = jnp.max(jnp.abs(grad_jax - grad_ana))
    print(f"Max error (rad): {err:.2e}")
    assert err < 1e-12, f"DECJ derivative mismatch: {err:.2e}"

    # ------------------------------------------------------------------
    # PMRA / PMDEC — Roemer + PM (PX=0)
    # ------------------------------------------------------------------
    params_pm = {
        'RAJ': psr_ra_val, 'DECJ': psr_dec_val,
        'POSEPOCH': posepoch,
        'PMRA': pmra_val, 'PMDEC': pmdec_val, 'PX': 0.0,
    }

    # PMRA
    print("\n--- PMRA ---")
    grad_jax = jax.jacobian(
        lambda x: delay_wrt_param(x, 'PMRA', params_pm)
    )(pmra_val)
    grad_ana = d_delay_d_PMRA(
        psr_ra_val, psr_dec_val, ssb_obs_pos_ls,
        toas_mjd=toas_mjd, posepoch_mjd=posepoch,
    )
    # d_delay_d_PMRA returns s/(rad/yr); JAX differentiates w.r.t. mas/yr
    grad_ana_mas = grad_ana / MAS_PER_RAD
    err = jnp.max(jnp.abs(grad_jax - grad_ana_mas))
    print(f"Max error (mas/yr): {err:.2e}")
    assert err < 1e-10, f"PMRA derivative mismatch: {err:.2e}"

    # PMDEC
    print("\n--- PMDEC ---")
    grad_jax = jax.jacobian(
        lambda x: delay_wrt_param(x, 'PMDEC', params_pm)
    )(pmdec_val)
    grad_ana = d_delay_d_PMDEC(
        psr_ra_val, psr_dec_val, ssb_obs_pos_ls,
        toas_mjd=toas_mjd, posepoch_mjd=posepoch,
    )
    grad_ana_mas = grad_ana / MAS_PER_RAD
    err = jnp.max(jnp.abs(grad_jax - grad_ana_mas))
    print(f"Max error (mas/yr): {err:.2e}")
    assert err < 1e-10, f"PMDEC derivative mismatch: {err:.2e}"

    # ------------------------------------------------------------------
    # PX — full parameter set (PM + PX)
    # ------------------------------------------------------------------
    params_full = {
        'RAJ': psr_ra_val, 'DECJ': psr_dec_val,
        'POSEPOCH': posepoch,
        'PMRA': pmra_val, 'PMDEC': pmdec_val, 'PX': px_val,
    }

    print("\n--- PX ---")
    pmra_rad_yr = pmra_val * (jnp.pi / 180.0 / 3600000.0)
    pmdec_rad_yr = pmdec_val * (jnp.pi / 180.0 / 3600000.0)
    grad_jax = jax.jacobian(
        lambda x: delay_wrt_param(x, 'PX', params_full)
    )(px_val)
    grad_ana = d_delay_d_PX(
        psr_ra_val, psr_dec_val, ssb_obs_pos_ls,
        toas_mjd=toas_mjd, posepoch_mjd=posepoch,
        pmra_rad_yr=pmra_rad_yr, pmdec_rad_yr=pmdec_rad_yr,
    )
    grad_ana_mas = grad_ana / MAS_PER_RAD
    err = jnp.max(jnp.abs(grad_jax - grad_ana_mas))
    print(f"Max error (mas): {err:.2e}")
    assert err < 1e-10, f"PX derivative mismatch: {err:.2e}"

    print("\nSUCCESS: All 5 astrometry derivatives match JAX autodiff!")

if __name__ == "__main__":
    test_derivatives()
