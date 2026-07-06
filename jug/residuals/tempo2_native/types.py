"""JAX-native tempo2 chain term containers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Tempo2NativeTerms:
    """JAX-native tempo2 clock/delay/spin terms for one TOA batch."""

    sat_mjd: jnp.ndarray
    correction_tt_sec: jnp.ndarray
    correction_tt_tb_sec: jnp.ndarray
    roemer_sec: jnp.ndarray
    tdis1_sec: jnp.ndarray
    tdis2_sec: jnp.ndarray
    shapiro_sun_sec: jnp.ndarray
    shapiro_planets_sec: jnp.ndarray
    shapiro_delay_sec: jnp.ndarray
    tropospheric_sec: jnp.ndarray
    prebinary_sec: jnp.ndarray
    bat_corr_day: jnp.ndarray
    bat_corr_day_residual: jnp.ndarray
    bat_mjd: jnp.ndarray
    bbat_mjd: jnp.ndarray
    shklovskii_sec: jnp.ndarray
    torb_sec: jnp.ndarray
    dt_emission_sec: jnp.ndarray
    dt_ssb_sec: jnp.ndarray
    bclt_iterations: jnp.ndarray
    converged: jnp.ndarray


def native_terms_to_numpy(terms: Tempo2NativeTerms) -> dict[str, object]:
    """Device-get native terms at the public export boundary only."""
    import jax
    import numpy as np

    out: dict[str, object] = {}
    for name in terms.__dataclass_fields__:
        value = getattr(terms, name)
        arr = jax.device_get(value)
        out[name] = np.asarray(arr)
    return out
