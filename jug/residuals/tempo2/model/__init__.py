"""Tempo2 JAX model package."""

from jug.residuals.tempo2.clock_jax import (
    compute_einstein_rate_jax,
    compute_tempo2_get_correction_tt_jax,
    compute_tempo2_correction_tt_tb_jax,
)
from .tail import _tempo2_residual_tail_jax
from .static import (
    Tempo2ModelStatic,
    _dm_coeffs_from_params,
    _dm_vals_numpy,
    _eop_to_jax,
    _spk_to_jax,
    build_tempo2_model_static,
    compute_dm_vals_jax,
    compute_tempo2_einstein_rate_exact,
    host_frozen_vectors_from_tempo2_obs_state,
    planet_rsa_tuple_from_dict,
    planet_rsa_tuple_jax_from_dict,
    prepare_ephemeris_inputs_jax,
    tempo2_einstein_rate_host,
)
from .full import (
    compute_tempo2_toa_model_jax,
    run_tempo2_toa_model,
    run_tempo2_toa_model_with_fixed_ifte_geometry,
)
from .staged import compute_tempo2_toa_model_staging_with_host_inputs_jax
from .fixed_state import compute_tempo2_toa_model_fixed_state_nonlinear_jax

__all__ = [
    "Tempo2ModelStatic",
    "_dm_coeffs_from_params",
    "_dm_vals_numpy",
    "_eop_to_jax",
    "_spk_to_jax",
    "_tempo2_residual_tail_jax",
    "build_tempo2_model_static",
    "compute_dm_vals_jax",
    "compute_einstein_rate_jax",
    "compute_tempo2_correction_tt_tb_jax",
    "compute_tempo2_get_correction_tt_jax",
    "compute_tempo2_toa_model_fixed_state_nonlinear_jax",
    "compute_tempo2_toa_model_jax",
    "compute_tempo2_toa_model_staging_with_host_inputs_jax",
    "host_frozen_vectors_from_tempo2_obs_state",
    "planet_rsa_tuple_from_dict",
    "planet_rsa_tuple_jax_from_dict",
    "prepare_ephemeris_inputs_jax",
    "run_tempo2_toa_model",
    "run_tempo2_toa_model_with_fixed_ifte_geometry",
    "compute_tempo2_einstein_rate_exact",
    "tempo2_einstein_rate_host",
]
