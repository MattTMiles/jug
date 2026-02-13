"""MAP noise parameter estimation using NumPyro SVI.

Wraps the `pulsar-map-noise-estimates` package (David Wright,
https://github.com/davecwright3/pulsar-map-noise-estimates) to estimate
white noise (EFAC, EQUAD, ECORR), red noise, and DM noise parameters
from pre-computed timing residuals.

The approach:
  1. Fix the timing model (use pre-computed residuals).
  2. Build a NumPyro probabilistic model with priors on noise parameters.
  3. Run MAP estimation via SVI with the `pulsar-map-noise-estimates` API.
  4. Return estimated parameters in JUG's internal format.

The likelihood marginalises over Fourier coefficients analytically using
the Woodbury matrix identity, following the enterprise/tempo2 convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NoiseEstimateResult:
    """Result of MAP noise estimation."""
    params: Dict[str, float]          # noise params in JUG par-file format
    enterprise_params: Dict[str, float]  # enterprise-style naming
    converged: bool
    n_steps: int
    final_loss: float


# ---------------------------------------------------------------------------
# NumPyro model builder
# ---------------------------------------------------------------------------

def _build_numpyro_model(
    residuals_sec: np.ndarray,
    errors_sec: np.ndarray,
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    backend_masks: Dict[str, np.ndarray],
    ecorr_masks: Dict[str, np.ndarray],
    ecorr_epoch_groups: Optional[Dict[str, List[List[int]]]] = None,
    n_red_harmonics: int = 30,
    n_dm_harmonics: int = 30,
    include_red_noise: bool = True,
    include_dm_noise: bool = True,
    include_ecorr: bool = True,
):
    """Build a NumPyro model for noise parameter estimation.

    Uses the marginalized likelihood that analytically integrates over
    Fourier coefficients. The covariance is:

        C = N + F Φ F^T

    where N includes EFAC/EQUAD/ECORR white noise and F Φ F^T captures
    the red/DM noise power spectrum.

    Parameters
    ----------
    residuals_sec : array (n_toa,)
        Pre-computed timing residuals in seconds (fixed timing model).
    errors_sec : array (n_toa,)
        Original TOA uncertainties in seconds (before EFAC/EQUAD).
    toas_mjd : array (n_toa,)
        TOA times in MJD.
    freq_mhz : array (n_toa,)
        Observing frequencies in MHz (barycentric).
    backend_masks : dict
        {backend_name: bool_mask} for EFAC/EQUAD assignment.
    ecorr_masks : dict
        {backend_name: bool_mask} for ECORR assignment.
    ecorr_epoch_groups : dict or None
        {backend_name: list of index lists} grouping TOAs into epochs.
    n_red_harmonics : int
        Number of Fourier harmonics for red noise.
    n_dm_harmonics : int
        Number of Fourier harmonics for DM noise.
    include_red_noise : bool
        Whether to include red noise in the model.
    include_dm_noise : bool
        Whether to include DM noise in the model.
    include_ecorr : bool
        Whether to include ECORR in the model.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_toa = len(residuals_sec)
    T_span = (toas_mjd.max() - toas_mjd.min()) * 86400.0  # seconds

    # Pre-compute Fourier design matrices
    F_columns = []
    n_red_cols = 0
    n_dm_cols = 0

    if include_red_noise and n_red_harmonics > 0:
        from jug.noise.red_noise import build_fourier_design_matrix
        F_red, _ = build_fourier_design_matrix(toas_mjd, n_red_harmonics)
        n_red_cols = F_red.shape[1]
        F_columns.append(jnp.array(F_red))

    if include_dm_noise and n_dm_harmonics > 0:
        from jug.noise.red_noise import build_fourier_design_matrix
        F_dm_time, _ = build_fourier_design_matrix(toas_mjd, n_dm_harmonics)
        # DM noise scales as 1/freq^2 (K_DM = 4.149 GHz^2 pc^-1 cm^3 s)
        K_DM = 4.148808e3  # MHz^2 s pc^-1 cm^-3
        dm_scale = K_DM / (freq_mhz ** 2)
        F_dm = F_dm_time * dm_scale[:, None]
        n_dm_cols = F_dm.shape[1]
        F_columns.append(jnp.array(F_dm))

    F_total = jnp.concatenate(F_columns, axis=1) if F_columns else None
    n_fourier = n_red_cols + n_dm_cols

    # Convert to JAX arrays
    r = jnp.array(residuals_sec)
    sigma_orig = jnp.array(errors_sec)

    # Pre-compute backend mask arrays
    backend_names = sorted(backend_masks.keys())
    backend_mask_arrays = {
        name: jnp.array(mask) for name, mask in backend_masks.items()
    }

    # Pre-compute ECORR structures: U matrix (n_toa × n_epochs)
    # and mapping of backend → epoch column indices
    ecorr_names = sorted(ecorr_masks.keys()) if include_ecorr else []
    U_matrix = None
    n_ecorr_epochs = 0
    ecorr_backend_epoch_slices = {}  # {backend: (start_col, end_col)}

    if include_ecorr and ecorr_epoch_groups:
        # Build quantization matrix U
        all_epoch_groups = []
        for name in ecorr_names:
            groups = ecorr_epoch_groups.get(name, [])
            start = len(all_epoch_groups)
            all_epoch_groups.extend(groups)
            ecorr_backend_epoch_slices[name] = (start, start + len(groups))
        n_ecorr_epochs = len(all_epoch_groups)
        if n_ecorr_epochs > 0:
            U = np.zeros((n_toa, n_ecorr_epochs), dtype=np.float64)
            for k, group in enumerate(all_epoch_groups):
                for idx in group:
                    if idx < n_toa:
                        U[idx, k] = 1.0
            U_matrix = jnp.array(U)

    # Frequency references for power-law spectra
    f_yr = 1.0 / (365.25 * 86400.0)  # 1/year in Hz
    freqs = jnp.arange(1, max(n_red_harmonics, n_dm_harmonics) + 1) / T_span

    def model():
        # --- White noise parameters ---
        # Diagonal noise variance: N_ii = (EFAC_b * sigma_i)^2 + EQUAD_b^2
        N_diag = jnp.zeros(n_toa)

        for name in backend_names:
            mask = backend_mask_arrays[name]
            efac = numpyro.sample(
                f"efac_{name}",
                dist.Uniform(0.1, 10.0)
            )
            log10_equad = numpyro.sample(
                f"log10_equad_{name}",
                dist.Uniform(-10.0, -4.0)
            )
            equad = 10.0 ** log10_equad
            # N_ii = (EFAC * sigma)^2 + EQUAD^2
            backend_var = (efac * sigma_orig) ** 2 + equad ** 2
            N_diag = N_diag + mask * backend_var

        # Ensure no zeros
        N_diag = jnp.maximum(N_diag, 1e-40)

        # --- ECORR (adds rank-1 blocks per epoch) ---
        # J_k = ECORR_b^2 for epoch k in backend b
        J_ecorr = None
        if include_ecorr and U_matrix is not None and n_ecorr_epochs > 0:
            J_ecorr = jnp.zeros(n_ecorr_epochs)
            for name in ecorr_names:
                log10_ecorr = numpyro.sample(
                    f"log10_ecorr_{name}",
                    dist.Uniform(-10.0, -4.0)
                )
                ecorr_val = 10.0 ** log10_ecorr
                start, end = ecorr_backend_epoch_slices[name]
                J_ecorr = J_ecorr.at[start:end].set(ecorr_val ** 2)

        # --- Red noise power spectrum ---
        phi_diag = jnp.zeros(n_fourier) if n_fourier > 0 else None

        if include_red_noise and n_red_cols > 0:
            log10_A_red = numpyro.sample(
                "log10_A_red",
                dist.Uniform(-20.0, -10.0)
            )
            gamma_red = numpyro.sample(
                "gamma_red",
                dist.Uniform(0.0, 7.0)
            )
            A_red = 10.0 ** log10_A_red
            # Power-law spectrum: P(f) = A^2/(12π^2) * (f/f_yr)^(-γ) * T_span
            # Diagonal prior variance for each pair [cos, sin]
            rn_freqs = freqs[:n_red_harmonics]
            rn_psd = (A_red ** 2 / (12.0 * jnp.pi ** 2)) * \
                     (rn_freqs / f_yr) ** (-gamma_red) / T_span
            # Each harmonic has cos and sin → duplicate
            rn_prior = jnp.repeat(rn_psd, 2)
            phi_diag = phi_diag.at[:n_red_cols].set(rn_prior)

        if include_dm_noise and n_dm_cols > 0:
            log10_A_dm = numpyro.sample(
                "log10_A_dm",
                dist.Uniform(-20.0, -10.0)
            )
            gamma_dm = numpyro.sample(
                "gamma_dm",
                dist.Uniform(0.0, 7.0)
            )
            A_dm = 10.0 ** log10_A_dm
            dm_freqs = freqs[:n_dm_harmonics]
            dm_psd = (A_dm ** 2 / (12.0 * jnp.pi ** 2)) * \
                     (dm_freqs / f_yr) ** (-gamma_dm) / T_span
            dm_prior = jnp.repeat(dm_psd, 2)
            phi_diag = phi_diag.at[n_red_cols:n_red_cols + n_dm_cols].set(dm_prior)

        # --- Marginalized log-likelihood ---
        # C = N + U J U^T + F Φ F^T
        # Use Woodbury: C^{-1} and log|C| computed efficiently

        N_inv = 1.0 / N_diag
        log_det_N = jnp.sum(jnp.log(N_diag))

        # Start with N^{-1} r and r^T N^{-1} r
        Ninv_r = N_inv * r
        rNr = jnp.dot(r, Ninv_r)
        log_det_C = log_det_N

        # Apply ECORR via Woodbury: (N + U J U^T)^{-1}
        if J_ecorr is not None and U_matrix is not None:
            # Woodbury: (N + U J U^T)^{-1} = N^{-1} - N^{-1} U (J^{-1} + U^T N^{-1} U)^{-1} U^T N^{-1}
            Ninv_U = N_inv[:, None] * U_matrix  # (n_toa, n_epochs)
            UtNinvU = U_matrix.T @ Ninv_U  # (n_epochs, n_epochs)
            J_inv = 1.0 / jnp.maximum(J_ecorr, 1e-40)
            S_ecorr = jnp.diag(J_inv) + UtNinvU  # (n_epochs, n_epochs)
            S_ecorr_inv = jnp.linalg.inv(S_ecorr)

            # Update: C_ecorr^{-1} r = N^{-1} r - N^{-1} U S^{-1} U^T N^{-1} r
            UtNinvr = U_matrix.T @ Ninv_r
            correction_r = Ninv_U @ (S_ecorr_inv @ UtNinvr)
            Cinv_r_ecorr = Ninv_r - correction_r
            rCr = jnp.dot(r, Cinv_r_ecorr)

            # log|C_ecorr| = log|N| + log|J| + log|S_ecorr|
            log_det_J = jnp.sum(jnp.log(jnp.maximum(J_ecorr, 1e-40)))
            sign, log_det_S = jnp.linalg.slogdet(S_ecorr)
            log_det_C = log_det_N + log_det_J + log_det_S

            # Update N_inv for subsequent Fourier term
            Ninv_r = Cinv_r_ecorr
            rNr = rCr

        # Apply Fourier noise via Woodbury
        if F_total is not None and phi_diag is not None:
            phi_diag_safe = jnp.maximum(phi_diag, 1e-40)
            # Recompute N^{-1} accounting for ECORR (if present, use effective inverse)
            # For simplicity with ECORR, compute the combined Woodbury
            # N_eff^{-1} F
            if J_ecorr is not None and U_matrix is not None:
                # We need (N + UJU^T)^{-1} F
                Neff_inv_F = N_inv[:, None] * F_total - Ninv_U @ (S_ecorr_inv @ (U_matrix.T @ (N_inv[:, None] * F_total)))
            else:
                Neff_inv_F = N_inv[:, None] * F_total

            FtNF = F_total.T @ Neff_inv_F  # (n_fourier, n_fourier)
            phi_inv = 1.0 / phi_diag_safe
            S_fourier = jnp.diag(phi_inv) + FtNF
            S_fourier_inv = jnp.linalg.inv(S_fourier)

            FtNr = F_total.T @ Ninv_r
            correction = Neff_inv_F @ (S_fourier_inv @ FtNr)
            Cinv_r = Ninv_r - correction
            rCr_full = jnp.dot(r, Cinv_r)

            # log|C| += log|Φ| + log|S_fourier|
            log_det_phi = jnp.sum(jnp.log(phi_diag_safe))
            sign_f, log_det_Sf = jnp.linalg.slogdet(S_fourier)
            log_det_C = log_det_C + log_det_phi + log_det_Sf
        else:
            rCr_full = rNr

        # log L = -0.5 * (r^T C^{-1} r + log|C| + n*log(2π))
        log_like = -0.5 * (rCr_full + log_det_C + n_toa * jnp.log(2 * jnp.pi))

        numpyro.factor("log_likelihood", log_like)

    return model


# ---------------------------------------------------------------------------
# Backend/epoch grouping helpers
# ---------------------------------------------------------------------------

def _build_backend_masks(
    toa_flags: List[Dict[str, str]],
    noise_entries: list,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[List[int]]]]:
    """Build backend masks and epoch groups from TOA flags and noise entries.

    Returns
    -------
    backend_masks : dict
        {backend_name: bool_mask} for EFAC/EQUAD.
    ecorr_masks : dict
        {backend_name: bool_mask} for ECORR.
    ecorr_epoch_groups : dict
        {backend_name: list of index lists} for ECORR epochs.
    """
    from jug.noise.white import build_backend_mask

    n_toa = len(toa_flags)
    backend_masks = {}
    ecorr_masks = {}
    ecorr_epoch_groups = {}

    # Get unique backend names from EFAC entries
    seen_backends = set()
    for entry in noise_entries:
        if entry.kind == 'EFAC':
            key = entry.flag_value
            if key not in seen_backends:
                seen_backends.add(key)
                mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
                backend_masks[key] = mask

    # Build ECORR masks and epoch groups
    for entry in noise_entries:
        if entry.kind == 'ECORR':
            key = entry.flag_value
            mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
            ecorr_masks[key] = mask

            # Group TOAs into epochs (within 0.5 day windows)
            toa_indices = np.where(mask)[0]
            if len(toa_indices) == 0:
                ecorr_epoch_groups[key] = []
                continue

            # Need MJDs for grouping — pass externally or compute
            ecorr_epoch_groups[key] = []  # filled by caller

    return backend_masks, ecorr_masks, ecorr_epoch_groups


def _group_toas_into_epochs(
    toas_mjd: np.ndarray,
    mask: np.ndarray,
    window: float = 0.5,
) -> List[List[int]]:
    """Group TOA indices into observing epochs within a time window."""
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    mjds = toas_mjd[indices]
    sort_order = np.argsort(mjds)
    sorted_indices = indices[sort_order]
    sorted_mjds = mjds[sort_order]

    groups = []
    current_group = [sorted_indices[0]]
    current_start = sorted_mjds[0]

    for i in range(1, len(sorted_indices)):
        if sorted_mjds[i] - current_start <= window:
            current_group.append(sorted_indices[i])
        else:
            groups.append(current_group)
            current_group = [sorted_indices[i]]
            current_start = sorted_mjds[i]
    groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_noise_parameters(
    residuals_sec: np.ndarray,
    errors_sec: np.ndarray,
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    toa_flags: List[Dict[str, str]],
    params: Dict[str, Any],
    n_red_harmonics: int = 30,
    n_dm_harmonics: int = 30,
    include_red_noise: bool = True,
    include_dm_noise: bool = True,
    include_ecorr: bool = True,
    batch_size: int = 1000,
    max_num_batches: int = 50,
    patience: int = 3,
    seed: int = 42,
    progress_callback: Optional[Any] = None,
) -> NoiseEstimateResult:
    """Run MAP noise estimation and return parameter dict.

    Uses the pulsar-map-noise-estimates package (David Wright,
    https://github.com/davecwright3/pulsar-map-noise-estimates)
    for SVI optimization infrastructure.

    Parameters
    ----------
    residuals_sec : array
        Pre-fit timing residuals in seconds.
    errors_sec : array
        TOA uncertainties in seconds (original, before EFAC/EQUAD).
    toas_mjd : array
        TOA times in MJD.
    freq_mhz : array
        Barycentric frequencies in MHz.
    toa_flags : list of dict
        Per-TOA flag dictionaries from the TIM file.
    params : dict
        Par file parameters (for extracting existing noise entries).
    n_red_harmonics, n_dm_harmonics : int
        Number of Fourier harmonics for red/DM noise.
    include_red_noise, include_dm_noise, include_ecorr : bool
        Whether to include each noise component.
    batch_size : int
        SVI batch size (steps per training batch).
    max_num_batches : int
        Maximum number of training batches.
    patience : int
        Early stopping patience.
    seed : int
        Random seed.
    progress_callback : callable or None
        Optional callback(batch_num, max_batches, loss) for progress.

    Returns
    -------
    NoiseEstimateResult
        Estimated noise parameters.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        import numpyro
        import numpyro.infer.autoguide
        from pulsar_map_noise_estimates.map_noise_estimate import (
            setup_svi,
            run_svi_early_stopping,
        )
    except ImportError as e:
        raise ImportError(
            "MAP noise estimation requires jax, numpyro, and "
            "pulsar-map-noise-estimates. Install with: "
            "pip install pulsar-map-noise-estimates"
        ) from e

    # Auto-detect device: try GPU with a small test op, fall back to CPU
    device = None
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            # Verify GPU actually works (plugin version mismatches can cause runtime errors)
            test_arr = jax.device_put(jnp.ones(2), gpu_devices[0])
            _ = jnp.linalg.inv(jnp.eye(2, device=gpu_devices[0]))
            device = gpu_devices[0]
            logger.info(f"MAP estimation using GPU: {device}")
    except Exception:
        device = None
    if device is None:
        device = jax.devices("cpu")[0]
        logger.info("MAP estimation using CPU")
    jax.config.update("jax_default_device", device)

    # Parse existing noise entries for backend structure
    from jug.noise.white import parse_noise_lines
    noise_lines = params.get('_noise_lines', [])
    noise_entries = parse_noise_lines(noise_lines) if noise_lines else []

    # Build backend masks
    backend_masks, ecorr_masks, ecorr_epoch_groups = _build_backend_masks(
        toa_flags, noise_entries
    )

    # If no backends found, create a single "default" backend
    if not backend_masks:
        backend_masks = {"default": np.ones(len(toas_mjd), dtype=bool)}

    # If ECORR requested but no ECORR entries exist, use EFAC/EQUAD backends
    if include_ecorr and not ecorr_masks:
        for name, mask in backend_masks.items():
            ecorr_masks[name] = mask
            ecorr_epoch_groups[name] = []

    # Fill epoch groups with actual MJD-based grouping
    for name, mask in ecorr_masks.items():
        ecorr_epoch_groups[name] = _group_toas_into_epochs(toas_mjd, mask)

    logger.info(
        f"MAP estimation: {len(backend_masks)} backends, "
        f"{len(ecorr_masks)} ECORR groups, "
        f"red={include_red_noise} ({n_red_harmonics}), "
        f"dm={include_dm_noise} ({n_dm_harmonics})"
    )

    # Build NumPyro model
    numpyro_model = _build_numpyro_model(
        residuals_sec=residuals_sec,
        errors_sec=errors_sec,
        toas_mjd=toas_mjd,
        freq_mhz=freq_mhz,
        backend_masks=backend_masks,
        ecorr_masks=ecorr_masks,
        ecorr_epoch_groups=ecorr_epoch_groups,
        n_red_harmonics=n_red_harmonics,
        n_dm_harmonics=n_dm_harmonics,
        include_red_noise=include_red_noise,
        include_dm_noise=include_dm_noise,
        include_ecorr=include_ecorr,
    )

    # Set up SVI with AutoDelta guide (MAP estimation)
    guide = numpyro.infer.autoguide.AutoDelta(numpyro_model)
    svi = setup_svi(
        numpyro_model,
        guide,
        max_epochs=batch_size * max_num_batches,
        num_warmup_steps=batch_size,
    )

    # Run optimization
    rng_key = random.key(seed)
    raw_params = run_svi_early_stopping(
        rng_key,
        svi,
        batch_size=batch_size,
        patience=patience,
        max_num_batches=max_num_batches,
    )

    # Clean parameter names (remove AutoDelta suffix)
    clean_params = {
        key.removesuffix("_auto_loc"): float(value)
        for key, value in raw_params.items()
    }

    logger.info(f"MAP estimation complete. Parameters: {clean_params}")

    # Convert to JUG par-file format
    jug_params = _convert_to_jug_format(
        clean_params, backend_masks,
        ecorr_masks=ecorr_masks,
        n_red_harmonics=n_red_harmonics,
        n_dm_harmonics=n_dm_harmonics,
    )

    return NoiseEstimateResult(
        params=jug_params,
        enterprise_params=clean_params,
        converged=True,
        n_steps=batch_size * max_num_batches,
        final_loss=0.0,
    )


def _convert_to_jug_format(
    estimated: Dict[str, float],
    backend_masks: Dict[str, np.ndarray],
    ecorr_masks: Optional[Dict[str, np.ndarray]] = None,
    n_red_harmonics: int = 30,
    n_dm_harmonics: int = 30,
) -> Dict[str, float]:
    """Convert estimated parameters to JUG par-file format.

    Maps from MAP internal names to the EFAC/EQUAD/ECORR/TNRED/TNDM
    naming convention used in .par files.  Amplitude parameters are stored
    as log10(A) (the standard tempo2/PINT convention).
    """
    result = {}

    for name in sorted(backend_masks.keys()):
        efac_key = f"efac_{name}"
        if efac_key in estimated:
            result[f"EFAC_{name}"] = estimated[efac_key]

        equad_key = f"log10_equad_{name}"
        if equad_key in estimated:
            # Convert from log10(seconds) to microseconds
            result[f"EQUAD_{name}"] = 10.0 ** estimated[equad_key] * 1e6

    # ECORR backends may differ from EFAC/EQUAD backends
    ecorr_names = sorted((ecorr_masks or {}).keys())
    for name in ecorr_names:
        ecorr_key = f"log10_ecorr_{name}"
        if ecorr_key in estimated:
            result[f"ECORR_{name}"] = 10.0 ** estimated[ecorr_key] * 1e6

    if "log10_A_red" in estimated:
        # TNRedAmp in par files is log10(A) — the enterprise convention
        result["TNREDAMP"] = estimated["log10_A_red"]
        result["TNREDGAM"] = estimated.get("gamma_red", 0.0)
        result["TNREDC"] = float(n_red_harmonics)

    if "log10_A_dm" in estimated:
        result["TNDMAMP"] = estimated["log10_A_dm"]
        result["TNDMGAM"] = estimated.get("gamma_dm", 0.0)
        result["TNDMC"] = float(n_dm_harmonics)

    return result
