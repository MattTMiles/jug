"""Regression tests for the MAP noise estimator's red/DM amplitude convention.

JUG's MAP estimator (jug.noise.map_estimator) must recover red- and DM-noise
log10-amplitudes on the SAME enterprise/Lentati (2014) convention used by
jug.noise.red_noise (the GLS path) and by external samplers (Discovery):

    phi_k = A^2/(12 pi^2) * f_yr^(gamma-3) * f_k^(-gamma) * (1/T_span)

with the DM Fourier basis weighted by (1400 MHz / nu)^2.

Two past bugs these tests guard against:
  1. Dropping the f_yr^(-3) factor in the power-law prior -> log10_A ~11 dex off
     the enterprise scale and the red/DM term effectively unconstrained.
  2. Using K_DM/nu^2 instead of (1400/nu)^2 for the DM basis -> log10_A_dm on a
     different (~2.7 dex offset) scale.

Both manifest as a multi-dex error in the recovered amplitude, so a 0.5 dex
tolerance cleanly distinguishes the correct convention from a regression while
tolerating single-realization MAP scatter.
"""

import numpy as np
import pytest

# The estimator needs numpyro/optax; skip cleanly if unavailable.
pytest.importorskip("numpyro")
pytest.importorskip("optax")

from jug.noise.red_noise import RedNoiseProcess, DMNoiseProcess, ChromaticNoiseProcess
from jug.noise.map_estimator import estimate_noise_parameters


def _toas(n=600, span_days=5000.0):
    mjd = np.linspace(55000.0, 55000.0 + span_days, n)
    return mjd, float(mjd.max() - mjd.min())


def _run_map(residuals_sec, errors_sec, mjd, freq_mhz, *, red, dm, chrom=False, n_mode=20):
    flags = [{"f": "sys"} for _ in range(len(mjd))]
    est = estimate_noise_parameters(
        residuals_sec, errors_sec, mjd, freq_mhz, flags, {"_noise_lines": []},
        n_red_harmonics=n_mode, n_dm_harmonics=n_mode, n_chrom_harmonics=n_mode,
        include_red_noise=red, include_dm_noise=dm, include_ecorr=False,
        include_chrom=chrom,
        batch_size=1000, max_num_batches=25, patience=4, seed=3,
    )
    return est.enterprise_params


def test_map_recovers_injected_enterprise_red_amplitude():
    """Inject a known enterprise-convention red process; MAP must recover log10_A."""
    A_true, g_true, n_mode = -13.0, 4.0, 20      # strong red -> good SNR
    mjd, Tspan = _toas()
    freq = np.full(len(mjd), 1400.0)
    errors_sec = np.full(len(mjd), 0.3e-6)

    rng = np.random.default_rng(100)
    F, phi = RedNoiseProcess(log10_A=A_true, gamma=g_true,
                             n_harmonics=n_mode).build_basis_and_prior(mjd, Tspan_days=Tspan)
    res = F @ rng.normal(0.0, np.sqrt(phi)) + rng.normal(0.0, errors_sec)

    ep = _run_map(res, errors_sec, mjd, freq, red=True, dm=False, n_mode=n_mode)

    # Amplitude is the convention guard (a reverted f_yr^-3 lands ~11 dex away).
    assert abs(ep["log10_A_red"] - A_true) < 0.5, ep["log10_A_red"]
    # Spectral index is only weakly constrained from one realization (MAP-mode
    # bias + intrinsic scatter ~0.2-0.3); assert a loose sanity bound only.
    assert abs(ep["gamma_red"] - g_true) < 1.5, ep["gamma_red"]


def test_map_recovers_injected_enterprise_dm_amplitude():
    """Inject a known enterprise-convention DM process; MAP must recover log10_A_dm.

    Guards the DM chromatic basis: (1400/nu)^2 (enterprise), not K_DM/nu^2.
    """
    A_true, g_true, n_mode = -13.0, 3.0, 20
    mjd, Tspan = _toas()
    # Multiple frequencies so the chromatic (1400/nu)^2 weighting is exercised.
    freq = np.tile([820.0, 1400.0, 2100.0], len(mjd) // 3 + 1)[:len(mjd)]
    errors_sec = np.full(len(mjd), 0.3e-6)

    rng = np.random.default_rng(101)
    F_dm, phi = DMNoiseProcess(log10_A=A_true, gamma=g_true,
                               n_harmonics=n_mode).build_basis_and_prior(mjd, freq, Tspan_days=Tspan)
    res = F_dm @ rng.normal(0.0, np.sqrt(phi)) + rng.normal(0.0, errors_sec)

    ep = _run_map(res, errors_sec, mjd, freq, red=False, dm=True, n_mode=n_mode)

    assert abs(ep["log10_A_dm"] - A_true) < 0.5, ep["log10_A_dm"]
    assert abs(ep["gamma_dm"] - g_true) < 1.5, ep["gamma_dm"]


def test_map_recovers_chromatic_scattering_with_variable_index():
    """Inject a chromatic process with a known chromaticity index beta; MAP must
    recover both log10_A_chrom (enterprise scale) and the fitted beta.
    """
    A_true, g_true, beta_true, n_mode = -13.0, 3.0, 4.0, 20   # scattering-like
    mjd, Tspan = _toas(n=900)
    freq = np.tile([820.0, 1400.0, 2100.0], len(mjd) // 3 + 1)[:len(mjd)]
    errors_sec = np.full(len(mjd), 0.3e-6)

    rng = np.random.default_rng(5)
    F, phi = ChromaticNoiseProcess(
        log10_A=A_true, gamma=g_true, chrom_idx=beta_true, n_harmonics=n_mode
    ).build_basis_and_prior(mjd, freq, Tspan_days=Tspan)
    res = F @ rng.normal(0.0, np.sqrt(phi)) + rng.normal(0.0, errors_sec)

    ep = _run_map(res, errors_sec, mjd, freq, red=False, dm=False, chrom=True, n_mode=n_mode)

    assert abs(ep["log10_A_chrom"] - A_true) < 0.5, ep["log10_A_chrom"]
    # The chromaticity index is well-constrained from multi-frequency data.
    assert abs(ep["chrom_idx"] - beta_true) < 0.7, ep["chrom_idx"]
    assert abs(ep["gamma_chrom"] - g_true) < 1.5, ep["gamma_chrom"]
