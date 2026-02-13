"""Red noise and DM noise processes — Fourier-basis GP models.

Implements the standard power-law spectral models used in pulsar timing:

* **Achromatic red noise** (spin noise): frequency-dependent correlated
  noise with power spectrum P(f) = A² / (12π²) * (f/f_yr)^(-γ) * f_yr⁻¹.
  This noise is common to all observing frequencies and dominates at
  low timing frequencies.

* **DM noise** (chromatic): correlated noise whose amplitude scales as
  1/ν² (where ν is the observing frequency), following the expected
  chromatic signature of interstellar medium variations.

Both processes are modelled via a Fourier basis of sine/cosine pairs
evaluated at the TOA epochs.  The Fourier design matrix F has shape
(n_toa, 2 * n_harmonics), and the spectral coefficients are marginalised
analytically using a Woodbury identity or fitted via MCMC / likelihood
maximisation (following enterprise / PINT conventions).

For weighted least squares integration, the module provides:
  * ``build_fourier_design_matrix(toas_mjd, n_harmonics, Tspan)``
  * ``powerlaw_spectrum(freqs, log10_A, gamma)``
  * ``RedNoiseProcess`` / ``DMNoiseProcess`` dataclasses that bundle
    parameters and can compute the prior covariance φ = diag(spectrum).

References
----------
- van Haasteren & Levin (2013), "Understanding and analysing
  time-correlated stochastic signals in pulsar timing"
- Lentati et al. (2014), "Hyper-efficient model-independent Bayesian
  method for the analysis of pulsar timing data"
- NANOGrav / enterprise noise conventions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# JAX for JIT-compiled Fourier evaluation
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECS_PER_YEAR = 365.25 * 86400.0  # Julian year in seconds
_F_YR = 1.0 / _SECS_PER_YEAR      # 1/yr in Hz


# ---------------------------------------------------------------------------
# Fourier design matrix
# ---------------------------------------------------------------------------

@jax.jit
def _fourier_design_jax(
    t_sec: jnp.ndarray,
    freqs_hz: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled Fourier design matrix.

    Parameters
    ----------
    t_sec : (n_toa,)
        TOA times in seconds (relative to some reference).
    freqs_hz : (n_harmonics,)
        Fourier frequencies in Hz.

    Returns
    -------
    F : (n_toa, 2 * n_harmonics)
        Columns [sin(2π f₁ t), cos(2π f₁ t), sin(2π f₂ t), …].
    """
    phase = 2.0 * jnp.pi * jnp.outer(t_sec, freqs_hz)   # (n_toa, n_harm)
    sin_part = jnp.sin(phase)
    cos_part = jnp.cos(phase)
    # Interleave: [sin_1, cos_1, sin_2, cos_2, ...]
    n_toa = t_sec.shape[0]
    n_harm = freqs_hz.shape[0]
    F = jnp.empty((n_toa, 2 * n_harm))
    F = F.at[:, 0::2].set(sin_part)
    F = F.at[:, 1::2].set(cos_part)
    return F


def build_fourier_design_matrix(
    toas_mjd: np.ndarray,
    n_harmonics: int = 30,
    Tspan_days: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a Fourier design matrix for red/DM noise.

    Parameters
    ----------
    toas_mjd : np.ndarray, shape (n_toa,)
        TOA MJDs.
    n_harmonics : int, default 30
        Number of Fourier harmonics (each contributes sin + cos = 2 columns).
    Tspan_days : float, optional
        Time span in days.  If None, uses max(toas_mjd) - min(toas_mjd).
        The fundamental frequency is 1/Tspan.

    Returns
    -------
    F : np.ndarray, shape (n_toa, 2 * n_harmonics)
        Fourier design matrix.
    freqs_hz : np.ndarray, shape (n_harmonics,)
        Fourier frequencies in Hz.
    """
    toas_mjd = np.asarray(toas_mjd, dtype=np.float64)

    if Tspan_days is None:
        Tspan_days = float(toas_mjd.max() - toas_mjd.min())
    if Tspan_days <= 0:
        # Single-TOA or zero span: use 1 year as default span
        Tspan_days = 365.25

    Tspan_sec = Tspan_days * 86400.0

    # Fourier frequencies: k/Tspan for k = 1, …, n_harmonics
    freqs_hz = np.arange(1, n_harmonics + 1, dtype=np.float64) / Tspan_sec

    # Reference time at start of span
    t0_mjd = toas_mjd.min()
    t_sec = (toas_mjd - t0_mjd) * 86400.0

    F = np.asarray(_fourier_design_jax(jnp.array(t_sec), jnp.array(freqs_hz)))
    return F, freqs_hz


# ---------------------------------------------------------------------------
# Power-law spectrum
# ---------------------------------------------------------------------------

def powerlaw_spectrum(
    freqs_hz: np.ndarray,
    log10_A: float,
    gamma: float,
) -> np.ndarray:
    """Power-law PSD evaluated at given frequencies.

    .. math::

        P(f) = \\frac{A^2}{12 \\pi^2} \\left(\\frac{f}{f_{\\rm yr}}\\right)^{-\\gamma} f_{\\rm yr}^{-1}

    Parameters
    ----------
    freqs_hz : np.ndarray, shape (n,)
        Frequencies in Hz.
    log10_A : float
        Log10 of the spectral amplitude.
    gamma : float
        Spectral index (positive = red).

    Returns
    -------
    P : np.ndarray, shape (n,)
        PSD values (s³ = s²/Hz).
    """
    A = 10.0 ** log10_A
    return (A ** 2 / (12.0 * np.pi ** 2)) * (freqs_hz / _F_YR) ** (-gamma) / _F_YR


def turnover_spectrum(
    freqs_hz: np.ndarray,
    log10_A: float,
    gamma: float,
    f_bend_hz: float,
    kappa: float = 1.0,
) -> np.ndarray:
    """Power-law spectrum with optional low-frequency turnover.

    .. math::

        P(f) = P_{\\rm pl}(f) \\times \\left(1 + (f / f_{\\rm bend})^{-\\kappa / \\gamma}\\right)^{-\\gamma}

    For ``f >> f_bend`` this reduces to the usual power law.

    Parameters
    ----------
    freqs_hz : np.ndarray
    log10_A, gamma : float
        Standard power-law parameters.
    f_bend_hz : float
        Bend / turnover frequency in Hz.
    kappa : float, default 1.0
        Smoothness of the turnover.

    Returns
    -------
    P : np.ndarray
    """
    P = powerlaw_spectrum(freqs_hz, log10_A, gamma)
    # Smooth turnover: multiply by [1 + (f_bend/f)^kappa]^{-1}
    # This suppresses power at f << f_bend
    ratio = f_bend_hz / freqs_hz
    turnover = 1.0 / (1.0 + ratio ** kappa)
    return P * turnover


# ---------------------------------------------------------------------------
# Noise process dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RedNoiseProcess:
    """Achromatic red noise (spin noise).

    Attributes
    ----------
    log10_A : float
        Log10 spectral amplitude.
    gamma : float
        Spectral index.
    n_harmonics : int
        Number of Fourier harmonics.
    """
    log10_A: float
    gamma: float
    n_harmonics: int = 30

    def spectrum(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Evaluate PSD at given frequencies."""
        return powerlaw_spectrum(freqs_hz, self.log10_A, self.gamma)

    def build_basis_and_prior(
        self,
        toas_mjd: np.ndarray,
        Tspan_days: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build Fourier basis F and diagonal prior φ.

        Uses the enterprise convention for the per-coefficient variance:

        .. math::

            \\phi_k = \\frac{A^2}{12\\pi^2} f_{\\rm yr}^{\\gamma-3} f_k^{-\\gamma} \\Delta f

        where Δf = 1/T_span is the frequency resolution.

        Returns
        -------
        F : (n_toa, 2 * n_harmonics)
        phi : (2 * n_harmonics,)
            Prior variance for each Fourier coefficient (s²).
        """
        F, freqs = build_fourier_design_matrix(
            toas_mjd, self.n_harmonics, Tspan_days
        )
        # Frequency resolution (all harmonics equally spaced)
        df = freqs[0]  # = 1/T_span (fundamental frequency = frequency spacing)
        A = 10.0 ** self.log10_A
        # Enterprise convention: phi = A²/(12π²) × f_yr^(γ-3) × f^(-γ) × Δf
        phi_per_harmonic = (A ** 2 / (12.0 * np.pi ** 2)) * \
            _F_YR ** (self.gamma - 3) * freqs ** (-self.gamma) * df
        # Each harmonic contributes [sin, cos], both with the same variance
        phi = np.repeat(phi_per_harmonic, 2)
        return F, phi


@dataclass
class DMNoiseProcess:
    """Chromatic DM noise (scales as 1/ν²).

    The DM noise Fourier basis is identical to red noise but each
    column is multiplied by the DM-delay chromatic weight:
    ``K_DM / ν²`` (in seconds per DM unit).

    Attributes
    ----------
    log10_A : float
        Log10 spectral amplitude [DM units].
    gamma : float
        Spectral index.
    n_harmonics : int
        Number of Fourier harmonics.
    """
    log10_A: float
    gamma: float
    n_harmonics: int = 30

    def spectrum(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Evaluate PSD at given frequencies."""
        return powerlaw_spectrum(freqs_hz, self.log10_A, self.gamma)

    def build_basis_and_prior(
        self,
        toas_mjd: np.ndarray,
        freq_mhz: np.ndarray,
        Tspan_days: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build chromatic Fourier basis F_dm and diagonal prior φ.

        The achromatic Fourier basis is scaled column-wise by
        ``(1400 / freq_mhz)²`` to impart the ν⁻² chromatic signature.

        Uses the enterprise convention for per-coefficient variance:

        .. math::

            \\phi_k = \\frac{A^2}{12\\pi^2} f_{\\rm yr}^{\\gamma-3} f_k^{-\\gamma} \\Delta f

        Parameters
        ----------
        toas_mjd : (n_toa,)
        freq_mhz : (n_toa,)
            Observing frequencies in MHz.
        Tspan_days : float, optional

        Returns
        -------
        F_dm : (n_toa, 2 * n_harmonics)
        phi : (2 * n_harmonics,)
            Prior variance for each Fourier coefficient (s²).
        """
        F, freqs = build_fourier_design_matrix(
            toas_mjd, self.n_harmonics, Tspan_days
        )
        # Chromatic weighting: (1400 / ν)²
        # Using 1400 MHz as normalisation keeps numerical values ≈ O(1)
        chromatic_weight = (1400.0 / freq_mhz) ** 2
        F_dm = F * chromatic_weight[:, None]
        # Enterprise convention for per-coefficient variance
        df = freqs[0]  # = 1/T_span
        A = 10.0 ** self.log10_A
        phi_per_harmonic = (A ** 2 / (12.0 * np.pi ** 2)) * \
            _F_YR ** (self.gamma - 3) * freqs ** (-self.gamma) * df
        phi = np.repeat(phi_per_harmonic, 2)
        return F_dm, phi


# ---------------------------------------------------------------------------
# Parsing helpers (for par-file integration)
# ---------------------------------------------------------------------------

def parse_red_noise_params(params: dict) -> Optional[RedNoiseProcess]:
    """Extract a RedNoiseProcess from a par file dict, if present.

    Looks for ``TNRedAmp`` / ``TNRedGam`` / ``TNRedC`` (TempoNest)
    or ``RN_log10_A`` / ``RN_gamma`` conventions.
    Handles both mixed-case and uppercase keys (par reader uppercases).

    Returns None if no red noise parameters are found.
    """
    # TempoNest convention (mixed case)
    if "TNRedAmp" in params and "TNRedGam" in params:
        return RedNoiseProcess(
            log10_A=float(params["TNRedAmp"]),
            gamma=float(params["TNRedGam"]),
            n_harmonics=int(params.get("TNRedC", 30)),
        )
    # TempoNest convention (uppercase — par reader uppercases keys)
    if "TNREDAMP" in params and "TNREDGAM" in params:
        return RedNoiseProcess(
            log10_A=float(params["TNREDAMP"]),
            gamma=float(params["TNREDGAM"]),
            n_harmonics=int(params.get("TNREDC", 30)),
        )
    # Alternative convention
    if "RN_log10_A" in params and "RN_gamma" in params:
        return RedNoiseProcess(
            log10_A=float(params["RN_log10_A"]),
            gamma=float(params["RN_gamma"]),
            n_harmonics=int(params.get("RN_ncoeff", 30)),
        )
    return None


def parse_dm_noise_params(params: dict) -> Optional[DMNoiseProcess]:
    """Extract a DMNoiseProcess from a par file dict, if present.

    Looks for ``TNDMAmp`` / ``TNDMGam`` / ``TNDMC`` (TempoNest)
    or ``DM_log10_A`` / ``DM_gamma`` conventions.
    Handles both mixed-case and uppercase keys (par reader uppercases).
    """
    # TempoNest convention (mixed case)
    if "TNDMAmp" in params and "TNDMGam" in params:
        return DMNoiseProcess(
            log10_A=float(params["TNDMAmp"]),
            gamma=float(params["TNDMGam"]),
            n_harmonics=int(params.get("TNDMC", 30)),
        )
    # TempoNest convention (uppercase)
    if "TNDMAMP" in params and "TNDMGAM" in params:
        return DMNoiseProcess(
            log10_A=float(params["TNDMAMP"]),
            gamma=float(params["TNDMGAM"]),
            n_harmonics=int(params.get("TNDMC", 30)),
        )
    # Alternative convention
    if "DM_log10_A" in params and "DM_gamma" in params:
        return DMNoiseProcess(
            log10_A=float(params["DM_log10_A"]),
            gamma=float(params["DM_gamma"]),
            n_harmonics=int(params.get("DM_ncoeff", 30)),
        )
    return None


# ---------------------------------------------------------------------------
# Noise realization — compute MAP (maximum a-posteriori) realization
# ---------------------------------------------------------------------------

def realize_red_noise(
    toas_mjd: np.ndarray,
    residuals_sec: np.ndarray,
    errors_sec: np.ndarray,
    log10_A: float,
    gamma: float,
    n_harmonics: int = 30,
    Tspan_days: Optional[float] = None,
) -> np.ndarray:
    """Compute MAP realization of achromatic red noise.

    Uses the Wiener filter: ``realization = φ F^T (F φ F^T + N)^{-1} r``
    where φ is the prior covariance, F is the Fourier design matrix,
    N is the white noise covariance, and r is the residual vector.

    Returns
    -------
    realization_sec : (n_toa,)
        MAP noise realization in seconds.
    """
    proc = RedNoiseProcess(log10_A, gamma, n_harmonics)
    F, phi = proc.build_basis_and_prior(toas_mjd, Tspan_days)
    return _wiener_filter(F, phi, residuals_sec, errors_sec)


def realize_dm_noise(
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    residuals_sec: np.ndarray,
    errors_sec: np.ndarray,
    log10_A: float,
    gamma: float,
    n_harmonics: int = 30,
    Tspan_days: Optional[float] = None,
) -> np.ndarray:
    """Compute MAP realization of chromatic DM noise.

    Returns
    -------
    realization_sec : (n_toa,)
        MAP noise realization in seconds.
    """
    proc = DMNoiseProcess(log10_A, gamma, n_harmonics)
    F, phi = proc.build_basis_and_prior(toas_mjd, freq_mhz, Tspan_days)
    return _wiener_filter(F, phi, residuals_sec, errors_sec)


def _wiener_filter(
    F: np.ndarray,
    phi: np.ndarray,
    residuals: np.ndarray,
    errors: np.ndarray,
) -> np.ndarray:
    """Wiener filter: φ F^T (F φ F^T + N)^{-1} r."""
    N_inv = 1.0 / (errors ** 2)
    FtNir = F.T @ (N_inv * residuals)
    FtNiF = F.T @ (N_inv[:, None] * F)
    phi_inv = 1.0 / phi
    A = np.diag(phi_inv) + FtNiF
    coeffs = np.linalg.solve(A, FtNir)
    return F @ coeffs
