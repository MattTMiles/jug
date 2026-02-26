"""Continuous gravitational wave (CW) deterministic signal.

Computes the Earth-term timing residual induced by a monochromatic
gravitational wave source, following the enterprise convention:

    s(t) = F_plus(theta,phi,psi) * s_plus(t) + F_cross(theta,phi,psi) * s_cross(t)

where F_plus/F_cross are antenna pattern functions and s_plus/s_cross
are the GW-induced timing residuals for the two polarisations.

Enterprise-compatible par parameters:

    CW_log10_h      -- log10 of GW strain amplitude
    CW_cos_gwtheta  -- cosine of GW source polar angle (ecliptic)
    CW_gwphi        -- GW source azimuthal angle (rad)
    CW_log10_fgw    -- log10 of GW frequency (Hz)
    CW_phase0       -- initial GW phase (rad)
    CW_cos_inc      -- cosine of orbital inclination
    CW_psi          -- GW polarisation angle (rad)

Reference: Ellis et al. (2012), ApJ 756, 175; enterprise CW model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp

from jug.utils.constants import SECS_PER_DAY
import numpy as np

from jug.signals.base import DeterministicSignal, register_signal


# ---------------------------------------------------------------------------
# Antenna pattern functions (JAX)
# ---------------------------------------------------------------------------

def _antenna_pattern(
    gwtheta: float,
    gwphi: float,
    psi: float,
    raj: float,
    decj: float,
) -> tuple:
    """Compute antenna pattern functions F_plus and F_cross.

    Parameters
    ----------
    gwtheta : float
        GW source polar angle (rad, from ecliptic north).
    gwphi : float
        GW source azimuthal angle (rad).
    psi : float
        GW polarisation angle (rad).
    raj : float
        Pulsar right ascension (rad).
    decj : float
        Pulsar declination (rad).

    Returns
    -------
    F_plus, F_cross : float
        Antenna pattern coefficients.
    """
    # GW propagation direction
    cos_gwtheta = jnp.cos(gwtheta)
    sin_gwtheta = jnp.sin(gwtheta)
    cos_gwphi = jnp.cos(gwphi)
    sin_gwphi = jnp.sin(gwphi)

    # GW source frame basis vectors
    m_hat = jnp.array([
        -sin_gwphi, cos_gwphi, 0.0
    ])
    n_hat = jnp.array([
        -cos_gwtheta * cos_gwphi,
        -cos_gwtheta * sin_gwphi,
        sin_gwtheta,
    ])
    omega_hat = jnp.array([
        -sin_gwtheta * cos_gwphi,
        -sin_gwtheta * sin_gwphi,
        -cos_gwtheta,
    ])

    # Pulsar direction
    cos_dec = jnp.cos(decj)
    sin_dec = jnp.sin(decj)
    cos_ra = jnp.cos(raj)
    sin_ra = jnp.sin(raj)
    p_hat = jnp.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])

    # Antenna pattern
    F_plus_no_psi = (
        0.5 * (jnp.dot(m_hat, p_hat) ** 2 - jnp.dot(n_hat, p_hat) ** 2)
        / (1.0 + jnp.dot(omega_hat, p_hat))
    )
    F_cross_no_psi = (
        jnp.dot(m_hat, p_hat) * jnp.dot(n_hat, p_hat)
        / (1.0 + jnp.dot(omega_hat, p_hat))
    )

    # Rotate by polarisation angle
    cos_2psi = jnp.cos(2 * psi)
    sin_2psi = jnp.sin(2 * psi)
    F_plus = F_plus_no_psi * cos_2psi - F_cross_no_psi * sin_2psi
    F_cross = F_plus_no_psi * sin_2psi + F_cross_no_psi * cos_2psi

    return F_plus, F_cross


# ---------------------------------------------------------------------------
# CW waveform (JAX)
# ---------------------------------------------------------------------------

def _cw_delay(
    toas_sec: jnp.ndarray,
    log10_h: float,
    cos_gwtheta: float,
    gwphi: float,
    log10_fgw: float,
    phase0: float,
    cos_inc: float,
    psi: float,
    raj: float,
    decj: float,
) -> jnp.ndarray:
    """Compute CW Earth-term timing residual.

    Parameters
    ----------
    toas_sec : jnp.ndarray, shape (n_toa,)
        TOA times in seconds (relative to some reference).
    log10_h, cos_gwtheta, gwphi, log10_fgw, phase0, cos_inc, psi : float
        GW source parameters (enterprise convention).
    raj, decj : float
        Pulsar right ascension and declination (rad).

    Returns
    -------
    jnp.ndarray, shape (n_toa,)
        Timing residual in seconds (Earth term only).
    """
    h = 10.0 ** log10_h
    fgw = 10.0 ** log10_fgw
    gwtheta = jnp.arccos(cos_gwtheta)
    inc = jnp.arccos(cos_inc)

    F_plus, F_cross = _antenna_pattern(gwtheta, gwphi, psi, raj, decj)

    omega_gw = 2.0 * jnp.pi * fgw
    phase = phase0 + omega_gw * toas_sec

    # Plus and cross polarisation waveforms (Earth term)
    s_plus = -(h / omega_gw) * (1.0 + cos_inc ** 2) * jnp.sin(phase)
    s_cross = (2.0 * h / omega_gw) * cos_inc * jnp.cos(phase)

    return F_plus * s_plus + F_cross * s_cross


# ---------------------------------------------------------------------------
# Signal class
# ---------------------------------------------------------------------------

@register_signal
@dataclass
class ContinuousWaveSignal(DeterministicSignal):
    """Continuous gravitational wave signal (Earth term)."""

    signal_name: str = "CW"

    log10_h: float = -14.0
    cos_gwtheta: float = 0.0
    gwphi: float = 0.0
    log10_fgw: float = -8.0
    phase0: float = 0.0
    cos_inc: float = 0.0
    psi: float = 0.0
    raj: float = 0.0
    decj: float = 0.0

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute CW timing residual at each TOA epoch."""
        # Convert MJD to seconds from first TOA
        toas_sec = jnp.asarray((toas_mjd - toas_mjd[0]) * SECS_PER_DAY)

        result = _cw_delay(
            toas_sec,
            self.log10_h,
            self.cos_gwtheta,
            self.gwphi,
            self.log10_fgw,
            self.phase0,
            self.cos_inc,
            self.psi,
            self.raj,
            self.decj,
        )
        return np.asarray(result)

    @classmethod
    def from_par(cls, params: dict) -> "ContinuousWaveSignal":
        """Construct from par-file parameters (enterprise convention)."""
        return cls(
            log10_h=float(params.get("CW_LOG10_H", -14.0)),
            cos_gwtheta=float(params.get("CW_COS_GWTHETA", 0.0)),
            gwphi=float(params.get("CW_GWPHI", 0.0)),
            log10_fgw=float(params.get("CW_LOG10_FGW", -8.0)),
            phase0=float(params.get("CW_PHASE0", 0.0)),
            cos_inc=float(params.get("CW_COS_INC", 0.0)),
            psi=float(params.get("CW_PSI", 0.0)),
            raj=float(params.get("RAJ", 0.0)),
            decj=float(params.get("DECJ", 0.0)),
        )

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["CW_LOG10_H", "CW_LOG10_FGW"]

    def summary(self) -> str:
        fgw = 10.0 ** self.log10_fgw
        h = 10.0 ** self.log10_h
        return f"CW: h={h:.2e}, f={fgw:.2e} Hz"
