"""Burst with memory (BWM) deterministic signal.

Computes the timing residual from a gravitational wave burst with memory:
a permanent step in strain that produces a linearly growing timing residual
after the burst epoch.

Enterprise-compatible par parameters:

    BWM_log10_h      -- log10 of GW memory strain amplitude
    BWM_cos_gwtheta  -- cosine of GW source polar angle
    BWM_gwphi        -- GW source azimuthal angle (rad)
    BWM_t0           -- burst epoch (MJD)
    BWM_pol          -- polarisation angle (rad)

The timing residual for a BWM is:

    s(t) = h * F(theta,phi,psi) * (t - t0) * Theta(t - t0)

where Theta is the Heaviside step function and F is the antenna pattern.

Reference: van Haasteren & Levin (2010), MNRAS 401, 2372.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp
import numpy as np

from jug.utils.constants import SECS_PER_DAY

from jug.signals.base import DeterministicSignal, register_signal
from jug.signals.continuous_wave import _antenna_pattern


# ---------------------------------------------------------------------------
# BWM waveform (JAX)
# ---------------------------------------------------------------------------

def _bwm_delay(
    toas_sec: jnp.ndarray,
    t0_sec: float,
    log10_h: float,
    cos_gwtheta: float,
    gwphi: float,
    pol: float,
    raj: float,
    decj: float,
) -> jnp.ndarray:
    """Compute BWM timing residual.

    Parameters
    ----------
    toas_sec : jnp.ndarray, shape (n_toa,)
        TOA times in seconds (relative to some reference).
    t0_sec : float
        Burst epoch in seconds (same reference as toas_sec).
    log10_h : float
        log10 of memory strain amplitude.
    cos_gwtheta, gwphi, pol : float
        GW source sky location and polarisation.
    raj, decj : float
        Pulsar right ascension and declination (rad).

    Returns
    -------
    jnp.ndarray, shape (n_toa,)
        Timing residual in seconds.
    """
    h = 10.0 ** log10_h
    gwtheta = jnp.arccos(cos_gwtheta)

    # BWM only has plus polarisation for memory
    F_plus, F_cross = _antenna_pattern(gwtheta, gwphi, pol, raj, decj)

    dt = toas_sec - t0_sec
    # Heaviside step: linear ramp after burst
    ramp = jnp.where(dt > 0.0, dt, 0.0)

    return h * F_plus * ramp


# ---------------------------------------------------------------------------
# Signal class
# ---------------------------------------------------------------------------

@register_signal
@dataclass
class BurstWithMemorySignal(DeterministicSignal):
    """Gravitational wave burst with memory signal."""

    signal_name: str = "BWM"

    log10_h: float = -14.0
    cos_gwtheta: float = 0.0
    gwphi: float = 0.0
    t0_mjd: float = 55000.0
    pol: float = 0.0
    raj: float = 0.0
    decj: float = 0.0

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute BWM timing residual at each TOA epoch."""
        ref_mjd = toas_mjd[0]
        toas_sec = jnp.asarray((toas_mjd - ref_mjd) * SECS_PER_DAY)
        t0_sec = (self.t0_mjd - ref_mjd) * SECS_PER_DAY

        result = _bwm_delay(
            toas_sec,
            t0_sec,
            self.log10_h,
            self.cos_gwtheta,
            self.gwphi,
            self.pol,
            self.raj,
            self.decj,
        )
        return np.asarray(result)

    @classmethod
    def from_par(cls, params: dict) -> "BurstWithMemorySignal":
        return cls(
            log10_h=float(params.get("BWM_LOG10_H", -14.0)),
            cos_gwtheta=float(params.get("BWM_COS_GWTHETA", 0.0)),
            gwphi=float(params.get("BWM_GWPHI", 0.0)),
            t0_mjd=float(params.get("BWM_T0", 55000.0)),
            pol=float(params.get("BWM_POL", 0.0)),
            raj=float(params.get("RAJ", 0.0)),
            decj=float(params.get("DECJ", 0.0)),
        )

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["BWM_LOG10_H", "BWM_T0"]

    def summary(self) -> str:
        h = 10.0 ** self.log10_h
        return f"BWM: h={h:.2e}, t0={self.t0_mjd:.1f} MJD"
