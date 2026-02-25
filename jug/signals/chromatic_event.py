"""Chromatic transient event deterministic signal.

Computes a frequency-dependent transient (exponential or Gaussian decay)
that can model DM events, scattering events, or other chromatic transients.

The timing residual is:

    s(t, ν) = A * exp(±(t - t0)/τ) * (ν / ν_ref)^(-idx) * Θ(±(t - t0))

where:
    A     — amplitude (seconds)
    t0    — event epoch (MJD)
    τ     — decay timescale (days)
    idx   — chromatic index (2 = DM-like, 4 = scattering-like)
    sign  — +1 for exponential rise, -1 for exponential decay (default)
    ν_ref — reference frequency (1400 MHz by default)

Par parameters:

    CHROMEV_epoch  — event epoch (MJD)
    CHROMEV_amp    — amplitude (seconds)
    CHROMEV_tau    — decay timescale (days)
    CHROMEV_idx    — chromatic index (default 2)
    CHROMEV_sign   — exponential sign: +1 or -1 (default -1)

Reference: Lentati et al. (2017) for chromatic noise modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp
import numpy as np

from jug.signals.base import DeterministicSignal, register_signal

# Reference frequency for chromatic scaling (MHz)
_FREF_MHZ = 1400.0


# ---------------------------------------------------------------------------
# Chromatic transient waveform (JAX)
# ---------------------------------------------------------------------------

def _chromatic_event_delay(
    toas_day: jnp.ndarray,
    freqs_mhz: jnp.ndarray,
    epoch_day: float,
    amp_sec: float,
    tau_day: float,
    chrom_idx: float,
    sign: float,
) -> jnp.ndarray:
    """Compute chromatic transient timing residual.

    Parameters
    ----------
    toas_day : jnp.ndarray, shape (n_toa,)
        TOA times in days (relative to some reference).
    freqs_mhz : jnp.ndarray, shape (n_toa,)
        Observing frequencies in MHz.
    epoch_day : float
        Event epoch in days (same reference as toas_day).
    amp_sec : float
        Amplitude in seconds.
    tau_day : float
        Decay timescale in days.
    chrom_idx : float
        Chromatic index (2 = DM, 4 = scattering).
    sign : float
        +1 for rise after epoch, -1 for decay after epoch.

    Returns
    -------
    jnp.ndarray, shape (n_toa,)
        Timing residual in seconds.
    """
    dt = toas_day - epoch_day

    # Active after epoch (decay) or before epoch (rise)
    # sign=-1: active for dt >= 0, envelope = exp(-dt/tau)
    # sign=+1: active for dt <= 0, envelope = exp(dt/tau)
    active = jnp.where(dt * (-sign) >= 0.0, 1.0, 0.0)
    envelope = active * jnp.exp(sign * dt / tau_day)

    # Chromatic scaling
    freq_scale = (freqs_mhz / _FREF_MHZ) ** (-chrom_idx)

    return amp_sec * envelope * freq_scale


# ---------------------------------------------------------------------------
# Signal class
# ---------------------------------------------------------------------------

@register_signal
@dataclass
class ChromaticEventSignal(DeterministicSignal):
    """Frequency-dependent transient event (DM event, scattering event)."""

    signal_name: str = "ChromaticEvent"

    epoch_mjd: float = 55000.0
    amp_sec: float = 1e-6
    tau_day: float = 30.0
    chrom_idx: float = 2.0
    sign: float = -1.0

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute chromatic event timing residual.

        Requires ``toa_freqs_mhz`` — raises ValueError if not provided.
        """
        if toa_freqs_mhz is None:
            raise ValueError(
                "ChromaticEventSignal requires TOA frequencies "
                "(toa_freqs_mhz). Pass them from the tim file."
            )

        ref_mjd = toas_mjd[0]
        toas_day = jnp.asarray(toas_mjd - ref_mjd)
        epoch_day = self.epoch_mjd - ref_mjd
        freqs = jnp.asarray(toa_freqs_mhz)

        result = _chromatic_event_delay(
            toas_day, freqs, epoch_day,
            self.amp_sec, self.tau_day, self.chrom_idx, self.sign,
        )
        return np.asarray(result)

    @classmethod
    def from_par(cls, params: dict) -> "ChromaticEventSignal":
        return cls(
            epoch_mjd=float(params.get("CHROMEV_EPOCH", 55000.0)),
            amp_sec=float(params.get("CHROMEV_AMP", 1e-6)),
            tau_day=float(params.get("CHROMEV_TAU", 30.0)),
            chrom_idx=float(params.get("CHROMEV_IDX", 2.0)),
            sign=float(params.get("CHROMEV_SIGN", -1.0)),
        )

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["CHROMEV_EPOCH", "CHROMEV_AMP", "CHROMEV_TAU"]

    def summary(self) -> str:
        return (
            f"ChromaticEvent: A={self.amp_sec:.2e} s, "
            f"τ={self.tau_day:.1f} d, idx={self.chrom_idx:.0f}"
        )
