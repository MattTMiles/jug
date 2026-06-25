"""Chromatic transient event deterministic signal.

Computes a frequency-dependent transient (exponential or Gaussian decay)
that can model DM events, scattering events, or other chromatic transients.

The timing residual is:

    s(t, nu) = A * exp(+/-(t - t0)/tau) * (nu / nu_ref)^(-idx) * Theta(+/-(t - t0))

where:
    A     -- amplitude (seconds)
    t0    -- event epoch (MJD)
    tau     -- decay timescale (days)
    idx   -- chromatic index (2 = DM-like, 4 = scattering-like)
    sign  -- +1 for exponential rise, -1 for exponential decay (default)
    nu_ref -- reference frequency (1400 MHz by default)

Par parameters:

    CHROMEV_epoch  -- event epoch (MJD)
    CHROMEV_amp    -- amplitude (seconds)
    CHROMEV_tau    -- decay timescale (days)
    CHROMEV_idx    -- chromatic index (default 2)
    CHROMEV_sign   -- exponential sign: +1 or -1 (default -1)

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
    # Zero the exponent OUTSIDE the active region before exp() so the inactive
    # branch can't overflow (exp(+large) -> inf, then 0*inf -> nan). This bit
    # exponential dips whose epoch sits near the data END (most TOAs precede it,
    # dt very negative): e.g. J1713+0747 EXPEP=59320.
    safe_arg = jnp.where(active > 0.0, sign * dt / tau_day, 0.0)
    envelope = active * jnp.exp(safe_arg)

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

        Requires ``toa_freqs_mhz`` -- raises ValueError if not provided.
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
            f"tau={self.tau_day:.1f} d, idx={self.chrom_idx:.0f}"
        )


@register_signal
@dataclass
class ExponentialDipSignal(DeterministicSignal):
    """Tempo2-style exponential chromatic dip events (EXPEP/EXPPH/EXPTAU/EXPINDEX).

    Tempo2 (formResiduals.C ~645) sums, per event k and for t > EXPEP_k:

        delay = EXPPH_k * (f_bary / 1.4 GHz)**EXPINDEX_k * exp(-(t - EXPEP_k)/EXPTAU_k)

    EXPINDEX defaults to -2 (DM-like); EXPPH in seconds; EXPTAU in days; t = barycentric
    TOA. This is the SIMPLE one-sided exponential the PPTA pars were fit with. PINT's
    SimpleExponentialDip uses a smoothed/normalized variant (EXPDIPEPS) that differs from
    Tempo2 near the epoch; JUG previously modelled NEITHER (detect_signals never bridged
    the EXP* keys) and left the full ~us chromatic dip unmodelled (J1713+0747 EXPEP_1 ->
    2160 ns at 676 MHz). Each event maps onto the chromatic-event kernel with
    chrom_idx = -EXPINDEX (the kernel uses (f/fref)**(-chrom_idx)) and sign = -1 (decay
    after epoch).
    """

    signal_name: str = "ExponentialDip"
    epochs: tuple = ()      # EXPEP_k, MJD
    amps: tuple = ()        # EXPPH_k, seconds
    taus: tuple = ()        # EXPTAU_k, days
    indices: tuple = ()     # EXPINDEX_k

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if toa_freqs_mhz is None:
            raise ValueError(
                "ExponentialDipSignal requires TOA frequencies (toa_freqs_mhz)."
            )
        ref_mjd = toas_mjd[0]
        toas_day = jnp.asarray(toas_mjd - ref_mjd)
        freqs = jnp.asarray(toa_freqs_mhz)
        out = jnp.zeros(len(toas_mjd))
        for ep, amp, tau, idx in zip(self.epochs, self.amps, self.taus, self.indices):
            out = out + _chromatic_event_delay(
                toas_day, freqs, float(ep) - ref_mjd,
                float(amp), float(tau), -float(idx), -1.0,
            )
        # Sign: the EXP event is a delay DIP (negative), matching PINT's
        # SimpleExponentialDip and Tempo2 (positive EXPPH -> phase advance ->
        # negative residual). The chromatic-event kernel returns +amp*envelope*
        # freq_scale, so negate. Verified: with this sign JUG's own J1713+0747
        # WRMS collapses to the par TRES; the un-negated sign doubled the dip.
        return -np.asarray(out)

    @classmethod
    def from_par(cls, params: dict) -> "ExponentialDipSignal":
        eps, amps, taus, idxs = [], [], [], []
        i = 1
        while f"EXPEP_{i}" in params:
            eps.append(float(params[f"EXPEP_{i}"]))
            amps.append(float(params.get(f"EXPPH_{i}", 0.0)))
            taus.append(float(params.get(f"EXPTAU_{i}", 1.0)))
            idxs.append(float(params.get(f"EXPINDEX_{i}", -2.0)))
            i += 1
        return cls(epochs=tuple(eps), amps=tuple(amps),
                   taus=tuple(taus), indices=tuple(idxs))

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["EXPEP_1", "EXPPH_1", "EXPTAU_1", "EXPINDEX_1"]

    def summary(self) -> str:
        return (f"ExponentialDip: {len(self.epochs)} event(s), "
                f"EXPEP={[round(e, 1) for e in self.epochs]}")
