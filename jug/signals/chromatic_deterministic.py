"""Deterministic chromatic signals: chromatic Gaussian bump and annual.

These mirror the MPTA/run_enterprise deterministic chromatic signals that PINT
only carries as *dummy* parameter holders (``ChromBump`` / ``ChromaticAnnual``
in pint.models.noise_model expose ``get_pl_vals()`` but apply no delay). JUG
models them as proper deterministic signals so they can be evaluated, plotted,
and -- via jug.scripts.compare_pint_batch -- injected into PINT as a frozen
delay component for a like-for-like comparison.

Conventions (enterprise):
  - Chromatic scaling: ``(fref / freq)^idx`` with ``fref = 1400 MHz`` (same as
    :class:`jug.signals.chromatic_event.ChromaticEventSignal`).
  - Amplitudes are log10(seconds): ``A = 10**CHROM*AMP``.
  - Bump: ``sign * A * exp(-(t - T0)^2 / (2 sigma^2)) * (fref/freq)^idx`` with
    ``T0`` an MJD and ``sigma`` in days.
  - Annual: ``A * sin(2*pi * MJD/365.25 + phase) * (fref/freq)^idx``.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - JAX always present in JUG envs
    jnp = np

from jug.signals.base import DeterministicSignal, register_signal

_FREF_MHZ = 1400.0
_DAYS_PER_YEAR = 365.25


@register_signal
@dataclass
class ChromBumpSignal(DeterministicSignal):
    """Chromatic Gaussian transient (``CHROMBUMP*``).

    delay(t, nu) = sign * 10**AMP * exp(-(t - T0)^2 / (2 sigma^2)) * (fref/nu)^idx
    """

    signal_name: str = "ChromBump"

    log10_amp: float = -6.0
    sign_param: float = 1.0
    t0_mjd: float = 55000.0
    sigma_day: float = 100.0  # Gaussian width in days
    chrom_idx: float = 4.0

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if toa_freqs_mhz is None:
            raise ValueError(
                "ChromBumpSignal requires TOA frequencies (toa_freqs_mhz)."
            )
        # discovery.deterministic.chromatic_gaussian, with t0/sigma carried in
        # days here (the par stores them in seconds; see from_par).
        t = jnp.asarray(np.asarray(toas_mjd, dtype=float))
        freqs = jnp.asarray(np.asarray(toa_freqs_mhz, dtype=float))
        amp = 10.0 ** self.log10_amp
        gauss = jnp.exp(-((t - self.t0_mjd) ** 2) / (2.0 * self.sigma_day ** 2))
        freq_scale = (_FREF_MHZ / freqs) ** self.chrom_idx
        return np.asarray(jnp.sign(self.sign_param) * amp * gauss * freq_scale)

    @classmethod
    def from_par(cls, params: dict) -> "ChromBumpSignal":
        # CHROMBUMPT and CHROMBUMPSIGMA are written in SECONDS in these par
        # files (T0 ~ MJD*86400, sigma in s); convert to days for the
        # MJD-based Gaussian. Amplitude is log10(seconds).
        _SEC_PER_DAY = 86400.0
        return cls(
            log10_amp=float(params.get("CHROMBUMPAMP", -6.0)),
            sign_param=float(params.get("CHROMBUMPSIGN", 1.0)),
            t0_mjd=float(params.get("CHROMBUMPT", 55000.0 * _SEC_PER_DAY)) / _SEC_PER_DAY,
            sigma_day=float(params.get("CHROMBUMPSIGMA", 100.0 * _SEC_PER_DAY)) / _SEC_PER_DAY,
            chrom_idx=float(params.get("CHROMBUMPIDX", 4.0)),
        )

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["CHROMBUMPAMP", "CHROMBUMPT", "CHROMBUMPSIGMA"]

    def summary(self) -> str:
        return (
            f"ChromBump: log10A={self.log10_amp:.2f}, "
            f"sign={np.sign(self.sign_param):+.0f}, T0={self.t0_mjd:.1f}, "
            f"sigma={self.sigma_day:.1f} d, idx={self.chrom_idx:.1f}"
        )


@register_signal
@dataclass
class ChromaticAnnualSignal(DeterministicSignal):
    """Chromatic annual sinusoid (``CHROMANNUAL*``).

    delay(t, nu) = 10**AMP * sin(2*pi * MJD/365.25 + phase) * (fref/nu)^idx
    """

    signal_name: str = "ChromaticAnnual"

    log10_amp: float = -6.0
    phase: float = 0.0
    chrom_idx: float = 4.0

    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if toa_freqs_mhz is None:
            raise ValueError(
                "ChromaticAnnualSignal requires TOA frequencies (toa_freqs_mhz)."
            )
        t = jnp.asarray(np.asarray(toas_mjd, dtype=float))
        freqs = jnp.asarray(np.asarray(toa_freqs_mhz, dtype=float))
        amp = 10.0 ** self.log10_amp
        annual = jnp.sin(2.0 * np.pi * t / _DAYS_PER_YEAR + self.phase)
        freq_scale = (_FREF_MHZ / freqs) ** self.chrom_idx
        return np.asarray(amp * annual * freq_scale)

    @classmethod
    def from_par(cls, params: dict) -> "ChromaticAnnualSignal":
        return cls(
            log10_amp=float(params.get("CHROMANNUALAMP", -6.0)),
            phase=float(params.get("CHROMANNUALPHASE", 0.0)),
            chrom_idx=float(params.get("CHROMANNUALIDX", 4.0)),
        )

    @classmethod
    def required_par_keys(cls) -> List[str]:
        return ["CHROMANNUALAMP", "CHROMANNUALPHASE"]

    def summary(self) -> str:
        return (
            f"ChromaticAnnual: log10A={self.log10_amp:.2f}, "
            f"phase={self.phase:.3f}, idx={self.chrom_idx:.0f}"
        )
