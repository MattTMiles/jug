"""Base class and registry for deterministic signals.

All signal implementations subclass :class:`DeterministicSignal` and register
themselves in :data:`SIGNAL_REGISTRY`.  The :func:`detect_signals` helper scans
parsed par-file parameters and instantiates every signal whose required keys
are present.

Waveform functions are written in JAX so that ``jax.grad`` can be used to
compute derivatives for future fittable-signal support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIGNAL_REGISTRY: Dict[str, Type["DeterministicSignal"]] = {}
"""Maps a canonical signal name (e.g. ``"CW"``) to its class."""


def register_signal(cls: Type["DeterministicSignal"]) -> Type["DeterministicSignal"]:
    """Class decorator that adds a signal to the registry."""
    SIGNAL_REGISTRY[cls.signal_name] = cls
    return cls


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DeterministicSignal(ABC):
    """Abstract base for a deterministic timing-residual waveform.

    Subclasses must define:

    * :attr:`signal_name` -- unique string identifier (e.g. ``"CW"``).
    * :meth:`compute_waveform` -- return ``(n_toa,)`` delay in **seconds**.
    * :meth:`from_par` -- construct an instance from parsed par parameters.
    * :meth:`required_par_keys` -- return the par keys that trigger detection.
    """

    signal_name: str = ""
    """Canonical name used in the noise registry and GUI."""

    # -- Abstract interface -------------------------------------------------

    @abstractmethod
    def compute_waveform(
        self,
        toas_mjd: np.ndarray,
        toa_freqs_mhz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the timing-residual waveform at each TOA.

        Parameters
        ----------
        toas_mjd : ndarray, shape (n_toa,)
            TOA epochs in MJD.
        toa_freqs_mhz : ndarray, shape (n_toa,), optional
            Observing frequencies in MHz (required for chromatic signals).

        Returns
        -------
        ndarray, shape (n_toa,)
            Time delay in **seconds** to subtract from residuals.
        """

    @classmethod
    @abstractmethod
    def from_par(cls, params: dict) -> "DeterministicSignal":
        """Construct from parsed par-file parameters.

        Parameters
        ----------
        params : dict
            Full parsed par dictionary (from ``parse_par_file``).
        """

    @classmethod
    @abstractmethod
    def required_par_keys(cls) -> List[str]:
        """Return par parameter names whose presence triggers detection.

        All listed keys must be present for the signal to be detected.
        """

    # -- Helpers ------------------------------------------------------------

    @classmethod
    def is_detected(cls, params: dict) -> bool:
        """Return True if all required par keys are present."""
        return all(k in params for k in cls.required_par_keys())

    def summary(self) -> str:
        """One-line human-readable summary of signal parameters."""
        return f"{self.signal_name} signal"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_signals(params: dict) -> List[DeterministicSignal]:
    """Scan par parameters and return instances of all detected signals."""
    found: List[DeterministicSignal] = []
    for cls in SIGNAL_REGISTRY.values():
        if cls.is_detected(params):
            found.append(cls.from_par(params))
    return found
