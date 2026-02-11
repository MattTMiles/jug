"""Per-process noise configuration with auto-detection from .par file.

Each noise process (EFAC, EQUAD, ECORR, RedNoise, DMNoise, ...) has its own
on/off toggle.  The default state is determined by scanning the parsed par
parameters: if the par file contains the relevant keywords, that process
starts enabled.

Usage
-----
>>> from jug.engine.noise_mode import NoiseConfig
>>> nc = NoiseConfig.from_par(params)
>>> nc.enabled["EFAC"]        # True if par had EFAC/T2EFAC lines
>>> nc.toggle("ECORR")        # flip one process
>>> nc.active_processes()      # list of currently-enabled names

The config is intentionally simple and extensible — adding a new noise type
only requires registering its detection function in ``_DETECTORS``.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Known noise process names
# ---------------------------------------------------------------------------

# Canonical process names.  These are the keys used in ``enabled``.
EFAC = "EFAC"
EQUAD = "EQUAD"
ECORR = "ECORR"
RED_NOISE = "RedNoise"
DM_NOISE = "DMNoise"


# ---------------------------------------------------------------------------
# Detection helpers — each returns True if the par params contain evidence
# of that noise process.
# ---------------------------------------------------------------------------

def _has_efac(params: dict) -> bool:
    """Detect EFAC/T2EFAC entries in _noise_lines."""
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key in ("EFAC", "T2EFAC"):
            return True
    return False


def _has_equad(params: dict) -> bool:
    """Detect EQUAD/T2EQUAD entries in _noise_lines."""
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key in ("EQUAD", "T2EQUAD"):
            return True
    return False


def _has_ecorr(params: dict) -> bool:
    """Detect ECORR entries in _noise_lines."""
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key == "ECORR":
            return True
    return False


def _has_red_noise(params: dict) -> bool:
    """Detect red noise parameters (TempoNest or enterprise conventions)."""
    if "TNRedAmp" in params and "TNRedGam" in params:
        return True
    if "TNREDAMP" in params and "TNREDGAM" in params:
        return True
    if "RN_log10_A" in params and "RN_gamma" in params:
        return True
    return False


def _has_dm_noise(params: dict) -> bool:
    """Detect DM noise parameters (TempoNest or enterprise conventions)."""
    if "TNDMAmp" in params and "TNDMGam" in params:
        return True
    if "TNDMAMP" in params and "TNDMGAM" in params:
        return True
    if "DM_log10_A" in params and "DM_gamma" in params:
        return True
    return False


# Registry: canonical name -> detection function.
# Extend this dict to support new noise types.
_DETECTORS: Dict[str, callable] = {
    EFAC: _has_efac,
    EQUAD: _has_equad,
    ECORR: _has_ecorr,
    RED_NOISE: _has_red_noise,
    DM_NOISE: _has_dm_noise,
}


# ---------------------------------------------------------------------------
# NoiseConfig
# ---------------------------------------------------------------------------

class NoiseConfig:
    """Per-process noise toggle configuration.

    Parameters
    ----------
    enabled : dict[str, bool], optional
        Initial enabled state.  Missing processes default to False.
    """

    def __init__(self, enabled: Optional[Dict[str, bool]] = None):
        self.enabled: Dict[str, bool] = dict(enabled) if enabled else {}

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_par(cls, params: dict) -> "NoiseConfig":
        """Build config by auto-detecting noise processes in parsed par params.

        Parameters
        ----------
        params : dict
            Parsed parameter dictionary (from ``read_par``).  Must include
            ``_noise_lines`` for white-noise detection and regular param keys
            for red/DM noise.

        Returns
        -------
        NoiseConfig
            Config with each detected process enabled.
        """
        enabled = {}
        for name, detector in _DETECTORS.items():
            enabled[name] = detector(params)
        return cls(enabled)

    # -- Queries -----------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """Return True if *name* is a known and enabled process."""
        return self.enabled.get(name, False)

    def active_processes(self) -> List[str]:
        """Return list of currently-enabled process names."""
        return [k for k, v in self.enabled.items() if v]

    def has_any_noise(self) -> bool:
        """Return True if at least one noise process is enabled."""
        return any(self.enabled.values())

    # -- Mutations ---------------------------------------------------------

    def toggle(self, name: str) -> bool:
        """Flip the toggle for *name* and return the new state.

        If the process was not previously tracked it becomes True (enabled).
        """
        new_state = not self.enabled.get(name, False)
        self.enabled[name] = new_state
        return new_state

    def enable(self, name: str) -> None:
        """Enable a single process."""
        self.enabled[name] = True

    def disable(self, name: str) -> None:
        """Disable a single process."""
        self.enabled[name] = False

    def enable_all(self) -> None:
        """Enable every known process."""
        for k in self.enabled:
            self.enabled[k] = True

    def disable_all(self) -> None:
        """Disable every known process."""
        for k in self.enabled:
            self.enabled[k] = False

    # -- Serialization (for session save/load) -----------------------------

    def to_dict(self) -> Dict[str, bool]:
        """Return a plain dict suitable for JSON serialization."""
        return dict(self.enabled)

    @classmethod
    def from_dict(cls, d: Dict[str, bool]) -> "NoiseConfig":
        """Restore from a serialized dict."""
        return cls(enabled=d)

    # -- Display -----------------------------------------------------------

    def __repr__(self) -> str:
        active = ", ".join(self.active_processes()) or "(none)"
        return f"NoiseConfig(active=[{active}])"

    def summary(self) -> str:
        """Human-readable summary of noise toggle states."""
        lines = []
        for name in sorted(self.enabled):
            state = "ON" if self.enabled[name] else "OFF"
            lines.append(f"  {name}: {state}")
        return "Noise processes:\n" + "\n".join(lines) if lines else "Noise processes: (none detected)"
