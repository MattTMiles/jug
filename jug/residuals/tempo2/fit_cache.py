"""Typed fit-cache container for tempo2-native JAX chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tempo2NativeChainStatic:
    """Frozen host state exported from ``compute_residuals_simple`` for tempo2 fits."""

    term_diagnostics: dict
    dt_sec: Any
    freq_bary_mhz: Any
    model_mjd: Any
    ssb_obs_pos_ls: Any | None
    obs_sun_pos_ls: Any | None
    obs_planet_pos_ls: Any | None
    toas: list | None

    def get(self, key: str, default=None):
        """Dict-like access for builder compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        return getattr(self, key)
