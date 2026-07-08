"""User-facing timing configuration for JUG sessions and fit exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tempo2GraphMode = Literal["staged_bclt", "fixed_state_nonlinear", "full"]
IersPolicy = Literal["warn", "strict"]

_GRAPH_MODES = frozenset({"staged_bclt", "fixed_state_nonlinear", "full"})
_IERS_POLICIES = frozenset({"warn", "strict"})


@dataclass(frozen=True)
class Tempo2NativeConfig:
    """Tempo2-native JAX graph and preflight settings for a timing session."""

    graph_mode: Tempo2GraphMode = "staged_bclt"
    iers_policy: IersPolicy = "warn"
    bclt_fixed_iter: int = 12
    force_cache_refresh: bool = False
    require_native_cache: bool = True


def normalize_tempo2_native(
    value: str | Tempo2NativeConfig | None,
    *,
    compatibility: str,
) -> Tempo2NativeConfig | None:
    """Normalize ``tempo2_native`` session argument."""
    if value is None:
        if str(compatibility).lower().startswith("tempo2"):
            return Tempo2NativeConfig()
        return None
    if isinstance(value, Tempo2NativeConfig):
        return value
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _GRAPH_MODES:
        allowed = ", ".join(sorted(_GRAPH_MODES))
        raise ValueError(f"Unknown tempo2_native={value!r}; expected one of {allowed}")
    return Tempo2NativeConfig(graph_mode=mode)  # type: ignore[arg-type]


def iers_strict_for_policy(policy: IersPolicy) -> bool:
    return policy == "strict"


__all__ = [
    "Tempo2NativeConfig",
    "Tempo2GraphMode",
    "IersPolicy",
    "normalize_tempo2_native",
    "iers_strict_for_policy",
]
