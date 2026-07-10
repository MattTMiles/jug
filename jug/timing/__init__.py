"""User-facing timing configuration for JUG sessions and fit exports."""

from __future__ import annotations

from typing import Any, Literal

Tempo2GraphMode = Literal[
    "staged_bclt", "fixed_state_bclt", "fixed_state_stripped", "full"
]
IersPolicy = Literal["warn", "strict"]

_IERS_POLICIES = frozenset({"warn", "strict"})
_DEFAULT_GRAPH_MODE: Tempo2GraphMode = "staged_bclt"

DEFAULT_TEMPO2_JUG_OPTIONS: dict[str, Any] = {
    "iers_policy": "warn",
    "bclt_fixed_iter": 12,
    "force_cache_refresh": False,
    "require_native_cache": True,
}
_TEMPO2_JUG_OPTION_KEYS = frozenset(DEFAULT_TEMPO2_JUG_OPTIONS)


def validate_tempo2_graph_mode(value: str) -> str:
    """Normalize and validate a tempo2-native JAX graph mode string.

    Delegates to :func:`jug.residuals.tempo2.graph_config.tempo2_graph_mode`
    (single source of truth).  Only canonical mode strings are accepted.
    """
    from jug.residuals.tempo2.graph_config import tempo2_graph_mode

    return tempo2_graph_mode(value)


def validate_tempo2_iers_policy(value: str) -> str:
    """Normalize and validate an IERS preflight policy string."""
    policy = str(value).strip().lower()
    if policy not in _IERS_POLICIES:
        allowed = ", ".join(sorted(_IERS_POLICIES))
        raise ValueError(f"Unknown iers_policy={value!r}; expected one of {allowed}")
    return policy


def validate_tempo2_bclt_fixed_iter(value: int) -> int:
    """Validate the fixed-length BCLT scan iteration count for JAX AD."""
    try:
        n_iter = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bclt_fixed_iter must be a positive integer; got {value!r}"
        ) from exc
    if n_iter < 1:
        raise ValueError(f"bclt_fixed_iter must be >= 1; got {n_iter}")
    return n_iter


def resolve_tempo2_jug_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user ``tempo2_jug_options`` with :data:`DEFAULT_TEMPO2_JUG_OPTIONS`."""
    resolved = dict(DEFAULT_TEMPO2_JUG_OPTIONS)
    if options is None:
        return resolved
    if not isinstance(options, dict):
        raise TypeError(
            f"tempo2_jug_options must be a dict or None; got {type(options).__name__}"
        )
    unknown = set(options) - _TEMPO2_JUG_OPTION_KEYS
    if unknown:
        allowed = ", ".join(sorted(_TEMPO2_JUG_OPTION_KEYS))
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown tempo2_jug_options key(s): {bad}; expected {allowed}")
    if "iers_policy" in options:
        resolved["iers_policy"] = validate_tempo2_iers_policy(options["iers_policy"])
    if "bclt_fixed_iter" in options:
        resolved["bclt_fixed_iter"] = validate_tempo2_bclt_fixed_iter(
            options["bclt_fixed_iter"]
        )
    if "force_cache_refresh" in options:
        resolved["force_cache_refresh"] = bool(options["force_cache_refresh"])
    if "require_native_cache" in options:
        resolved["require_native_cache"] = bool(options["require_native_cache"])
    return resolved


def resolve_tempo2_session_args(
    compatibility: str,
    tempo2_native: str | None,
    tempo2_jug_options: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve ``(graph_mode, options)`` from session timing kwargs.

    When ``compatibility`` is tempo2 and ``tempo2_native`` is omitted, the
    default graph mode is ``staged_bclt``.  For non-tempo2 sessions,
    ``graph_mode`` is ``None`` unless ``tempo2_native`` is set explicitly.
    """
    from jug.residuals.engine_conventions import normalize_compatibility_mode

    options = resolve_tempo2_jug_options(tempo2_jug_options)
    is_tempo2 = normalize_compatibility_mode(compatibility) == "tempo2"

    if tempo2_native is None:
        if is_tempo2:
            return validate_tempo2_graph_mode(_DEFAULT_GRAPH_MODE), options
        return None, options

    return validate_tempo2_graph_mode(tempo2_native), options


__all__ = [
    "DEFAULT_TEMPO2_JUG_OPTIONS",
    "Tempo2GraphMode",
    "IersPolicy",
    "validate_tempo2_graph_mode",
    "validate_tempo2_iers_policy",
    "validate_tempo2_bclt_fixed_iter",
    "resolve_tempo2_jug_options",
    "resolve_tempo2_session_args",
]