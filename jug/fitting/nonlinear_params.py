"""Linearization / residual-path mode (caller-declared).

nonlinear_params chooses how residual deltas are computed. It does not choose
which parameters are free or sampled. JUG validates and executes; it does not
select a mode from the δ-axis list or model type.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NONLINEAR_PARAMS_BINARY = "binary"
NONLINEAR_PARAMS_BINARY_PLUS = "binary+"

_NONLINEAR_PARAMS_MODES = frozenset(
    {
        NONLINEAR_PARAMS_BINARY,
        NONLINEAR_PARAMS_BINARY_PLUS,
    }
)


def nonlinear_params_allowed_strings() -> tuple[str, ...]:
    """Closed mode strings (excludes None)."""
    return tuple(sorted(_NONLINEAR_PARAMS_MODES))


def validate_nonlinear_params(value: str | None) -> str | None:
    """Return None or a normalized mode string; raise ValueError otherwise."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Unknown nonlinear_params={value!r}; expected None or one of "
            f"{', '.join(repr(s) for s in nonlinear_params_allowed_strings())}"
        )
    normalized = str(value).strip().lower()
    # accept "binary+" as written; lower() leaves '+' intact
    if normalized not in _NONLINEAR_PARAMS_MODES:
        allowed = ", ".join(repr(s) for s in nonlinear_params_allowed_strings())
        raise ValueError(
            f"Unknown nonlinear_params={value!r}; expected None or one of {allowed}"
        )
    return normalized


def is_hybrid_nonlinear_params(value: str | None) -> bool:
    return validate_nonlinear_params(value) is not None


def plan_live_keys(mode: str | None) -> frozenset[str]:
    """Keys that remain live inside the hybrid binary/Kopeikin plan call."""
    resolved = validate_nonlinear_params(mode)
    if resolved == NONLINEAR_PARAMS_BINARY_PLUS:
        return frozenset({"PX"})
    return frozenset()


def warn_if_tempo2_native_ignored(
    nonlinear_params: str | None,
    tempo2_native: str | None,
) -> None:
    """Log when hybrid ignores a non-default tempo2_native setting."""
    mode = validate_nonlinear_params(nonlinear_params)
    if mode is None:
        return
    resolved = None if tempo2_native is None else str(tempo2_native).strip().lower()
    if resolved is None or resolved == "fixed_state_stripped":
        return
    logger.warning(
        "nonlinear_params=%r ignores tempo2_native=%r for residual_delta_jax "
        "(host cache build may still use it).",
        mode,
        tempo2_native,
    )
