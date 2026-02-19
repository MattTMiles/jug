"""Flag aliasing and backend resolution for TOA data.

Provides a configurable mapping from TOA flags to semantic identifiers
(e.g. ``"backend"``).  This decouples noise models and grouping logic
from the raw flag names used in ``.tim`` files, which vary across
observatories and data releases.

Design
------
* Setup-time / Python only — not on the hot path.
* A ``FlagMappingConfig`` stores ordered candidate flag keys and an
  optional explicit alias table.
* ``resolve_backend_for_toa`` walks the candidates and returns the first
  match, or a configurable fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FlagMappingConfig:
    """Configuration for resolving a semantic flag from TOA flags.

    Attributes
    ----------
    candidates : list of str
        Ordered list of flag keys to try (without leading ``'-'``).
        The first key found in a TOA's flags dict wins.
        Default: ``["f", "be", "backend"]``.
    aliases : dict of str → str, optional
        Explicit value aliases.  Applied *after* the candidate key is
        resolved.  Example: ``{"KAT": "MKBF"}`` maps the value
        ``"KAT"`` to ``"MKBF"``.
    fallback : str
        Value returned when no candidate key is present.
        Default: ``"__unknown__"``.
    """
    candidates: List[str] = field(
        default_factory=lambda: ["f", "be", "backend"]
    )
    aliases: Dict[str, str] = field(default_factory=dict)
    fallback: str = "__unknown__"


# Singleton default config (can be replaced at startup)
DEFAULT_BACKEND_CONFIG = FlagMappingConfig()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_backend_for_toa(
    toa_flags: Dict[str, str],
    config: Optional[FlagMappingConfig] = None,
) -> str:
    """Resolve a backend identifier for a single TOA.

    Walks the candidate flag keys in order and returns the first value
    found.  Applies aliases if configured.

    Parameters
    ----------
    toa_flags : dict
        Flag dictionary for one TOA (keys without leading dash).
    config : FlagMappingConfig, optional
        If None, uses ``DEFAULT_BACKEND_CONFIG``.

    Returns
    -------
    str
        Resolved backend identifier (may be the fallback).
    """
    if config is None:
        config = DEFAULT_BACKEND_CONFIG

    for key in config.candidates:
        if key in toa_flags:
            value = toa_flags[key]
            if isinstance(value, list):
                value = value[0]
            return config.aliases.get(value, value)

    return config.fallback


def resolve_backends(
    toa_flags_list: Sequence[Dict[str, str]],
    config: Optional[FlagMappingConfig] = None,
) -> List[str]:
    """Resolve backend identifiers for all TOAs.

    Parameters
    ----------
    toa_flags_list : sequence of dict
        Per-TOA flag dictionaries.
    config : FlagMappingConfig, optional
        If None, uses ``DEFAULT_BACKEND_CONFIG``.

    Returns
    -------
    list of str
        Backend identifiers, one per TOA.
    """
    if config is None:
        config = DEFAULT_BACKEND_CONFIG
    return [resolve_backend_for_toa(f, config) for f in toa_flags_list]


def resolve_flag_for_toa(
    toa_flags: Dict[str, str],
    candidates: Sequence[str],
    aliases: Optional[Dict[str, str]] = None,
    fallback: str = "__unknown__",
) -> str:
    """Generic flag resolver — like ``resolve_backend_for_toa`` for any
    semantic concept (frequency band, receiver, etc.).

    Parameters
    ----------
    toa_flags : dict
        Flag dictionary for one TOA.
    candidates : sequence of str
        Ordered candidate flag keys.
    aliases : dict, optional
        Value aliases.
    fallback : str
        Returned when no candidate matches.

    Returns
    -------
    str
        Resolved flag value.
    """
    if aliases is None:
        aliases = {}
    for key in candidates:
        if key in toa_flags:
            value = toa_flags[key]
            if isinstance(value, list):
                value = value[0]
            return aliases.get(value, value)
    return fallback
