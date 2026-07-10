"""Ephemeris-name resolution to on-disk kernels (offline)."""

from __future__ import annotations

import os

from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path
from jug.residuals.simple_calculator import (
    _bundled_ephemeris_path,
    _resolve_ephemeris,
)


def test_de440_resolves_to_bundled_kernel():
    """DE440 must resolve to the shipped ``de440s.bsp`` without network access.

    The tempo2 geometry path (jplephem) needs an on-disk SPK, and the DE440
    design-matrix fixtures fail at setup if DE440 cannot be resolved offline.
    """
    for name in ("de440", "DE440", "de440s"):
        resolved = _resolve_ephemeris(name)
        assert isinstance(resolved, str)
        assert os.path.isfile(resolved), f"{name} -> {resolved} is not a file"
        assert resolved.endswith("de440s.bsp")


def test_resolve_tempo2_ephemeris_path_de440_is_ondisk():
    path = resolve_tempo2_ephemeris_path("DE440")
    assert os.path.isfile(path)
    assert path.endswith("de440s.bsp")


def test_bundled_lookup_returns_none_for_unbundled_names():
    assert _bundled_ephemeris_path("de405") is None
    assert _bundled_ephemeris_path("not-an-ephemeris") is None
