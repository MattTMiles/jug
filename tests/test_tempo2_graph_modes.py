"""Tempo2 graph mode selector."""

from __future__ import annotations

import pytest

from jug.residuals.tempo2.graph_config import tempo2_graph_mode


def test_graph_mode_default_is_staged_bclt():
    assert tempo2_graph_mode() == "staged_bclt"
    assert tempo2_graph_mode(None) == "staged_bclt"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fixed_state_nonlinear", "fixed_state_nonlinear"),
        ("staged_bclt", "staged_bclt"),
        ("full", "full"),
        ("STAGED_BCLT", "staged_bclt"),
    ],
)
def test_graph_mode_explicit_strings(raw, expected):
    assert tempo2_graph_mode(raw) == expected


def test_graph_mode_invalid_raises():
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        tempo2_graph_mode("full_ingraph")
