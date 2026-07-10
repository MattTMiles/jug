"""Tempo2 graph mode selector and session validation."""

from __future__ import annotations

import pytest

from jug.residuals.tempo2.graph_config import (
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
    TEMPO2_GRAPH_FULL,
    TEMPO2_GRAPH_STAGED_BCLT,
    tempo2_graph_mode,
    tempo2_graph_mode_allowed_strings,
)
from jug.timing import validate_tempo2_graph_mode


def test_graph_mode_default_is_staged_bclt():
    assert tempo2_graph_mode() == TEMPO2_GRAPH_STAGED_BCLT
    assert tempo2_graph_mode(None) == TEMPO2_GRAPH_STAGED_BCLT


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fixed_state_bclt", TEMPO2_GRAPH_FIXED_STATE_BCLT),
        ("fixed_state_stripped", TEMPO2_GRAPH_FIXED_STATE_STRIPPED),
        ("staged_bclt", TEMPO2_GRAPH_STAGED_BCLT),
        ("full", TEMPO2_GRAPH_FULL),
        ("STAGED_BCLT", TEMPO2_GRAPH_STAGED_BCLT),
    ],
)
def test_graph_mode_explicit_strings(raw, expected):
    assert tempo2_graph_mode(raw) == expected


def test_graph_mode_invalid_raises():
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        tempo2_graph_mode("full_ingraph")


def test_legacy_fixed_state_nonlinear_string_rejected():
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        tempo2_graph_mode("fixed_state_nonlinear")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fixed_state_bclt", TEMPO2_GRAPH_FIXED_STATE_BCLT),
        ("fixed_state_stripped", TEMPO2_GRAPH_FIXED_STATE_STRIPPED),
        ("staged_bclt", TEMPO2_GRAPH_STAGED_BCLT),
        ("full", TEMPO2_GRAPH_FULL),
    ],
)
def test_validate_tempo2_graph_mode_matches_graph_config(raw, expected):
    assert validate_tempo2_graph_mode(raw) == expected
    assert validate_tempo2_graph_mode(raw) == tempo2_graph_mode(raw)


def test_validate_tempo2_graph_mode_invalid_matches_graph_config():
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        validate_tempo2_graph_mode("not_a_mode")
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        tempo2_graph_mode("not_a_mode")


def test_allowed_strings_are_canonical_only():
    assert tempo2_graph_mode_allowed_strings() == (
        TEMPO2_GRAPH_FIXED_STATE_BCLT,
        TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
        TEMPO2_GRAPH_FULL,
        TEMPO2_GRAPH_STAGED_BCLT,
    )


def test_session_resolve_stores_canonical_mode():
    from jug.timing import resolve_tempo2_session_args

    mode, _opts = resolve_tempo2_session_args("tempo2", "fixed_state_bclt", None)
    assert mode == TEMPO2_GRAPH_FIXED_STATE_BCLT