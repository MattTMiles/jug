"""open_session forwards tempo2_native and tempo2_jug_options to TimingSession."""

from __future__ import annotations

import inspect

from jug.engine import open_session
from jug.timing import resolve_tempo2_jug_options, resolve_tempo2_session_args


def test_open_session_accepts_tempo2_native_kwarg():
    sig = inspect.signature(open_session)
    assert "tempo2_native" in sig.parameters


def test_tempo2_session_args_resolve_graph_mode_and_options():
    graph_mode, options = resolve_tempo2_session_args(
        "tempo2",
        tempo2_native="staged_bclt",
        tempo2_jug_options={"iers_policy": "strict", "bclt_fixed_iter": 8},
    )
    assert graph_mode == "staged_bclt"
    assert options["iers_policy"] == "strict"
    assert options["bclt_fixed_iter"] == 8


def test_tempo2_jug_options_defaults():
    options = resolve_tempo2_jug_options(None)
    assert options["bclt_fixed_iter"] == 12
    assert options["require_native_cache"] is True
