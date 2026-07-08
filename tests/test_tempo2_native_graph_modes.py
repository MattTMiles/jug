"""Tempo2-native config precedence over environment variables."""

from __future__ import annotations

import importlib
import warnings

import pytest

from jug.timing import Tempo2NativeConfig


def _reload_quarantine():
    import jug.residuals.tempo2_native_quarantine as q

    return importlib.reload(q)


def test_graph_mode_default_is_staged_bclt(monkeypatch):
    monkeypatch.delenv("JUG_TEMPO2_NATIVE_GRAPH_MODE", raising=False)
    q = _reload_quarantine()
    assert q.tempo2_native_graph_mode() == "staged_bclt"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fixed_state_nonlinear", "fixed_state_nonlinear"),
        ("staged_bclt", "staged_bclt"),
        ("full", "full"),
        ("staged-bclt", "staged_bclt"),
    ],
)
def test_graph_mode_env(monkeypatch, raw, expected):
    monkeypatch.setenv("JUG_TEMPO2_NATIVE_GRAPH_MODE", raw)
    q = _reload_quarantine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mode = q.tempo2_native_graph_mode()
    assert mode == expected
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    )


def test_graph_mode_config_overrides_env(monkeypatch):
    monkeypatch.setenv("JUG_TEMPO2_NATIVE_GRAPH_MODE", "full")
    q = _reload_quarantine()
    cfg = Tempo2NativeConfig(graph_mode="staged_bclt")
    assert q.tempo2_native_graph_mode(cfg) == "staged_bclt"


def test_graph_mode_invalid_raises(monkeypatch):
    monkeypatch.setenv("JUG_TEMPO2_NATIVE_GRAPH_MODE", "full_ingraph")
    q = _reload_quarantine()
    with pytest.raises(ValueError, match="Unknown tempo2-native graph mode"):
        q.tempo2_native_graph_mode()
