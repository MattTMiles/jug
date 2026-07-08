"""open_session forwards tempo2_native to TimingSession."""

from __future__ import annotations

import inspect

from jug.engine import open_session
from jug.timing import Tempo2NativeConfig


def test_open_session_accepts_tempo2_native_kwarg():
    sig = inspect.signature(open_session)
    assert "tempo2_native" in sig.parameters


def test_tempo2_native_config_fields():
    cfg = Tempo2NativeConfig(
        graph_mode="staged_bclt",
        bclt_fixed_iter=8,
        require_native_cache=False,
    )
    assert cfg.bclt_fixed_iter == 8
    assert cfg.require_native_cache is False
