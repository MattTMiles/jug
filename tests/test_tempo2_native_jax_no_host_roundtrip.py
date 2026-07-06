"""DEV ORACLE — production native chain avoids mid-graph ``device_get``."""

from __future__ import annotations

import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from tempo2_native_test_helpers import compute_native_terms_for_fixture, load_wsrt167_fixture


def test_native_chain_no_device_get_mid_bclt(monkeypatch):
    fixture = load_wsrt167_fixture()
    calls = []

    orig = jax.device_get

    def tracked(x, *args, **kwargs):
        calls.append(1)
        return orig(x, *args, **kwargs)

    monkeypatch.setattr(jax, "device_get", tracked)
    compute_native_terms_for_fixture(fixture)
    assert len(calls) == 0
