"""Tests for binary_chart_facts (eccentricity-vector reparameterization facts)."""

from __future__ import annotations

from jug.fitting.binary_delay_plan import BinaryChartFacts, binary_chart_facts

_BASE = {"A1": 1.0, "PB": 8.0, "ECC": 8e-4, "OM": 50.7, "T0": 55000.0}


def test_dd_no_secular_is_epoch_shift_exact():
    facts = binary_chart_facts({**_BASE, "BINARY": "DD"}, [])
    assert isinstance(facts, BinaryChartFacts)
    assert facts.convention_family == "dd"
    assert facts.epoch_shift_exact is True
    assert facts.secular_terms == ()


def test_explicit_secular_rate_breaks_epoch_shift():
    facts = binary_chart_facts({**_BASE, "BINARY": "DD", "OMDOT": 1.2e-3}, [])
    assert facts.epoch_shift_exact is False
    assert "OMDOT" in facts.secular_terms


def test_fitted_secular_rate_breaks_epoch_shift():
    facts = binary_chart_facts({**_BASE, "BINARY": "DD"}, ["OMDOT"])
    assert facts.epoch_shift_exact is False
    assert "OMDOT" in facts.secular_terms


def test_ddgr_derives_secular_rates_internally():
    # DDGR computes OMDOT/PBDOT from the masses; they are not explicit params,
    # so the resolved family alone would look epoch-shift exact — the original
    # BINARY name must be consulted.
    facts = binary_chart_facts({**_BASE, "BINARY": "DDGR"}, [])
    assert facts.convention_family == "dd"
    assert facts.epoch_shift_exact is False
    assert {"OMDOT", "PBDOT"} <= set(facts.secular_terms)


def test_bt_is_dd_convention():
    facts = binary_chart_facts({**_BASE, "BINARY": "BT"}, [])
    assert facts.convention_family == "dd"
    assert facts.epoch_shift_exact is True


def test_t2_reduces_to_dd_when_kepler_parameterized():
    # A T2 par with ECC/OM/T0 (no EPS1/EPS2, no KIN/KOM) resolves to DD.
    facts = binary_chart_facts({**_BASE, "BINARY": "T2"}, [])
    assert facts.convention_family == "dd"


def test_no_binary_returns_none():
    assert binary_chart_facts({"F0": 100.0}, []) is None
