"""Tests for jug.engine.diagnostics — noise/backend diagnostics report."""

import pytest

from jug.engine.diagnostics import build_noise_diagnostics, format_noise_report
from jug.engine.flag_mapping import FlagMappingConfig
from jug.noise.white import WhiteNoiseEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_toa_flags(n=20):
    """Simulate a MeerKAT-like dataset with two pseudo-backends."""
    flags = []
    for i in range(n):
        if i < 12:
            flags.append({"fe": "KAT", "be": "MKBF"})
        else:
            flags.append({"fe": "KAT", "be": "PTUSE"})
    return flags


def _sample_noise_entries():
    return [
        WhiteNoiseEntry("EFAC", "be", "MKBF", 1.1),
        WhiteNoiseEntry("EFAC", "be", "PTUSE", 0.9),
        WhiteNoiseEntry("EQUAD", "be", "MKBF", 0.5),
        WhiteNoiseEntry("ECORR", "be", "MKBF", 0.2),
    ]


# ---------------------------------------------------------------------------
# build_noise_diagnostics
# ---------------------------------------------------------------------------

class TestBuildNoiseDiagnostics:
    def test_basic_structure(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)

        assert diag["n_toas"] == 20
        assert "backends" in diag
        assert "noise_entries" in diag
        assert "unmatched_toas" in diag
        assert "effective_coverage" in diag
        assert "override_semantics" in diag

    def test_backend_counts(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)

        # Default config resolves "be" as second candidate
        # (first is "f" which is not present)
        assert diag["backends"].get("MKBF", 0) == 12
        assert diag["backends"].get("PTUSE", 0) == 8

    def test_noise_entry_matching(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)

        # EFAC -be MKBF should match 12 TOAs
        efac_mkbf = [
            e for e in diag["noise_entries"]
            if e["kind"] == "EFAC" and e["flag_value"] == "MKBF"
        ]
        assert len(efac_mkbf) == 1
        assert efac_mkbf[0]["matched_count"] == 12

        # EFAC -be PTUSE should match 8 TOAs
        efac_ptuse = [
            e for e in diag["noise_entries"]
            if e["kind"] == "EFAC" and e["flag_value"] == "PTUSE"
        ]
        assert len(efac_ptuse) == 1
        assert efac_ptuse[0]["matched_count"] == 8

    def test_effective_coverage(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)

        cov = diag["effective_coverage"]
        # EFAC: MKBF(12) + PTUSE(8) = 20
        assert cov["any_efac_count"] == 20
        # EQUAD: only MKBF(12)
        assert cov["any_equad_count"] == 12
        # ECORR: only MKBF(12)
        assert cov["any_ecorr_count"] == 12

    def test_unmatched_toas(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)

        # EQUAD -be MKBF: 8 unmatched (PTUSE TOAs)
        key = "EQUAD_be_MKBF"
        assert len(diag["unmatched_toas"][key]) == 8

    def test_empty_entries(self):
        flags = _sample_toa_flags()
        diag = build_noise_diagnostics(flags, [])
        assert diag["noise_entries"] == []
        cov = diag["effective_coverage"]
        assert cov["any_efac_count"] == 0

    def test_empty_toas(self):
        diag = build_noise_diagnostics([], _sample_noise_entries())
        assert diag["n_toas"] == 0
        for e in diag["noise_entries"]:
            assert e["matched_count"] == 0

    def test_with_custom_config(self):
        # Use "fe" as the backend key instead of "be"
        cfg = FlagMappingConfig(candidates=["fe"])
        flags = _sample_toa_flags()
        diag = build_noise_diagnostics(flags, _sample_noise_entries(), config=cfg)
        # All TOAs have fe=KAT
        assert diag["backends"].get("KAT", 0) == 20


# ---------------------------------------------------------------------------
# format_noise_report
# ---------------------------------------------------------------------------

class TestFormatNoiseReport:
    def test_report_is_string(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)
        report = format_noise_report(diag)
        assert isinstance(report, str)

    def test_report_contains_key_info(self):
        flags = _sample_toa_flags()
        entries = _sample_noise_entries()
        diag = build_noise_diagnostics(flags, entries)
        report = format_noise_report(diag)

        assert "Total TOAs: 20" in report
        assert "MKBF" in report
        assert "PTUSE" in report
        assert "EFAC" in report
        assert "EQUAD" in report
        assert "ECORR" in report
        assert "Effective coverage" in report

    def test_report_fully_matched(self):
        """When all entries match all TOAs, report should say so."""
        flags = [{"be": "ALL"}] * 5
        entries = [WhiteNoiseEntry("EFAC", "be", "ALL", 1.0)]
        diag = build_noise_diagnostics(flags, entries)
        report = format_noise_report(diag)
        assert "All entries fully matched" in report

    def test_report_unmatched(self):
        flags = [{"be": "A"}] * 3 + [{"be": "B"}] * 3
        entries = [WhiteNoiseEntry("EFAC", "be", "A", 1.0)]
        diag = build_noise_diagnostics(flags, entries)
        report = format_noise_report(diag)
        assert "unmatched" in report.lower()


# ---------------------------------------------------------------------------
# Integration with real data patterns
# ---------------------------------------------------------------------------

class TestRealPatterns:
    def test_nanograv_style(self):
        """NANOGrav uses -f flag with values like 'L-wide.PUPPI'."""
        flags = [
            {"f": "L-wide.PUPPI"} for _ in range(10)
        ] + [
            {"f": "430.ASP"} for _ in range(5)
        ]
        entries = [
            WhiteNoiseEntry("EFAC", "f", "L-wide.PUPPI", 1.2),
            WhiteNoiseEntry("EFAC", "f", "430.ASP", 0.8),
            WhiteNoiseEntry("EQUAD", "f", "L-wide.PUPPI", 0.3),
        ]
        diag = build_noise_diagnostics(flags, entries)

        # Check all EFAC coverage
        assert diag["effective_coverage"]["any_efac_count"] == 15
        assert diag["effective_coverage"]["any_equad_count"] == 10
