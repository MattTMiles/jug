"""Tests for jug.engine.validation -- TOA data integrity checks."""

import numpy as np
import pytest

from jug.engine.validation import (
    Severity,
    ValidationIssue,
    validate_toas,
    validate_toas_from_simple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_data(n=10):
    """Return a valid TOA dataset with *n* entries."""
    mjd = np.linspace(58000.0, 59000.0, n)
    freq = np.full(n, 1284.0)
    err = np.full(n, 1.0)
    flags = [{"be": "MKBF", "fe": "KAT"}] * n
    return mjd, freq, err, flags


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

class TestValidateToasCleanData:
    """No issues for well-formed data."""

    def test_clean_data_no_issues(self):
        mjd, freq, err, flags = _good_data()
        issues = validate_toas(mjd, freq, err, flags)
        assert issues == []

    def test_clean_data_no_flags(self):
        mjd, freq, err, _ = _good_data()
        issues = validate_toas(mjd, freq, err)
        assert issues == []

    def test_single_toa(self):
        issues = validate_toas(
            np.array([58000.0]),
            np.array([1284.0]),
            np.array([1.0]),
        )
        assert issues == []


# ---------------------------------------------------------------------------
# NaN / Inf
# ---------------------------------------------------------------------------

class TestNanInf:
    def test_nan_mjd(self):
        mjd, freq, err, flags = _good_data()
        mjd[3] = np.nan
        issues = validate_toas(mjd, freq, err, flags)
        errors = [i for i in issues if i.code == "NAN_MJD"]
        assert len(errors) == 1
        assert 3 in errors[0].indices

    def test_inf_freq(self):
        mjd, freq, err, flags = _good_data()
        freq[0] = np.inf
        issues = validate_toas(mjd, freq, err, flags)
        assert any(i.code == "NAN_FREQ" for i in issues)

    def test_nan_error(self):
        mjd, freq, err, flags = _good_data()
        err[5] = np.nan
        issues = validate_toas(mjd, freq, err, flags)
        assert any(i.code == "NAN_ERROR" for i in issues)


# ---------------------------------------------------------------------------
# Non-positive values
# ---------------------------------------------------------------------------

class TestNonPositive:
    def test_zero_freq(self):
        mjd, freq, err, flags = _good_data()
        freq[2] = 0.0
        issues = validate_toas(mjd, freq, err, flags)
        codes = {i.code for i in issues}
        assert "NONPOSITIVE_FREQ" in codes

    def test_negative_error(self):
        mjd, freq, err, flags = _good_data()
        err[0] = -0.5
        issues = validate_toas(mjd, freq, err, flags)
        assert any(i.code == "NONPOSITIVE_ERROR" for i in issues)

    def test_zero_error(self):
        mjd, freq, err, flags = _good_data()
        err[7] = 0.0
        issues = validate_toas(mjd, freq, err, flags)
        assert any(i.code == "NONPOSITIVE_ERROR" for i in issues)


# ---------------------------------------------------------------------------
# Duplicated MJDs
# ---------------------------------------------------------------------------

class TestDuplicateMJD:
    def test_exact_duplicate(self):
        mjd, freq, err, flags = _good_data()
        mjd[4] = mjd[3]
        issues = validate_toas(mjd, freq, err, flags)
        assert any(i.code == "DUPLICATE_MJD" for i in issues)
        dup = [i for i in issues if i.code == "DUPLICATE_MJD"][0]
        assert 3 in dup.indices and 4 in dup.indices

    def test_no_false_positive_for_close_but_distinct(self):
        """MJDs that differ by more than tolerance should not be flagged."""
        mjd, freq, err, _ = _good_data()
        # Ensure spacing is > 1e-15 days (it is: ~111 days apart)
        issues = validate_toas(mjd, freq, err)
        assert not any(i.code == "DUPLICATE_MJD" for i in issues)


# ---------------------------------------------------------------------------
# Missing flags
# ---------------------------------------------------------------------------

class TestMissingFlags:
    def test_missing_required_flag(self):
        mjd, freq, err, flags = _good_data(5)
        flags = [{"be": "MKBF"}] * 5  # missing 'fe'
        issues = validate_toas(mjd, freq, err, flags, required_flags=["fe"])
        assert any(i.code == "MISSING_FLAG" for i in issues)

    def test_all_flags_present(self):
        mjd, freq, err, flags = _good_data(5)
        issues = validate_toas(mjd, freq, err, flags, required_flags=["be", "fe"])
        assert not any(i.code == "MISSING_FLAG" for i in issues)

    def test_partial_missing(self):
        """Only some TOAs missing a flag."""
        mjd, freq, err, _ = _good_data(4)
        flags = [
            {"be": "MKBF", "fe": "KAT"},
            {"be": "MKBF"},  # missing fe
            {"be": "MKBF", "fe": "KAT"},
            {"be": "MKBF"},  # missing fe
        ]
        issues = validate_toas(mjd, freq, err, flags, required_flags=["fe"])
        flag_issue = [i for i in issues if i.code == "MISSING_FLAG"][0]
        assert set(flag_issue.indices) == {1, 3}


# ---------------------------------------------------------------------------
# Length mismatch
# ---------------------------------------------------------------------------

class TestLengthMismatch:
    def test_freq_length_mismatch(self):
        issues = validate_toas(
            np.array([58000.0, 58001.0]),
            np.array([1284.0]),  # too short
            np.array([1.0, 1.0]),
        )
        assert any(i.code == "LENGTH_MISMATCH" for i in issues)


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_raises_on_error(self):
        mjd, freq, err, _ = _good_data()
        mjd[0] = np.nan
        with pytest.raises(ValueError, match="NAN_MJD"):
            validate_toas(mjd, freq, err, strict=True)

    def test_strict_ok_with_warnings_only(self):
        """Warnings don't raise in strict mode."""
        mjd, freq, err, _ = _good_data()
        mjd[2] = mjd[1]  # duplicate -> warning
        # Should NOT raise
        issues = validate_toas(mjd, freq, err, strict=True)
        assert any(i.severity == Severity.WARNING for i in issues)


# ---------------------------------------------------------------------------
# Ordering: errors before warnings
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_errors_before_warnings(self):
        mjd, freq, err, _ = _good_data()
        mjd[0] = np.nan  # error
        mjd[3] = mjd[2]  # warning (duplicate)
        issues = validate_toas(mjd, freq, err)
        assert len(issues) >= 2
        assert issues[0].severity == Severity.ERROR


# ---------------------------------------------------------------------------
# SimpleTOA wrapper
# ---------------------------------------------------------------------------

class TestSimpleTOAWrapper:
    def test_from_simple_toas(self):
        from jug.io.tim_reader import SimpleTOA

        toas = [
            SimpleTOA("58000.0", 58000, 0.0, 1284.0, 1.0, "meerkat", {"be": "MKBF"}),
            SimpleTOA("58001.0", 58001, 0.0, 1284.0, -0.5, "meerkat", {"be": "MKBF"}),
        ]
        issues = validate_toas_from_simple(toas)
        assert any(i.code == "NONPOSITIVE_ERROR" for i in issues)


# ---------------------------------------------------------------------------
# Issue __str__
# ---------------------------------------------------------------------------

class TestIssueStr:
    def test_str_format(self):
        issue = ValidationIssue(Severity.ERROR, "NAN_MJD", "test", (0, 1))
        s = str(issue)
        assert "ERROR" in s
        assert "NAN_MJD" in s
        assert "[0, 1]" in s
