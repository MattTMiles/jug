"""TOA data integrity checks.

Validates TOA tables for common issues before fitting:
- NaN/Inf values in MJD, frequency, or uncertainty
- Missing or non-positive observing frequencies
- Non-positive TOA uncertainties
- Duplicated MJDs (exact duplicates within a tolerance)
- Missing required flags

Provides both a list of warnings/errors and an optional strict mode
that raises on the first error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Severity level for validation issues."""
    WARNING = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Validation issue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue found in TOA data.

    Attributes
    ----------
    severity : Severity
        WARNING or ERROR.
    code : str
        Short machine-readable code (e.g. ``'NAN_MJD'``).
    message : str
        Human-readable description.
    indices : tuple of int
        TOA indices involved (may be empty for global issues).
    """
    severity: Severity
    code: str
    message: str
    indices: tuple = ()

    def __str__(self) -> str:
        prefix = "ERROR" if self.severity == Severity.ERROR else "WARNING"
        idx_str = f" [TOA indices: {list(self.indices)}]" if self.indices else ""
        return f"{prefix}: {self.code} -- {self.message}{idx_str}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_toas(
    mjd_values: np.ndarray,
    freq_mhz: np.ndarray,
    errors_us: np.ndarray,
    flags: Optional[Sequence[Dict[str, str]]] = None,
    *,
    required_flags: Optional[Sequence[str]] = None,
    dup_mjd_tol_days: float = 1e-15,
    strict: bool = False,
) -> List[ValidationIssue]:
    """Validate a TOA table for common data-quality issues.

    Parameters
    ----------
    mjd_values : np.ndarray, shape (n,)
        MJD values (float64 or longdouble).
    freq_mhz : np.ndarray, shape (n,)
        Observing frequencies in MHz.
    errors_us : np.ndarray, shape (n,)
        TOA uncertainties in microseconds.
    flags : sequence of dict, optional
        Per-TOA flag dictionaries.
    required_flags : sequence of str, optional
        Flag keys that every TOA must have (e.g. ``['be', 'fe']``).
    dup_mjd_tol_days : float, default 1e-15
        Two MJDs closer than this are flagged as duplicates.
    strict : bool, default False
        If True, raise ``ValueError`` on the first ERROR-level issue.

    Returns
    -------
    list of ValidationIssue
        All issues found, ordered by severity (errors first).

    Raises
    ------
    ValueError
        If *strict* is True and an ERROR-level issue is detected.
    """
    issues: List[ValidationIssue] = []
    n = len(mjd_values)

    mjd_values = np.asarray(mjd_values, dtype=np.float64)
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)

    # --- Array length consistency ----------------------------------------
    if len(freq_mhz) != n:
        issues.append(ValidationIssue(
            Severity.ERROR, "LENGTH_MISMATCH",
            f"freq_mhz length ({len(freq_mhz)}) != mjd length ({n})",
        ))
    if len(errors_us) != n:
        issues.append(ValidationIssue(
            Severity.ERROR, "LENGTH_MISMATCH",
            f"errors_us length ({len(errors_us)}) != mjd length ({n})",
        ))
    if flags is not None and len(flags) != n:
        issues.append(ValidationIssue(
            Severity.ERROR, "LENGTH_MISMATCH",
            f"flags length ({len(flags)}) != mjd length ({n})",
        ))

    # --- NaN / Inf in MJD ------------------------------------------------
    bad_mjd = np.where(~np.isfinite(mjd_values))[0]
    if len(bad_mjd) > 0:
        issues.append(ValidationIssue(
            Severity.ERROR, "NAN_MJD",
            f"{len(bad_mjd)} TOA(s) have NaN/Inf MJD",
            tuple(bad_mjd.tolist()),
        ))

    # --- NaN / Inf in frequency ------------------------------------------
    bad_freq = np.where(~np.isfinite(freq_mhz))[0]
    if len(bad_freq) > 0:
        issues.append(ValidationIssue(
            Severity.ERROR, "NAN_FREQ",
            f"{len(bad_freq)} TOA(s) have NaN/Inf frequency",
            tuple(bad_freq.tolist()),
        ))

    # --- Non-positive frequency ------------------------------------------
    nonpos_freq = np.where(freq_mhz <= 0)[0]
    if len(nonpos_freq) > 0:
        issues.append(ValidationIssue(
            Severity.ERROR, "NONPOSITIVE_FREQ",
            f"{len(nonpos_freq)} TOA(s) have non-positive frequency",
            tuple(nonpos_freq.tolist()),
        ))

    # --- NaN / Inf in uncertainty ----------------------------------------
    bad_err = np.where(~np.isfinite(errors_us))[0]
    if len(bad_err) > 0:
        issues.append(ValidationIssue(
            Severity.ERROR, "NAN_ERROR",
            f"{len(bad_err)} TOA(s) have NaN/Inf uncertainty",
            tuple(bad_err.tolist()),
        ))

    # --- Non-positive uncertainty ----------------------------------------
    nonpos_err = np.where(errors_us <= 0)[0]
    if len(nonpos_err) > 0:
        issues.append(ValidationIssue(
            Severity.ERROR, "NONPOSITIVE_ERROR",
            f"{len(nonpos_err)} TOA(s) have non-positive uncertainty",
            tuple(nonpos_err.tolist()),
        ))

    # --- Duplicated MJDs -------------------------------------------------
    if n > 1:
        sorted_idx = np.argsort(mjd_values)
        sorted_mjd = mjd_values[sorted_idx]
        diffs = np.diff(sorted_mjd)
        dup_mask = np.abs(diffs) < dup_mjd_tol_days
        if np.any(dup_mask):
            dup_positions = np.where(dup_mask)[0]
            # Map back to original indices
            dup_orig = set()
            for p in dup_positions:
                dup_orig.add(int(sorted_idx[p]))
                dup_orig.add(int(sorted_idx[p + 1]))
            issues.append(ValidationIssue(
                Severity.WARNING, "DUPLICATE_MJD",
                f"{len(dup_orig)} TOA(s) have MJDs within tolerance "
                f"({dup_mjd_tol_days} days)",
                tuple(sorted(dup_orig)),
            ))

    # --- Missing required flags ------------------------------------------
    if flags is not None and required_flags:
        for flag_key in required_flags:
            missing_idx = [
                i for i, f in enumerate(flags)
                if flag_key not in f
            ]
            if missing_idx:
                issues.append(ValidationIssue(
                    Severity.WARNING, "MISSING_FLAG",
                    f"{len(missing_idx)} TOA(s) missing flag '{flag_key}'",
                    tuple(missing_idx),
                ))

    # --- Sort: errors first, then warnings -------------------------------
    issues.sort(key=lambda x: (0 if x.severity == Severity.ERROR else 1, x.code))

    # --- Strict mode: raise on first error -------------------------------
    if strict:
        for issue in issues:
            if issue.severity == Severity.ERROR:
                raise ValueError(str(issue))

    return issues


def validate_toas_from_simple(
    toas,
    *,
    required_flags: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> List[ValidationIssue]:
    """Convenience wrapper: validate from a list of SimpleTOA objects.

    Parameters
    ----------
    toas : list of SimpleTOA
        TOA objects from ``parse_tim_file_mjds()``.
    required_flags : sequence of str, optional
        Flag keys that every TOA must have.
    strict : bool, default False
        Raise on first ERROR.

    Returns
    -------
    list of ValidationIssue
    """
    mjd = np.array([t.mjd_int + t.mjd_frac for t in toas], dtype=np.float64)
    freq = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    err = np.array([t.error_us for t in toas], dtype=np.float64)
    flags = [t.flags for t in toas]

    return validate_toas(
        mjd, freq, err, flags,
        required_flags=required_flags,
        strict=strict,
    )
