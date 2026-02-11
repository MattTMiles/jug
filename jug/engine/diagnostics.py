"""Noise / backend diagnostics report.

Generates a structured summary of how noise model entries map to TOAs,
which backends are present, and any unmatched entries.  Useful for
debugging par files and catching configuration issues before fitting.

Design
------
* Setup-time / Python only — not on the hot path.
* Consumes the parsed noise entries (``WhiteNoiseEntry``) and per-TOA flags.
* Returns both a human-readable text summary and a structured ``dict``
  so the GUI can display it programmatically.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from jug.engine.flag_mapping import (
    FlagMappingConfig,
    resolve_backends,
)
from jug.noise.white import WhiteNoiseEntry, build_backend_mask


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------

def build_noise_diagnostics(
    toa_flags: Sequence[Dict[str, str]],
    noise_entries: Sequence[WhiteNoiseEntry],
    config: Optional[FlagMappingConfig] = None,
) -> Dict:
    """Build a noise / backend diagnostics payload.

    Parameters
    ----------
    toa_flags : sequence of dict
        Per-TOA flag dictionaries (from ``SimpleTOA.flags``).
    noise_entries : sequence of WhiteNoiseEntry
        Parsed EFAC / EQUAD / ECORR entries.
    config : FlagMappingConfig, optional
        Backend resolution config.  If None uses default.

    Returns
    -------
    dict
        Structured diagnostics with keys:

        ``"n_toas"`` : int
            Total number of TOAs.
        ``"backends"`` : dict[str, int]
            Backend IDs detected and their TOA counts.
        ``"noise_entries"`` : list of dict
            Per-entry diagnostics with keys:
            ``"kind"``, ``"flag_name"``, ``"flag_value"``, ``"value"``,
            ``"matched_count"``, ``"matched_indices"`` (list of int).
        ``"unmatched_toas"`` : dict[str, list of int]
            Per-noise-entry key → list of *unmatched* TOA indices.
        ``"override_semantics"`` : str
            Description of how overlapping entries are resolved.
        ``"effective_coverage"`` : dict
            ``"any_efac_count"``, ``"any_equad_count"``, ``"any_ecorr_count"`` —
            how many TOAs are covered by at least one entry of each kind.
    """
    n_toas = len(toa_flags)

    # --- Backends --------------------------------------------------------
    backends = resolve_backends(toa_flags, config)
    backend_counts: Dict[str, int] = dict(Counter(backends))

    # --- Per-entry matching ---------------------------------------------
    entry_diags: List[Dict] = []
    # Track per-kind coverage (union of all entries of that kind)
    efac_covered = set()
    equad_covered = set()
    ecorr_covered = set()

    unmatched_toas: Dict[str, List[int]] = {}

    for entry in noise_entries:
        mask = build_backend_mask(
            list(toa_flags), entry.flag_name, entry.flag_value
        )
        matched_idx = [int(i) for i, m in enumerate(mask) if m]
        unmatched_idx = [int(i) for i, m in enumerate(mask) if not m]

        entry_key = f"{entry.kind}_{entry.flag_name}_{entry.flag_value}"

        entry_diags.append({
            "kind": entry.kind,
            "flag_name": entry.flag_name,
            "flag_value": entry.flag_value,
            "value": entry.value,
            "matched_count": len(matched_idx),
            "matched_indices": matched_idx,
        })

        unmatched_toas[entry_key] = unmatched_idx

        kind_upper = entry.kind.upper()
        if kind_upper == "EFAC":
            efac_covered.update(matched_idx)
        elif kind_upper == "EQUAD":
            equad_covered.update(matched_idx)
        elif kind_upper == "ECORR":
            ecorr_covered.update(matched_idx)

    return {
        "n_toas": n_toas,
        "backends": backend_counts,
        "noise_entries": entry_diags,
        "unmatched_toas": unmatched_toas,
        "override_semantics": "last match wins (later entries override earlier ones for the same TOA)",
        "effective_coverage": {
            "any_efac_count": len(efac_covered),
            "any_equad_count": len(equad_covered),
            "any_ecorr_count": len(ecorr_covered),
        },
    }


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def format_noise_report(diag: Dict) -> str:
    """Format a diagnostics dict as a human-readable text report.

    Parameters
    ----------
    diag : dict
        Output of ``build_noise_diagnostics()``.

    Returns
    -------
    str
        Multi-line text report.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  Noise / Backend Diagnostics Report")
    lines.append("=" * 60)

    # Backends
    lines.append(f"\nTotal TOAs: {diag['n_toas']}")
    lines.append("\nBackends detected:")
    for be, count in sorted(diag["backends"].items()):
        lines.append(f"  {be:30s} {count:6d} TOAs")

    # Noise entries
    lines.append(f"\nNoise entries ({len(diag['noise_entries'])}):")
    for ed in diag["noise_entries"]:
        lines.append(
            f"  {ed['kind']:6s} -{ed['flag_name']} {ed['flag_value']:20s} "
            f"= {ed['value']:12.6g}  →  {ed['matched_count']} TOAs matched"
        )

    # Unmatched
    has_unmatched = False
    for key, idx_list in diag["unmatched_toas"].items():
        if idx_list:
            if not has_unmatched:
                lines.append("\nUnmatched TOAs per entry:")
                has_unmatched = True
            lines.append(f"  {key}: {len(idx_list)} unmatched")
    if not has_unmatched:
        lines.append("\nAll entries fully matched.")

    # Coverage
    cov = diag["effective_coverage"]
    lines.append(f"\nEffective coverage:")
    lines.append(f"  EFAC : {cov['any_efac_count']:6d} / {diag['n_toas']} TOAs")
    lines.append(f"  EQUAD: {cov['any_equad_count']:6d} / {diag['n_toas']} TOAs")
    lines.append(f"  ECORR: {cov['any_ecorr_count']:6d} / {diag['n_toas']} TOAs")

    # Override semantics
    lines.append(f"\nOverride semantics: {diag['override_semantics']}")
    lines.append("=" * 60)

    return "\n".join(lines)
