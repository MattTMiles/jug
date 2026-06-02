#!/usr/bin/env python3
"""Run Phase A term-by-term tempo2 diagnostics for Case B/C fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
sys.path.insert(0, str(TESTS_DIR))

from tempo2_fixtures import get_tempo2_fixture, list_tempo2_tdb_diagnostic_fixtures

from jug.residuals.diagnostic_conventions import DiagnosticConventions
from jug.testing.phase_a_comparison import compare_fixture_phase_a, rank_phase_b_ports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        help="Fixture id (default: all TDB diagnostic fixtures)",
    )
    parser.add_argument(
        "--residual-metric",
        choices=("raw", "weighted_centered"),
        default="raw",
        help="Residual comparison metric (tempo2 acceptance uses raw only)",
    )
    parser.add_argument(
        "--term-set",
        choices=("core", "extended"),
        default="core",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    fixture_ids = args.fixtures or [fx["id"] for fx in list_tempo2_tdb_diagnostic_fixtures()]
    conv = DiagnosticConventions(
        residual_metric=args.residual_metric,
        term_set=args.term_set,
    )
    if args.residual_metric != "raw":
        print("Warning: weighted_centered is for pint-family diagnostics only", file=sys.stderr)

    reports = []
    for fixture_id in fixture_ids:
        fixture = get_tempo2_fixture(fixture_id)
        report = compare_fixture_phase_a(fixture, conventions=conv)
        ranking = rank_phase_b_ports(report)
        print(f"\n=== {fixture_id} ===")
        for label, stats in report.residual_stats.items():
            print(
                f"  {label}: RMS={stats.rms_ns:.3f} ns, "
                f"p99={stats.p99_ns:.3f} ns, annual~={stats.annual_amp_ns}"
            )
        print("  term ranking (RMS ns):")
        if ranking:
            for term in ranking:
                for prefix in ("oracle_delta::", "pint_mode_delta::"):
                    key = f"{prefix}{term}"
                    if key in report.term_stats:
                        s = report.term_stats[key]
                        print(f"    {term}: {s.rms_ns:.3f}")
                        break
        else:
            for key, s in sorted(
                report.term_stats.items(), key=lambda kv: kv[1].rms_ns, reverse=True
            ):
                print(f"    {key}: {s.rms_ns:.3f}")
        reports.append(report.to_dict())

    if args.output:
        args.output.write_text(json.dumps(reports, indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
