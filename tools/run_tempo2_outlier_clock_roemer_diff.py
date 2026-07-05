#!/usr/bin/env python3
"""CLI for per-TOA clock / Roemer diff on tempo2 parity outliers."""

from __future__ import annotations

import argparse
from pathlib import Path

from jug.testing.tempo2_outlier_diff import compare_clock_roemer_per_toa, format_outlier_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare JUG vs libstempo clock and Roemer terms per TOA."
    )
    parser.add_argument("par", type=Path, help="Path to .par file")
    parser.add_argument("tim", type=Path, help="Path to .tim file")
    parser.add_argument("--fixture-id", default="", help="Label for the report header")
    parser.add_argument(
        "--outlier-threshold-ns",
        type=float,
        default=10.0,
        help="Mark TOAs with |residual diff| above this threshold (default 10 ns)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all TOAs, not just outliers",
    )
    args = parser.parse_args()

    report = compare_clock_roemer_per_toa(
        args.par,
        args.tim,
        fixture_id=args.fixture_id,
        outlier_threshold_ns=args.outlier_threshold_ns,
    )
    print(format_outlier_report(report, show_all=args.show_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
