#!/usr/bin/env python3
"""Record wsrt167 tempo2 graph-mode pack-build and JIT timings.

Writes JSON plus a human-readable summary. No pass/fail thresholds — use for
before/after comparisons when optimizing compile time.

Usage:
    cd ref-packages/jug
    python tools/run_tempo2_graph_timing_wsrt167.py
    python tools/run_tempo2_graph_timing_wsrt167.py --fit-params F0
    python tools/run_tempo2_graph_timing_wsrt167.py --output /tmp/timing.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TESTS = _REPO / "tests"
for p in (_REPO, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tempo2_fixtures import get_tempo2_fixture  # noqa: E402

from jug.testing.tempo2_graph_timing import (  # noqa: E402
    benchmark_wsrt167_graph_modes,
    write_timing_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-params",
        nargs="+",
        default=["RAJ", "DECJ", "F0", "DM"],
        help="Fit parameter names (default: RAJ DECJ F0 DM)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/jug_tempo2_graph_timing_wsrt167.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--fixture",
        default="wsrt167",
        help="Tempo2 fixture id from tests/data_tempo2",
    )
    args = parser.parse_args()

    fixture = get_tempo2_fixture(args.fixture)
    report = benchmark_wsrt167_graph_modes(
        fixture["par_path"],
        fixture["tim_path"],
        args.fit_params,
        fixture_id=args.fixture,
    )
    write_timing_report(args.output, report)

    print(f"Wrote {args.output}")
    for line in report.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())