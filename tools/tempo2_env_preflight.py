#!/usr/bin/env python3
"""Read-only preflight: compare JUG ephemeris/clock data paths vs tempo2 expectations.

Environment mismatches (DE440 missing, BIPM clock extrapolation) can produce
µs-scale residual offsets unrelated to the staged_bclt algorithm. This script
prints what JUG loads and whether common tempo2 data files exist locally.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _exists(path: Path | None) -> str:
    if path is None:
        return "missing"
    return "ok" if path.is_file() else f"NOT FOUND ({path})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ephem",
        default=os.environ.get("TEMPO2", ""),
        help="Tempo2 root (optional; checks $TEMPO2/ephem if set)",
    )
    args = parser.parse_args()

    print("JUG tempo2 environment preflight")
    print("=" * 60)

    try:
        from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path

        for name in ("DE440", "DE405"):
            try:
                p = resolve_tempo2_ephemeris_path(name)
                print(f"  ephemeris {name}: {_exists(Path(p))}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ephemeris {name}: error ({exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"  ephemeris resolver: error ({exc})")

    tempo2_root = Path(args.ephem) if args.ephem else None
    if tempo2_root and tempo2_root.is_dir():
        for rel in (
            "ephem/de440.bsp",
            "ephem/de405.bsp",
            "clock/bipm0019.clk",
            "clock/ut1.dat",
        ):
            print(f"  tempo2 {rel}: {_exists(tempo2_root / rel)}")
    else:
        print("  tempo2 tree: not checked (set --ephem or $TEMPO2)")

    print(
        "\nNote: NG5/EPTA offsets from clock-file extrapolation are closed by "
        "installing matching BIPM/TT data, not by algorithm changes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
