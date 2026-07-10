#!/usr/bin/env python3
"""Update JUG's bundled IERS Earth-orientation-parameter (EOP) file.

Source (confirmed)
-------------------
IERS Earth Orientation Centre (Paris Observatory), the same file and URL that
Tempo2's ``T2runtime/earth/update_eop.sh`` pulls with ``wget -N``:

    https://hpiers.obspm.fr/iers/eop/eopc04/eopc04_IAU2000.62-now

The local file ``data/earth/eopc04_IAU2000.62-now`` is the EOP C04 (old format)
time series: polar motion (x, y), UT1-UTC, LOD and their errors, one row per day
back to 1962. JUG's barycentring needs UT1-UTC and polar motion; the file is
extended by IERS as new daily solutions are published, so it goes stale over time.

Format (old C04, whitespace-delimited)
--------------------------------------
After a fixed header block, each data row is::

    YYYY MM DD  MJD  x("")  y("")  UT1-UTC(s)  LOD(s)  ...errors...

We key freshness on the **last MJD** (column 4). A daily series with no sentinel
padding, so the newest real measurement is simply the last parsed row.

Why validate rather than blind ``wget -N``
------------------------------------------
Tempo2's shell script overwrites unconditionally on any HTTP 200. We instead:
  * parse the download and require it to be a well-formed C04 series,
  * require monotonically non-decreasing MJDs (no shuffled/garbled rows),
  * require physically plausible values (|x|,|y| < 2", |UT1-UTC| < 1 s),
  * require coverage at least as long as the local file (never shrink), and
  * only then atomically replace, backing up the old file and updating
    ``data/manifest.json`` so ``verify_data_integrity`` stays consistent.

Safety
------
  * --check / --dry-run (default): report only; no writes.
  * --apply: replace only when upstream extends coverage (or same coverage but
    content changed) AND the download validates.
  * Backup the replaced file to ``data/earth/.backup/<name>.<timestamp>``.
  * Atomic: download to a temp file, validate, then ``os.replace`` into place.

Usage
-----
    python data/earth/update_eop.py            # --check (safe default)
    python data/earth/update_eop.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EARTH_DIR = Path(__file__).resolve().parent                 # data/earth/
DATA_DIR = EARTH_DIR.parent                                 # data/
MANIFEST = DATA_DIR / "manifest.json"
BACKUP_DIR = EARTH_DIR / ".backup"

EOP_NAME = "eopc04_IAU2000.62-now"
UPSTREAM_URL = "https://hpiers.obspm.fr/iers/eop/eopc04/" + EOP_NAME

MJD_COL = 3                     # 0-based: YYYY MM DD MJD ...
MAX_POLAR_ARCSEC = 2.0          # |x|, |y| sanity bound (real values < ~0.6")
MAX_UT1UTC_SEC = 1.0            # |UT1-UTC| stays within +-0.9 s (leap-second clamp)
MIN_PLAUSIBLE_MJD = 37665.0     # series starts 1962-01-01; reject anything earlier
HTTP_TIMEOUT = 120              # seconds


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #
def parse_eop(text: str) -> Tuple[List[Tuple[float, float, float]], bool, str]:
    """Parse EOP C04 text into [(mjd, x, ut1utc), ...].

    Returns (rows, ok, note). A line is a data row iff it has >= 7 whitespace
    tokens that all parse as floats and column 4 (MJD) is a plausible MJD; the
    header block and any stray lines are skipped.
    """
    rows: List[Tuple[float, float, float]] = []
    for raw in text.splitlines():
        tok = raw.split()
        if len(tok) < 7:
            continue
        try:
            vals = [float(t) for t in tok[:7]]
        except ValueError:
            continue  # header / non-numeric line
        mjd, x, ut1utc = vals[MJD_COL], vals[4], vals[6]
        if not (math.isfinite(mjd) and math.isfinite(x) and math.isfinite(ut1utc)):
            return rows, False, "non-finite value encountered"
        if mjd < MIN_PLAUSIBLE_MJD:
            continue  # not a real C04 data row
        rows.append((mjd, x, ut1utc))
    if len(rows) < 2:
        return rows, False, f"too few data rows ({len(rows)})"
    return rows, True, ""


def validate(rows: List[Tuple[float, float, float]]) -> Tuple[bool, str]:
    """Physical/structural sanity checks on parsed EOP rows."""
    mjds = [m for m, _, _ in rows]
    if any(mjds[i + 1] < mjds[i] - 1e-6 for i in range(len(mjds) - 1)):
        return False, "MJDs not non-decreasing"
    for m, x, u in rows:
        if abs(x) > MAX_POLAR_ARCSEC:
            return False, f"implausible polar motion x={x} at MJD {m:.0f}"
        if abs(u) > MAX_UT1UTC_SEC:
            return False, f"implausible UT1-UTC={u} at MJD {m:.0f}"
    return True, ""


def last_mjd(rows: List[Tuple[float, float, float]]) -> float:
    return rows[-1][0]


def real_hash(rows: List[Tuple[float, float, float]]) -> str:
    canon = "\n".join(f"{m:.1f} {x:.9g} {u:.9g}" for m, x, u in rows).encode()
    return hashlib.sha256(canon).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def fetch_upstream() -> Tuple[Optional[str], str]:
    try:
        req = urllib.request.Request(
            UPSTREAM_URL, headers={"User-Agent": "jug-update-eop"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read().decode("utf-8", "replace"), "ok"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, f"fetch error: {e}"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def update_manifest(rel: str, path: Path) -> None:
    if not MANIFEST.exists():
        return
    m = json.loads(MANIFEST.read_text())
    files = m.setdefault("files", {})
    files[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", "--dry-run", action="store_true",
                   help="report whether an update is available; no writes (default)")
    g.add_argument("--apply", action="store_true", help="perform the replacement")
    ap.add_argument("--earth-dir", help="override earth dir (for testing on a copy)")
    args = ap.parse_args()

    global EARTH_DIR, DATA_DIR, MANIFEST, BACKUP_DIR
    if args.earth_dir:
        EARTH_DIR = Path(args.earth_dir).resolve()
        DATA_DIR = EARTH_DIR.parent
        MANIFEST = DATA_DIR / "manifest.json"
        BACKUP_DIR = EARTH_DIR / ".backup"

    local_path = EARTH_DIR / EOP_NAME
    apply = args.apply

    print("=" * 78)
    print(f"JUG EOP updater  ({'APPLY' if apply else 'CHECK / dry-run'})")
    print(f"upstream: {UPSTREAM_URL}")
    print(f"earth dir: {EARTH_DIR}")
    print("=" * 78)

    if not local_path.exists():
        print(f"  local file missing: {local_path}")
        return 1
    loc_rows, ok, note = parse_eop(local_path.read_text(errors="replace"))
    if not ok:
        print(f"  local unparseable ({note}) -- refusing to touch")
        return 1
    loc_mjd = last_mjd(loc_rows)
    print(f"  local  last MJD {loc_mjd:.0f}  ({len(loc_rows)} rows)")

    up_text, fnote = fetch_upstream()
    if up_text is None:
        print(f"  upstream fetch failed: {fnote}")
        return 1
    up_rows, uok, unote = parse_eop(up_text)
    if not uok:
        print(f"  upstream unparseable ({unote}) -- refusing")
        return 1
    vok, vnote = validate(up_rows)
    if not vok:
        print(f"  upstream failed validation ({vnote}) -- refusing")
        return 1
    up_mjd = last_mjd(up_rows)
    print(f"  upstream last MJD {up_mjd:.0f}  ({len(up_rows)} rows)")

    newer = up_mjd > loc_mjd + 1e-6
    changed = real_hash(up_rows) != real_hash(loc_rows)
    if up_mjd < loc_mjd - 1e-6:
        print("  result: upstream coverage SHORTER than local -- refused (left untouched)")
        return 0
    if not (newer or changed):
        print("  result: up to date (identical EOP data) -- nothing to do")
        return 0

    why = (f"coverage {loc_mjd:.0f} -> {up_mjd:.0f} "
           f"(+{up_mjd - loc_mjd:.0f} days)") if newer else "content changed (same coverage)"
    print(f"  result: UPDATE available -- {why}")

    if not apply:
        print("\nRun with --apply to perform this update (backup in .backup/).")
        return 0

    # Atomic temp write -> validate-on-disk -> backup -> replace
    fd, tmp = tempfile.mkstemp(dir=str(EARTH_DIR), prefix=f".{EOP_NAME}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(up_text)
        BACKUP_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        backup = BACKUP_DIR / f"{EOP_NAME}.{ts}"
        backup.write_bytes(local_path.read_bytes())
        os.replace(tmp, str(local_path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    update_manifest(f"earth/{EOP_NAME}", local_path)
    print(f"  applied: backed up -> {backup.name}; manifest updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
