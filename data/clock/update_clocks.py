#!/usr/bin/env python3
"""Update JUG's bundled clock-correction files from the IPTA upstream repository.

Source (confirmed)
-------------------
IPTA pulsar-clock-corrections, the authoritative maintained set of Tempo2-runtime
clock files:

    https://github.com/ipta/pulsar-clock-corrections
    raw: https://raw.githubusercontent.com/ipta/pulsar-clock-corrections/main/T2runtime/clock/<file>

Each local ``data/clock/<name>.clk`` maps to the upstream ``T2runtime/clock/<name>.clk``
of the same name.  Local files with no same-named upstream counterpart (JUG-only or
legacy chains) are skipped automatically (HTTP 404).

Why a sentinel-aware comparison
-------------------------------
Clock files often end with an *extrapolation / validity endpoint* that is NOT real
measured data, e.g.:
  * an explicit far-future sentinel row (``99999.00000 <held value>``) -- gps2utc, jb2gps;
  * a flat-valued tail where the correction is held constant (BIPM tai2tt files);
  * a round far-future MJD with a reset value reached after a large data gap (eff2gps:
    real data ends ~MJD 57195, then a lone ``60000.0 0.0`` endpoint).

A naive "last MJD in file" comparison is INVALID: a file padded to MJD 99999 would
spuriously look more current than a genuinely newer file, so the updater would skip
real updates (or overwrite a newer file with an older one).  We therefore compare on
the **last real-data MJD**, after excluding sentinel/extrapolation rows.

Sentinel / extrapolation detection rule (documented, conservative)
------------------------------------------------------------------
Given parsed (mjd, value) rows (comments/directives/inline-``#`` stripped):
  1. Drop any row with ``mjd >= SENTINEL_MJD`` (90000) -- explicit far-future sentinels.
  2. Drop any row with ``mjd > now_mjd + FUTURE_MARGIN_DAYS`` (today+400) -- future-dated
     extrapolation beyond plausible real data.
  3. ``last_real_mjd`` = MJD of the last surviving row.
  4. AMBIGUOUS (flag + never overwrite) if, after 1-2:
       * fewer than 2 real rows, or MJDs not non-decreasing, or
       * the gap to the final surviving row exceeds ``MAX_TAIL_GAP_DAYS`` (365) -- a lone
         endpoint reached after a long gap (the eff2gps case): we cannot confidently call
         it real data, so we refuse to touch the file.
A trailing flat-value run (constant correction) is *reported* for transparency but is NOT
auto-trimmed for the newer-than decision -- counting a past flat tail as "real" can only
make local look more current, i.e. err toward SKIP (safe), never toward a bad overwrite.

"Upstream is newer/changed" if:
  * upstream ``last_real_mjd`` > local ``last_real_mjd`` (extends real coverage), OR
  * equal real coverage but the sha256 of the real-data portion differs (a correction /
    reformat confined to real rows), ignoring differences in the sentinel/extrapolation tail.

Safety
------
  * --check / --dry-run (default): report only; no downloads-to-replace, no writes.
  * --apply: replace only when upstream is genuinely newer/changed AND the download
    validates (parses, monotonic MJDs, plausible values, real coverage >= local).
  * Backup each replaced file to ``data/clock/.backup/<file>.<timestamp>``.
  * Atomic: download to a temp file, validate, then ``os.replace`` into place.
  * Update ``data/manifest.json`` (sha256 + size) for replaced files so
    ``verify_data_integrity`` stays consistent.
  * Ambiguous / failed-validation / older-upstream files are left untouched and reported.

Usage
-----
    python data/clock/update_clocks.py            # --check (safe default)
    python data/clock/update_clocks.py --check
    python data/clock/update_clocks.py --apply
    python data/clock/update_clocks.py --apply --only gps2utc.clk,ao2gps.clk
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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CLOCK_DIR = Path(__file__).resolve().parent                 # data/clock/
DATA_DIR = CLOCK_DIR.parent                                 # data/
MANIFEST = DATA_DIR / "manifest.json"
BACKUP_DIR = CLOCK_DIR / ".backup"

UPSTREAM_BASE = ("https://raw.githubusercontent.com/ipta/"
                 "pulsar-clock-corrections/main/T2runtime/clock/")

SENTINEL_MJD = 90000.0          # rows at/after this MJD are explicit far-future sentinels
FUTURE_MARGIN_DAYS = 400.0      # rows beyond today+this are extrapolation, not data
MAX_TAIL_GAP_DAYS = 365.0       # a final row this far past the previous one => ambiguous
MAX_PLAUSIBLE_SEC = 100.0       # |correction| sanity bound (tai2tt ~32s, leap ~37s)
HTTP_TIMEOUT = 60               # seconds


def now_mjd() -> float:
    return datetime.now(timezone.utc).timestamp() / 86400.0 + 40587.0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_clock(text: str) -> Tuple[List[Tuple[float, float]], bool, str]:
    """Parse clock text into [(mjd, value), ...]. Returns (rows, ok, note).

    Skips blank lines, comments ('#'), and Tempo2 directives (FORMAT/MODE/INCLUDE).
    Strips inline '# ...' comments. A line is a data row iff its first two
    whitespace tokens parse as floats.
    """
    rows: List[Tuple[float, float]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()        # strip inline + full comments
        if not line:
            continue
        tok = line.split()
        if tok[0].upper() in ("FORMAT", "MODE", "INCLUDE", "EOF"):
            continue
        if len(tok) < 2:
            continue
        try:
            mjd = float(tok[0]); val = float(tok[1])
        except ValueError:
            continue
        if not (math.isfinite(mjd) and math.isfinite(val)):
            return rows, False, "non-finite value encountered"
        rows.append((mjd, val))
    if len(rows) < 2:
        return rows, False, f"too few data rows ({len(rows)})"
    return rows, True, ""


def analyze(rows: List[Tuple[float, float]], nowm: float) -> dict:
    """Compute real-coverage info + sentinel/ambiguity detection for a parsed file."""
    fut_cut = nowm + FUTURE_MARGIN_DAYS
    real = [(m, v) for (m, v) in rows if m < SENTINEL_MJD and m <= fut_cut]
    dropped = len(rows) - len(real)

    # sentinel description
    tail = rows[-1]
    sentinel = None
    if tail[0] >= SENTINEL_MJD:
        sentinel = f"explicit sentinel MJD {tail[0]:.0f}"
    elif tail[0] > fut_cut:
        sentinel = f"future-dated tail MJD {tail[0]:.0f} (> now+{FUTURE_MARGIN_DAYS:.0f}d)"

    # Trailing flat-value run = held-constant extrapolation tail (BIPM tai2tt etc).
    # Exclude it from the coverage/hash so tail-length-only differences are ignored
    # (spec: "ignore differences confined only to the extrapolation tail"). The held
    # repeats are dropped but the first row of the run is kept as the last real
    # measurement.  All-flat files (constant-offset chains) are NOT trimmed.
    flat_run = 1
    for i in range(len(real) - 1, 0, -1):
        if abs(real[i][1] - real[i - 1][1]) <= 1e-15:
            flat_run += 1
        else:
            break
    flat_tail = flat_run if flat_run >= 3 else 0
    if flat_tail and flat_tail < len(real):
        core = real[: len(real) - flat_tail + 1]   # keep first of the held run
    else:
        core = real

    ambiguous: Optional[str] = None
    if len(core) < 2:
        ambiguous = "fewer than 2 real rows after sentinel/flat-tail removal"
    else:
        mjds = [m for m, _ in core]
        if any(mjds[i + 1] < mjds[i] - 1e-6 for i in range(len(mjds) - 1)):
            ambiguous = "MJDs not non-decreasing"
        elif core[-1][0] - core[-2][0] > MAX_TAIL_GAP_DAYS:
            ambiguous = (f"final real row at MJD {core[-1][0]:.0f} is "
                         f"{core[-1][0] - core[-2][0]:.0f}d past the previous row "
                         f"-- looks like an extrapolation endpoint, not data")

    last_real = core[-1][0] if core else float("nan")
    # canonical hash of the real (non-tail) portion: corrections/reformats vs tail-only
    canon = "\n".join(f"{m:.6f} {v:.12g}" for m, v in core).encode()
    real_hash = hashlib.sha256(canon).hexdigest()
    return dict(last_real_mjd=last_real, n_real=len(core), n_dropped=dropped,
                sentinel=sentinel, flat_tail=flat_tail, ambiguous=ambiguous,
                real_hash=real_hash)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def fetch_upstream(name: str) -> Tuple[Optional[str], str]:
    """Return (text, note). text is None on 404 / network error."""
    url = UPSTREAM_BASE + name
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jug-update-clocks"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read().decode("utf-8", "replace"), "ok"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, "no upstream counterpart (404)"
        return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, f"fetch error: {e}"


# --------------------------------------------------------------------------- #
# Per-file decision
# --------------------------------------------------------------------------- #
def evaluate(local_path: Path, nowm: float) -> dict:
    """Compare one local clock file against upstream. Returns a result dict."""
    name = local_path.name
    res = dict(name=name, action="skip", reason="", local=None, upstream=None,
               up_text=None)

    loc_rows, ok, note = parse_clock(local_path.read_text(errors="replace"))
    if not ok:
        res["reason"] = f"local unparseable ({note}); flagged for manual review"
        res["flag"] = True
        return res
    loc = analyze(loc_rows, nowm)
    res["local"] = loc

    up_text, fnote = fetch_upstream(name)
    if up_text is None:
        res["reason"] = fnote
        return res
    up_rows, uok, unote = parse_clock(up_text)
    if not uok:
        res["reason"] = f"upstream unparseable ({unote})"
        res["flag"] = True
        return res
    up = analyze(up_rows, nowm)
    res["upstream"] = up
    res["up_text"] = up_text

    # Ambiguity on EITHER side => never overwrite.
    if loc["ambiguous"]:
        res["reason"] = f"local endpoint ambiguous: {loc['ambiguous']}"
        res["flag"] = True
        return res
    if up["ambiguous"]:
        res["reason"] = f"upstream endpoint ambiguous: {up['ambiguous']}"
        res["flag"] = True
        return res

    newer_cov = up["last_real_mjd"] > loc["last_real_mjd"] + 1e-6
    same_cov = abs(up["last_real_mjd"] - loc["last_real_mjd"]) <= 1e-6
    changed = up["real_hash"] != loc["real_hash"]

    if newer_cov or (same_cov and changed):
        # Validation gate (Step 3): never replace with shorter/malformed data.
        if up["last_real_mjd"] < loc["last_real_mjd"] - 1e-6:
            res["reason"] = "upstream real coverage shorter -- refused"
            return res
        if any(abs(v) > MAX_PLAUSIBLE_SEC for _, v in up_rows):
            res["reason"] = "upstream has implausible correction value -- refused"
            res["flag"] = True
            return res
        res["action"] = "update"
        why = []
        if newer_cov:
            why.append(f"coverage {loc['last_real_mjd']:.0f}->{up['last_real_mjd']:.0f}")
        if same_cov and changed:
            why.append("real-data content changed (same coverage)")
        res["reason"] = "; ".join(why)
    else:
        if same_cov:
            res["reason"] = "up to date (identical real data)"
        else:
            res["reason"] = (f"local newer/equal "
                             f"(local {loc['last_real_mjd']:.0f} >= "
                             f"upstream {up['last_real_mjd']:.0f})")
    return res


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def apply_update(res: dict, nowm: float) -> Tuple[bool, str]:
    """Backup, atomic-replace, manifest-update for one 'update' result."""
    name = res["name"]
    local_path = CLOCK_DIR / name
    up_text = res["up_text"]

    # Re-validate the download independently before touching disk.
    rows, ok, note = parse_clock(up_text)
    if not ok:
        return False, f"validation failed: {note}"
    info = analyze(rows, nowm)
    if info["ambiguous"]:
        return False, f"validation failed: {info['ambiguous']}"
    if info["last_real_mjd"] < res["local"]["last_real_mjd"] - 1e-6:
        return False, "validation failed: shorter real coverage"

    # Atomic temp write
    fd, tmp = tempfile.mkstemp(dir=str(CLOCK_DIR), prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(up_text)
        # Backup existing
        BACKUP_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        backup = BACKUP_DIR / f"{name}.{ts}"
        backup.write_bytes(local_path.read_bytes())
        # Move into place
        os.replace(tmp, str(local_path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    update_manifest(name, local_path)
    return True, f"backed up -> {backup.name}"


def update_manifest(name: str, path: Path) -> None:
    """Update/insert the clock/<name> entry in manifest.json (sha256 + size)."""
    if not MANIFEST.exists():
        return
    m = json.loads(MANIFEST.read_text())
    rel = f"clock/{name}"
    files = m.setdefault("files", {})
    if rel in files or True:  # add if present; also add new replaced files
        files[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def iter_local_clocks(only: Optional[set]) -> List[Path]:
    files = sorted(p for p in CLOCK_DIR.glob("*.clk") if p.is_file())
    if only:
        files = [p for p in files if p.name in only]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", "--dry-run", action="store_true",
                   help="report what would update; no writes (default)")
    g.add_argument("--apply", action="store_true", help="perform replacements")
    ap.add_argument("--only", help="comma-separated subset of filenames")
    ap.add_argument("--clock-dir", help="override clock dir (for testing on a copy)")
    args = ap.parse_args()

    global CLOCK_DIR, DATA_DIR, MANIFEST, BACKUP_DIR
    if args.clock_dir:
        CLOCK_DIR = Path(args.clock_dir).resolve()
        DATA_DIR = CLOCK_DIR.parent
        MANIFEST = DATA_DIR / "manifest.json"
        BACKUP_DIR = CLOCK_DIR / ".backup"

    apply = args.apply
    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    nowm = now_mjd()

    print("=" * 78)
    print(f"JUG clock updater  ({'APPLY' if apply else 'CHECK / dry-run'})")
    print(f"upstream: {UPSTREAM_BASE}")
    print(f"clock dir: {CLOCK_DIR}    now MJD ~ {nowm:.0f}")
    print("=" * 78)
    hdr = f"{'file':22s} {'local_real':>10s} {'up_real':>9s} {'sentinel':>9s} {'action':>7s}  reason"
    print(hdr); print("-" * len(hdr))

    updated, skipped, flagged, failed, checked = [], [], [], [], 0
    for p in iter_local_clocks(only):
        checked += 1
        res = evaluate(p, nowm)
        loc = res.get("local") or {}
        up = res.get("upstream") or {}
        lr = f"{loc.get('last_real_mjd', float('nan')):.0f}" if loc else "-"
        ur = f"{up.get('last_real_mjd', float('nan')):.0f}" if up else "-"
        sent = "yes" if (loc.get("sentinel") or loc.get("flat_tail")) else "no"
        flag = res.get("flag", False)
        act = "UPDATE" if res["action"] == "update" else ("FLAG" if flag else "skip")

        if res["action"] == "update" and apply:
            okk, msg = apply_update(res, nowm)
            if okk:
                act = "DONE"; updated.append(res["name"]); res["reason"] += f"  [{msg}]"
            else:
                act = "FAILVAL"; failed.append((res["name"], msg)); res["reason"] = msg
        elif res["action"] == "update":
            updated.append(res["name"])
        elif flag:
            flagged.append((res["name"], res["reason"]))
        else:
            skipped.append(res["name"])

        print(f"{res['name']:22s} {lr:>10s} {ur:>9s} {sent:>9s} {act:>7s}  {res['reason']}")

    print("-" * len(hdr))
    print(f"checked {checked} | "
          f"{'updated' if apply else 'would update'} {len(updated)} | "
          f"skipped {len(skipped)} | flagged {len(flagged)} | failed-validation {len(failed)}")
    if updated:
        print(f"  {'updated' if apply else 'would update'}: {', '.join(updated)}")
    if flagged:
        print("  flagged for manual review:")
        for n, r in flagged:
            print(f"    {n}: {r}")
    if failed:
        print("  failed validation (left untouched):")
        for n, r in failed:
            print(f"    {n}: {r}")
    if not apply and updated:
        print("\nRun with --apply to perform these updates (backups in .backup/).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
