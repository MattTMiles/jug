#!/usr/bin/env python3
"""Generate tiny libstempo-simulated tempo2 par/tim fixtures for JUG tests.

Maintainer-only: run manually to refresh committed artifacts under
``tests/data_tempo2_sim/``. Normal pytest collection does not invoke this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "data_tempo2_sim"
GENERATOR_PATH = "tools/generate_tempo2_sim_fixtures.py"

ISOLATED_MJDS = (56000.0, 56010.0, 56020.0, 56030.0, 56040.0, 56050.0)
BINARY_MJDS = (56000.0, 56002.5, 56005.0, 56007.5, 56040.0, 56042.5, 56045.0, 56047.5)
TRACK2_MJDS = BINARY_MJDS
ADDSAT_MJDS = (56000.0, 56002.5, 56005.0, 56007.5, 56040.0, 56042.5, 56045.0, 56047.5, 56050.0, 56052.5)


def _track2_addsat_flags() -> tuple[str, ...]:
    flags = [""] * len(ADDSAT_MJDS)
    flags[3] = "-addsat -1"
    flags[6] = "-addsat +1"
    return tuple(flags)

COMMON_HEADER = """\
PSRJ            J0000+0000
F0              200.0              1
F1              -1e-15             1
PEPOCH          56000
POSEPOCH        56000
DM              10.0               1
DMEPOCH         56000
EPHEM           DE440
CLK             TT(BIPM2021)
TIMEEPH         IF99
T2CMETHOD       IAU2000B
TZRMJD          56000
TZRFRQ          1400
TZRSITE         ao
"""

CANONICAL_FIXTURE_IDS = (
    "sim_isolated_tcb",
    "sim_t2_tcb",
    "sim_ell1_tcb",
    "sim_ell1h_tcb",
    "sim_dd_tcb",
    "sim_ddh_tcb",
    "sim_bt_tcb",
    "sim_ddk_tcb",
    "sim_dd_tdb",
    "sim_dd_ecliptic_tcb",
    "sim_t2_track2_pn",
    "sim_t2_track2_addsat",
    "sim_t2_multisys",
    "sim_fd_tcb",
    "sim_dilatefreq_no",
)


@dataclass(frozen=True)
class TimPatch:
    pn_by_toa_index: dict[int, int] = field(default_factory=dict)
    addsat_by_toa_index: dict[int, int] = field(default_factory=dict)
    use_libstempo_pulse_numbers: bool = False


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    par_text: str
    mjds: tuple[float, ...]
    freq_mhz: tuple[float, ...] | float = 1400.0
    toaerr_us: float = 1.0
    observatory: tuple[str, ...] | str = "ao"
    flags: tuple[str, ...] | str = ""
    binary: str | None = None
    option_tags: tuple[str, ...] = field(default_factory=tuple)
    designmatrix_params: tuple[str, ...] = field(default_factory=tuple)
    tim_patch: TimPatch = field(default_factory=TimPatch)


def _equatorial_header(*, units: str = "TCB", dilatefreq: str = "Y") -> str:
    return (
        "MODE 1\n"
        f"{COMMON_HEADER}"
        f"RAJ             00:00:00\n"
        f"DECJ            +00:00:00\n"
        f"UNITS           {units}\n"
        f"DILATEFREQ      {dilatefreq}\n"
    )


def _ecliptic_header(*, units: str = "TCB", dilatefreq: str = "Y") -> str:
    """Ecliptic sky coords via LAMBDA/BETA (no ECL keyword — tempo2 warns on bare ECL)."""
    return (
        "MODE 1\n"
        f"{COMMON_HEADER}"
        f"LAMBDA          244.3476761957296\n"
        f"BETA            -10.071818621607784\n"
        f"PMLAMBDA        0.8336132358792978\n"
        f"PMBETA          -8.079673767426723\n"
        f"UNITS           {units}\n"
        f"DILATEFREQ      {dilatefreq}\n"
    )


def _t2_binary_block(*, track: str | None = None) -> str:
    lines = [
        "BINARY         T2",
        "PB             1.533449474299159639      1",
        "A1             1.8979912039474248083     1",
        "TASC           53113.950742015404956     1",
        "EPS1           4.9340687844353634473e-09 1",
        "EPS2           -1.3733351812132134104e-07 1",
        "PBDOT          5.1215961987477866184e-13 1",
        "XDOT           -1.1702324602187853937e-15 1",
        "M2             0.21839495035469286791    1",
    ]
    if track is not None:
        lines.append(f"TRACK          {track}")
    return "\n".join(lines) + "\n"


def _ell1_binary_block() -> str:
    return """\
BINARY         ELL1
PB             16.335348082509345013     1
A1             11.003314527165528484     1
TASC           59102.65984612830664      1
EPS1           -4.0120625958784907292e-06 1
EPS2           -9.150692249927781532e-06 1
M2             0.22
"""


def _ell1h_binary_block() -> str:
    return """\
BINARY         ELL1H
PB             2.0118037698861025832     1
A1             1.9019555266996599312     1
TASC           55162.281746099422996     1
EPS1           3.5481095478453669965e-06 1
EPS2           -2.8509260324487741175e-06 1
H3             6.0428813817245745826e-07 1
STIG           0.82184158622979877959    1
"""


def _dd_binary_block(*, units: str = "TCB") -> str:
    return f"""\
BINARY         DD
PB             14.3484575489552722       1
T0             52506.375998904087385     1
A1             8.8016496478606525596     1
OM             181.89033874817114354     1
ECC            0.00017368183334286273809 1
M2             0.72479357377044024887    1
SINI           0.80101353240028137781    1
{_equatorial_header(units=units).split('UNITS')[0]}UNITS           {units}
DILATEFREQ      Y
"""


def _ddh_binary_block() -> str:
    return """\
BINARY         DDH
PB             10.913177749957992911     1
T0             51575.770461846424745     1
A1             8.3504662586623404482     1
OM             219.46575691386730338     1
ECC            2.0338525783849762634e-05 1
H3             8.493160585536076319e-07  1
STIG           0.89860253442821124423    1
"""


def _bt_binary_block() -> str:
    return """\
BINARY         BT
PB             115.65378859656417472     1
T0             52890.249088598756469     1
A1             40.769520607128638134     1
OM             346.62669284100535816     1
ECC            2.3688449683527307792e-05 1
"""


def _ddk_binary_block() -> str:
    return """\
BINARY         DDK
PB             5.7410459                 1
T0             50000.0                   1
A1             3.3667144                 1
OM             1.35                      1
ECC            1.918e-5                  1
KIN            137.56                    1
KOM            207.0                     1
PX             6.396                     1
PMRA           121.438                   1
PMDEC          -71.475                   1
"""


def build_fixture_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec(
            fixture_id="sim_isolated_tcb",
            par_text=_equatorial_header(),
            mjds=ISOLATED_MJDS,
            option_tags=("BINARY=isolated", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "F1", "DM"),
        ),
        FixtureSpec(
            fixture_id="sim_t2_tcb",
            par_text=_equatorial_header() + _t2_binary_block(),
            mjds=BINARY_MJDS,
            binary="T2",
            option_tags=("BINARY=T2", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1", "EPS1", "EPS2"),
        ),
        FixtureSpec(
            fixture_id="sim_ell1_tcb",
            par_text=_equatorial_header() + _ell1_binary_block(),
            mjds=BINARY_MJDS,
            binary="ELL1",
            option_tags=("BINARY=ELL1", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1", "EPS1", "EPS2"),
        ),
        FixtureSpec(
            fixture_id="sim_ell1h_tcb",
            par_text=_equatorial_header() + _ell1h_binary_block(),
            mjds=BINARY_MJDS,
            binary="ELL1H",
            option_tags=("BINARY=ELL1H", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1", "EPS1", "EPS2"),
        ),
        FixtureSpec(
            fixture_id="sim_dd_tcb",
            par_text=_equatorial_header() + """\
BINARY         DD
PB             14.3484575489552722       1
T0             52506.375998904087385     1
A1             8.8016496478606525596     1
OM             181.89033874817114354     1
ECC            0.00017368183334286273809 1
M2             0.72479357377044024887    1
SINI           0.80101353240028137781    1
""",
            mjds=BINARY_MJDS,
            binary="DD",
            option_tags=("BINARY=DD", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1"),
        ),
        FixtureSpec(
            fixture_id="sim_ddh_tcb",
            par_text=_equatorial_header() + _ddh_binary_block(),
            mjds=BINARY_MJDS,
            binary="DDH",
            option_tags=("BINARY=DDH", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1"),
        ),
        FixtureSpec(
            fixture_id="sim_bt_tcb",
            par_text=_equatorial_header() + _bt_binary_block(),
            mjds=BINARY_MJDS,
            binary="BT",
            option_tags=("BINARY=BT", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1"),
        ),
        FixtureSpec(
            fixture_id="sim_ddk_tcb",
            par_text=_equatorial_header() + _ddk_binary_block(),
            mjds=BINARY_MJDS,
            binary="DDK",
            option_tags=("BINARY=DDK", "UNITS=TCB", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB", "A1"),
        ),
        FixtureSpec(
            fixture_id="sim_dd_tdb",
            par_text=_equatorial_header(units="TDB") + """\
BINARY         DD
PB             14.3484575489552722       1
T0             52506.375998904087385     1
A1             8.8016496478606525596     1
OM             181.89033874817114354     1
ECC            0.00017368183334286273809 1
M2             0.72479357377044024887    1
SINI           0.80101353240028137781    1
EPHEM           DE405
CLK             TT(BIPM2011)
""",
            mjds=BINARY_MJDS,
            binary="DD",
            option_tags=("BINARY=DD", "UNITS=TDB", "DILATEFREQ=Y"),
            designmatrix_params=("F0",),
        ),
        FixtureSpec(
            fixture_id="sim_dd_ecliptic_tcb",
            par_text=_ecliptic_header() + """\
BINARY         DD
PB             14.3484575489552722       1
T0             52506.375998904087385     1
A1             8.8016496478606525596     1
OM             181.89033874817114354     1
ECC            0.00017368183334286273809 1
M2             0.72479357377044024887    1
SINI           0.80101353240028137781    1
""",
            mjds=BINARY_MJDS,
            binary="DD",
            option_tags=("BINARY=DD", "UNITS=TCB", "COORDS=ecliptic", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "PB"),
        ),
        FixtureSpec(
            fixture_id="sim_t2_track2_pn",
            par_text=_equatorial_header() + _t2_binary_block(track="-2"),
            mjds=TRACK2_MJDS,
            binary="T2",
            option_tags=("BINARY=T2", "UNITS=TCB", "TRACK=-2", "TIM=-pn", "DILATEFREQ=Y"),
            tim_patch=TimPatch(use_libstempo_pulse_numbers=True),
        ),
        FixtureSpec(
            fixture_id="sim_t2_track2_addsat",
            par_text=_equatorial_header() + _t2_binary_block(track="-2"),
            mjds=ADDSAT_MJDS,
            binary="T2",
            flags=_track2_addsat_flags(),
            option_tags=("BINARY=T2", "UNITS=TCB", "TRACK=-2", "TIM=-pn", "TIM=-addsat", "DILATEFREQ=Y"),
            tim_patch=TimPatch(
                use_libstempo_pulse_numbers=True,
                addsat_by_toa_index={3: -1, 6: +1},
            ),
        ),
        FixtureSpec(
            fixture_id="sim_t2_multisys",
            par_text=_equatorial_header() + _t2_binary_block(),
            mjds=BINARY_MJDS,
            binary="T2",
            observatory="ao",
            flags=(
                "-sys SIM.A.1400",
                "-sys SIM.A.1400",
                "-sys SIM.B.1400",
                "-sys SIM.B.1400",
                "-sys SIM.A.1400",
                "-sys SIM.A.1400",
                "-sys SIM.B.1400",
                "-sys SIM.B.1400",
            ),
            option_tags=("BINARY=T2", "UNITS=TCB", "TIM=multi-sys", "DILATEFREQ=Y"),
        ),
        FixtureSpec(
            fixture_id="sim_fd_tcb",
            par_text=_equatorial_header() + _t2_binary_block() + "FD1            2.7124139917564653244e-05 1\n",
            mjds=BINARY_MJDS,
            binary="T2",
            option_tags=("BINARY=T2", "UNITS=TCB", "FD", "DILATEFREQ=Y"),
            designmatrix_params=("F0", "FD1"),
        ),
        FixtureSpec(
            fixture_id="sim_dilatefreq_no",
            par_text=_equatorial_header(dilatefreq="N"),
            mjds=ISOLATED_MJDS[:8],
            option_tags=("BINARY=isolated", "UNITS=TCB", "DILATEFREQ=N"),
            designmatrix_params=("F0", "DM"),
        ),
    )


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(normalize_text(text).rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def inject_tim_flags(
    tim_path: Path,
    *,
    pn_by_toa_index: dict[int, int] | None = None,
    addsat_by_toa_index: dict[int, int] | None = None,
) -> None:
    pn_by_toa_index = pn_by_toa_index or {}
    addsat_by_toa_index = addsat_by_toa_index or {}
    lines = tim_path.read_text(encoding="utf-8").splitlines()
    data_seen = -1
    out: list[str] = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0] in {"FORMAT", "MODE"}:
            out.append(line)
            continue
        data_seen += 1
        cleaned: list[str] = []
        skip_next = False
        for token in parts:
            if skip_next:
                skip_next = False
                continue
            if token in {"-pn", "-pnadd", "-addsat"}:
                skip_next = True
                continue
            cleaned.append(token)
        if data_seen in pn_by_toa_index:
            cleaned.extend(["-pn", str(pn_by_toa_index[data_seen])])
        if data_seen in addsat_by_toa_index:
            cleaned.extend(["-addsat", f"{addsat_by_toa_index[data_seen]:+d}"])
        out.append(" ".join(cleaned))
    write_text(tim_path, "\n".join(out))


def inject_pulse_numbers_from_libstempo(par_path: Path, tim_path: Path) -> dict[int, int]:
    import libstempo

    psr = libstempo.tempopulsar(str(par_path), str(tim_path), dofit=False)
    pulse_numbers = [int(value) for value in psr.pulsenumbers()]
    inject_tim_flags(tim_path, pn_by_toa_index={idx: value for idx, value in enumerate(pulse_numbers)})
    return {idx: value for idx, value in enumerate(pulse_numbers)}


def write_fixture(out_dir: Path, spec: FixtureSpec) -> dict[str, Any]:
    from libstempo.toasim import fakepulsar

    fixture_dir = out_dir / spec.fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    par_path = fixture_dir / f"{spec.fixture_id}.par"
    tim_path = fixture_dir / f"{spec.fixture_id}.tim"
    write_text(par_path, spec.par_text)
    psr = fakepulsar(
        parfile=str(par_path),
        obstimes=np.asarray(spec.mjds, dtype=np.longdouble),
        toaerr=spec.toaerr_us,
        freq=spec.freq_mhz,
        observatory=spec.observatory,
        flags=spec.flags,
        iters=3,
    )
    psr.savetim(str(tim_path))
    pn_by_toa_index = dict(spec.tim_patch.pn_by_toa_index)
    if spec.tim_patch.use_libstempo_pulse_numbers:
        pn_by_toa_index = inject_pulse_numbers_from_libstempo(par_path, tim_path)
    if spec.tim_patch.addsat_by_toa_index or (
        pn_by_toa_index and not spec.tim_patch.use_libstempo_pulse_numbers
    ):
        inject_tim_flags(
            tim_path,
            pn_by_toa_index=pn_by_toa_index,
            addsat_by_toa_index=spec.tim_patch.addsat_by_toa_index,
        )
    return {
        "binary": spec.binary or "isolated",
        "designmatrix_params": list(spec.designmatrix_params),
        "generated": True,
        "generator": GENERATOR_PATH,
        "id": spec.fixture_id,
        "option_tags": list(spec.option_tags),
        "par": f"{spec.fixture_id}/{spec.fixture_id}.par",
        "parity_status": "green_required",
        "provenance": "simulated_libstempo_fakepulsar",
        "tim": f"{spec.fixture_id}/{spec.fixture_id}.tim",
        "toa_count": len(spec.mjds),
    }


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = out_dir / "manifest.json"
    sorted_rows = sorted(rows, key=lambda row: row["id"])
    payload = json.dumps(sorted_rows, indent=2, sort_keys=True) + "\n"
    write_text(manifest_path, payload)


def generate_all(out_dir: Path) -> list[dict[str, Any]]:
    specs = build_fixture_specs()
    ids = [spec.fixture_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate fixture ids in generator specs")
    missing = set(CANONICAL_FIXTURE_IDS) - set(ids)
    extra = set(ids) - set(CANONICAL_FIXTURE_IDS)
    if missing or extra:
        raise RuntimeError(f"Canonical fixture id mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [write_fixture(out_dir, spec) for spec in specs]
    write_manifest(out_dir, rows)
    return rows


def compare_trees(expected_dir: Path, actual_dir: Path) -> list[str]:
    errors: list[str] = []
    expected_files = sorted(p for p in expected_dir.rglob("*") if p.is_file())
    actual_files = sorted(p for p in actual_dir.rglob("*") if p.is_file())
    expected_rel = [p.relative_to(expected_dir) for p in expected_files]
    actual_rel = [p.relative_to(actual_dir) for p in actual_files]
    if expected_rel != actual_rel:
        errors.append(f"file tree mismatch: expected={expected_rel} actual={actual_rel}")
        return errors
    for rel in expected_rel:
        exp = expected_dir / rel
        act = actual_dir / rel
        if rel.suffix == ".json":
            expected_payload = json.loads(exp.read_text(encoding="utf-8"))
            actual_payload = json.loads(act.read_text(encoding="utf-8"))
            if expected_payload != actual_payload:
                errors.append(f"manifest content mismatch: {rel}")
            continue
        if normalize_text(exp.read_text(encoding="utf-8")) != normalize_text(
            act.read_text(encoding="utf-8")
        ):
            errors.append(f"text content mismatch: {rel}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and compare with committed artifacts.",
    )
    args = parser.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory(prefix="jug_tempo2_sim_check_") as tmp:
            tmp_dir = Path(tmp)
            generate_all(tmp_dir)
            errors = compare_trees(OUT_DIR, tmp_dir)
            if errors:
                print("Generated tempo2 sim fixtures are stale:", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                return 1
        print("Generated tempo2 sim fixtures are up to date.")
        return 0

    generate_all(OUT_DIR)
    print(f"Wrote {len(CANONICAL_FIXTURE_IDS)} fixtures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
