"""Build the self-contained dataset for notebooks/quickstart/.

Copies the J1909-3744 test data under clean names and writes a *perturbed*
starting ephemeris, so the quick-start example demonstrates a fit converging
from a visibly wrong model rather than starting already converged.

Every free parameter is offset by ``SIGMA_MULT`` times its par-file
uncertainty (sign chosen by a fixed seed, so the dataset is reproducible).

    python tools/make_quickstart_dataset.py
"""

import shutil
from pathlib import Path

import numpy as np

from jug.io.par_reader import format_dec, format_ra, parse_dec, parse_ra

REPO = Path(__file__).resolve().parent.parent
SRC_PAR = REPO / "tests" / "data_golden" / "J1909_parity_noise.par"
SRC_TIM = REPO / "tests" / "data_golden" / "J1909_parity.tim"
OUT_DIR = REPO / "notebooks" / "quickstart"

SIGMA_MULT = 3.0   # offset each free parameter by this many sigma
SEED = 20260803


def perturb_par(src: Path, dest: Path, sigma_mult: float = SIGMA_MULT) -> dict:
    """Write *src* to *dest* with every fitted parameter offset by n sigma.

    Returns {name: (original, perturbed)} for reporting.
    """
    rng = np.random.default_rng(SEED)
    moved = {}
    out_lines = []

    for line in src.read_text().splitlines():
        parts = line.split()
        # Fitted parameters look like:  NAME  VALUE  1  SIGMA
        if len(parts) >= 4 and parts[2] == "1":
            name, value, sigma = parts[0], parts[1], float(parts[3])
            sign = 1.0 if rng.random() < 0.5 else -1.0
            offset = sign * sigma_mult * sigma

            if name in ("RAJ", "DECJ"):
                # Sexagesimal string; uncertainty is in radians.
                to_rad = parse_ra if name == "RAJ" else parse_dec
                to_str = format_ra if name == "RAJ" else format_dec
                original = to_rad(value)
                new = original + offset
                new_str = to_str(float(new))
            else:
                # longdouble keeps F0/TASC precision through the round trip.
                original = np.longdouble(value)
                new = original + np.longdouble(offset)
                new_str = f"{new:.20g}"

            moved[name] = (float(original), float(new))
            out_lines.append(f"{name:<12} {new_str} 1 {parts[3]}")
        else:
            out_lines.append(line)

    dest.write_text("\n".join(out_lines) + "\n")
    return moved


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC_TIM, OUT_DIR / "J1909-3744.tim")
    moved = perturb_par(SRC_PAR, OUT_DIR / "J1909-3744.par")

    # Keep the converged ephemeris alongside for reference/verification.
    shutil.copy(SRC_PAR, OUT_DIR / "J1909-3744_converged.par")

    print(f"wrote {OUT_DIR}/J1909-3744.par  ({len(moved)} parameters "
          f"offset by {SIGMA_MULT} sigma, seed {SEED})")
    for name, (old, new) in moved.items():
        print(f"  {name:<8} {old:>24.15g} -> {new:.15g}")


if __name__ == "__main__":
    main()
