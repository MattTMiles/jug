"""Writer for pulsar timing .tim files.

Supports commenting out deleted TOAs with the Tempo convention ('C ' prefix)
rather than removing them, preserving the original file structure.
"""

from pathlib import Path
from typing import Set, Union


def write_tim_file(
    original_path: Union[str, Path],
    output_path: Union[str, Path],
    deleted_indices: Set[int],
) -> int:
    """Write a tim file with deleted TOAs commented out.

    Reads the original tim file line-by-line.  TOA lines whose index
    (in parse order) is in *deleted_indices* are prefixed with ``C ``
    instead of being removed, following the Tempo/Tempo2 comment
    convention.  Non-TOA lines (FORMAT, MODE, existing comments, blank
    lines) are preserved unchanged.

    Parameters
    ----------
    original_path : str or Path
        Path to the original .tim file.
    output_path : str or Path
        Destination path for the written file.
    deleted_indices : set of int
        Zero-based indices of TOAs to comment out.

    Returns
    -------
    int
        Number of TOAs written (not commented out).
    """
    original_path = Path(original_path)
    output_path = Path(output_path)

    with open(original_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    toa_index = 0
    n_kept = 0

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines, comments, FORMAT/MODE directives as-is
        if (not stripped
                or stripped.startswith('C ')
                or stripped.startswith('c ')
                or stripped.startswith('#')
                or stripped.upper().startswith('FORMAT')
                or stripped.upper().startswith('MODE')):
            output_lines.append(line)
            continue

        # Heuristic: a TOA line has >= 3 whitespace-separated tokens
        parts = stripped.split()
        if len(parts) >= 3:
            if toa_index in deleted_indices:
                # Comment out with 'C ' prefix, preserving original content
                output_lines.append('C ' + line.lstrip())
            else:
                output_lines.append(line)
                n_kept += 1
            toa_index += 1
        else:
            # Not a TOA line — keep as-is
            output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    return n_kept


def write_tim_file_with_pn(
    original_path: Union[str, Path],
    output_path: Union[str, Path],
    pulse_numbers,
    deleted_indices: Set[int] = None,
) -> int:
    """Write a tim file with -pn (pulse number) flags on each TOA line.

    Reads the original tim file line-by-line.  For each active TOA line,
    adds or updates the ``-pn <value>`` flag using the corresponding
    entry from *pulse_numbers*.  The pulse numbers are relativised so
    the first active TOA has ``-pn 0``.

    Parameters
    ----------
    original_path : str or Path
        Path to the original .tim file.
    output_path : str or Path
        Destination path for the written file.
    pulse_numbers : array-like
        Pulse number for each TOA (same length and order as parsed TOAs).
        These are the absolute pulse numbers from the residual calculation;
        they will be shifted so the minimum active value is 0.
    deleted_indices : set of int, optional
        Zero-based indices of deleted TOAs to comment out.

    Returns
    -------
    int
        Number of active TOAs written.
    """
    import re
    import numpy as np

    original_path = Path(original_path)
    output_path = Path(output_path)
    if deleted_indices is None:
        deleted_indices = set()

    pn = np.asarray(pulse_numbers, dtype=np.int64)

    # Relativise: subtract the minimum pulse number among active TOAs
    active_mask = np.ones(len(pn), dtype=bool)
    for idx in deleted_indices:
        if 0 <= idx < len(active_mask):
            active_mask[idx] = False
    if np.any(active_mask):
        pn = pn - pn[active_mask].min()

    with open(original_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    toa_index = 0
    n_kept = 0
    # Regex to match an existing -pn flag and its value
    pn_re = re.compile(r'-pn\s+\S+')

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines, comments, FORMAT/MODE directives as-is
        if (not stripped
                or stripped.startswith('C ')
                or stripped.startswith('c ')
                or stripped.startswith('#')
                or stripped.upper().startswith('FORMAT')
                or stripped.upper().startswith('MODE')):
            output_lines.append(line)
            continue

        # Heuristic: a TOA line has >= 3 whitespace-separated tokens
        parts = stripped.split()
        if len(parts) >= 3:
            if toa_index in deleted_indices:
                output_lines.append('C ' + line.lstrip())
            else:
                # Strip trailing whitespace/newline, remove old -pn flag if present
                clean = line.rstrip('\n').rstrip()
                clean = pn_re.sub('', clean).rstrip()
                # Append new -pn flag
                clean += f' -pn {pn[toa_index]}\n'
                output_lines.append(clean)
                n_kept += 1
            toa_index += 1
        else:
            output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    return n_kept
