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
