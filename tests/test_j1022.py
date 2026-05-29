
import sys
from pathlib import Path

try:
    from tests.test_paths import get_j1022_paths, skip_if_missing
except ImportError:
    from test_paths import get_j1022_paths, skip_if_missing

import numpy as np
from jug.io.par_reader import parse_par_file
from jug.fitting.optimized_fitter import fit_parameters_optimized

par_path, tim_path = get_j1022_paths()
if not skip_if_missing(par_path, tim_path, "j1022"):
    print("\nSKIPPED: J1022+1001 test data not available")
    sys.exit(0)

par_file = par_path
tim_file = tim_path
data_dir = par_file.parent

# Perturb
original_params = parse_par_file(par_file)
perturbed_par = data_dir / 'J1022+1001_tdb_perturbed_debug.par'
with open(par_file, 'r') as f:
    lines = f.readlines()
with open(perturbed_par, 'w') as f:
    for line in lines:
        if line.strip().startswith('H3 '):
            parts = line.split()
            parts[1] = f'{original_params["H3"] * 1.5:.15e}'
            f.write('  '.join(parts) + '\n')
        elif line.strip().startswith('STIG '):
            parts = line.split()
            parts[1] = f'{original_params["STIG"] * 0.8:.15e}'
            f.write('  '.join(parts) + '\n')
        else:
            f.write(line)

print('Fitting H3 and STIG (max_iter=2)...')
fit_result = fit_parameters_optimized(
    str(perturbed_par), str(tim_file),
    fit_params=['H3', 'STIG'],
    max_iter=2,
    verbose=True
)
