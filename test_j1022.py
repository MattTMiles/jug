
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
from pathlib import Path
from jug.io.par_reader import parse_par_file
from jug.fitting.optimized_fitter import fit_parameters_optimized

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J1022+1001_tdb.par'
tim_file = data_dir / 'J1022+1001.tim'

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
