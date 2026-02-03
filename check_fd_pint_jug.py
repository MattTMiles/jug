
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
import pint.models
import pint.toa
from pint.models import get_model_and_toas
from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path

# Suppress PINT logs
import pint.logging
pint.logging.setup(level="WARNING")

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J0613-0200_tdb.par'
tim_file = data_dir / 'J0613-0200.tim'

print("Loading PINT...")
m, t = get_model_and_toas(str(par_file), str(tim_file))

print("Computing PINT FD Delay...")
# Access FD component delay function directly if possible
# Or compute total delay and isolate?
# PINT models usually allow access to component delays
fd_comp = m.components['FD']
pint_fd_delay = fd_comp.FD_delay(t).to('s').value

print("Loading JUG...")
jug_res = compute_residuals_simple(str(par_file), str(tim_file), verbose=False)

# JUG doesn't output FD delay array directly in the dictionary unless I added it, 
# but I can recompute it using the same function since I have the freqs.
jug_freqs = jug_res['freq_bary_mhz']
from jug.fitting.derivatives_fd import compute_fd_delay

# Extract FD params from jug_res? No, need from par file or pass in.
# compute_residuals_simple doesn't return params dict?
# I can verify values manually.
from jug.io.par_reader import parse_par_file
params = parse_par_file(str(par_file))
# Filter manually for JUG function
fd_params = {k: float(v) for k, v in params.items() if k.startswith('FD')}
# compute_fd_delay(freq_mhz, fd_params)
jug_fd_delay = compute_fd_delay(jug_freqs, fd_params)

print("\n--- FD Delay Comparison (All TOAs) ---")
diff = pint_fd_delay - jug_fd_delay
print(f"Mean Diff: {np.mean(diff):.6e} s")
print(f"RMS Diff:  {np.std(diff):.6e} s")
print(f"Max Diff:  {np.max(np.abs(diff)):.6e} s")

# Check if freqs diverge at high/low end
import matplotlib.pyplot as plt
# Just print extremes
min_idx = np.argmin(jug_freqs)
max_idx = np.argmax(jug_freqs)
print(f"Min Freq ({jug_freqs[min_idx]:.3f}): Diff = {diff[min_idx]:.6e}")
print(f"Max Freq ({jug_freqs[max_idx]:.3f}): Diff = {diff[max_idx]:.6e}")
