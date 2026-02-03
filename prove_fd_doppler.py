
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
import pint.models
import pint.toa
from pint.models import get_model_and_toas
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.derivatives_fd import compute_fd_delay
from jug.io.par_reader import parse_par_file
from pathlib import Path
import pint.logging

pint.logging.setup(level="WARNING")

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J0613-0200_tdb.par'
tim_file = data_dir / 'J0613-0200.tim'

# 1. Load Data & Params
print("Loading data...")
params = parse_par_file(str(par_file))
fd_params = {k: float(v) for k, v in params.items() if k.startswith('FD')}

# 2. Get Frequencies
print("Computing frequencies...")
# PINT (Topocentric)
m, t = get_model_and_toas(str(par_file), str(tim_file))
f_topo = np.array(t.table['freq'], dtype=np.float64)

# JUG (Barycentric)
jug_res = compute_residuals_simple(str(par_file), str(tim_file), verbose=False)
f_bary = jug_res['freq_bary_mhz']

# 3. Calculate FD Delays
print("Computing FD Delays...")
delay_using_topo = compute_fd_delay(f_topo, fd_params)
delay_using_bary = compute_fd_delay(f_bary, fd_params)

# 4. Compare
diff_sec = delay_using_bary - delay_using_topo
diff_us = diff_sec * 1e6

mean_diff = np.mean(diff_us)
rms_diff = np.std(diff_us)

print("\n" + "="*50)
print("HYPOTHESIS TEST: Barycentric vs Topocentric FD")
print("="*50)
print(f"FD(f_bary) - FD(f_topo):")
print(f"  Mean Diff: {mean_diff:.6f} µs")
print(f"  RMS Diff:  {rms_diff:.6f} µs")
print("-" * 50)

# Actual Obsverved Discrepancy (from previous run)
# Baseline Diff: 1.94 µs
# No_FD Diff:    0.75 µs
# Implied FD Diff: ~1.2 µs
target_diff = 1.936377 - 0.746861
print(f"Target Discrepancy (Observed - No_FD): {target_diff:.6f} µs")

if abs(rms_diff - target_diff) < 0.2:
    print("\n✅ PROVEN: The difference matches the Doppler shift effect!")
else:
    print("\n❌ FAILED: The numbers don't match.")
    print(f"   (Calculated {rms_diff:.3f} vs Expected {target_diff:.3f})")
