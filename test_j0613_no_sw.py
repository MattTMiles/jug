
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
import pint.models
import pint.fitter
import pint.logging
from pint.models import get_model_and_toas
from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path

# Suppress PINT logs
pint.logging.setup(level="WARNING")

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J0613-0200_tdb.par'
tim_file = data_dir / 'J0613-0200.tim'

# Create temp par file without NE_SW
no_sw_par = data_dir / 'J0613-0200_tdb_no_sw.par'
with open(par_file, 'r') as f:
    lines = f.readlines()
with open(no_sw_par, 'w') as f:
    for line in lines:
        if not line.strip().startswith('NE_SW') and not line.strip().startswith('TNsubtractPoly'):
            f.write(line)

print(f"Comparison without NE_SW for J0613-0200")
print("-" * 40)

# PINT
print("PINT...")
m, t = get_model_and_toas(str(no_sw_par), str(tim_file))
fitter = pint.fitter.WLSFitter(t, m)
pint_resids = fitter.resids.time_resids.to('s').value
pint_rms = fitter.resids.rms_weighted().to('us').value
print(f"PINT RMS: {pint_rms:.6f} us")

# JUG
print("JUG...")
jug_res = compute_residuals_simple(str(no_sw_par), str(tim_file), verbose=False)
jug_resids = jug_res['residuals_us'] * 1e-6
jug_rms = jug_res['weighted_rms_us']
print(f"JUG RMS: {jug_rms:.6f} us")

# Diff
diff = pint_resids - jug_resids
rms_diff_ns = np.std(diff) * 1e9
print(f"RMS Difference: {rms_diff_ns:.3f} ns")

if rms_diff_ns < 10.0:
    print("✅ Match! NE_SW was the culprit.")
else:
    print("❌ Still mismatch. NE_SW is NOT the only factor.")
