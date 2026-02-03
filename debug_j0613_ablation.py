
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
import pint.models
import pint.fitter
import pint.logging
from pint.models import get_model_and_toas
from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path

pint.logging.setup(level="WARNING")

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J0613-0200_tdb.par'
tim_file = data_dir / 'J0613-0200.tim'

def run_test(name, modify_par_func=None):
    print(f"\n--- TEST: {name} ---")
    
    # Create temp par file
    temp_par = data_dir / f'J0613_debug_{name}.par'
    with open(par_file, 'r') as f:
        lines = f.readlines()
    
    with open(temp_par, 'w') as f:
        for line in lines:
            if modify_par_func:
                new_line = modify_par_func(line)
                if new_line is not None:
                    f.write(new_line)
            else:
                f.write(line)
                
    # JUG
    jug_res = compute_residuals_simple(str(temp_par), str(tim_file), verbose=False)
    jug_rms = jug_res['weighted_rms_us']
    
    # PINT
    try:
        m, t = get_model_and_toas(str(temp_par), str(tim_file))
        fitter = pint.fitter.WLSFitter(t, m)
        pint_rms = fitter.resids.rms_weighted().to('us').value
    except Exception as e:
        print(f"PINT Error: {e}")
        pint_rms = float('nan')
        
    print(f"JUG RMS:  {jug_rms:.6f} us")
    print(f"PINT RMS: {pint_rms:.6f} us")
    print(f"Diff:     {abs(jug_rms - pint_rms):.6f} us")

# 1. Baseline
run_test("Baseline")

# 2. Remove DM1/DM2
def remove_dm_derivs(line):
    if line.startswith('DM1') or line.startswith('DM2'):
        return None # Skip
    return line
run_test("No_DM1_DM2", remove_dm_derivs)

# 3. Remove FD
def remove_fd(line):
    if line.startswith('FD'):
        return None
    return line
run_test("No_FD", remove_fd)

# 4. Remove Both
def remove_both(line):
    if line.startswith('DM1') or line.startswith('DM2') or line.startswith('FD'):
        return None
    return line
run_test("No_DM_FD", remove_both)
