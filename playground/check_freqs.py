
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
import numpy as np
import pint.models
import pint.toa
from pint.models import get_model_and_toas
from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path
import pint.logging
pint.logging.setup(level="WARNING")

data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
par_file = data_dir / 'J0613-0200_tdb.par'
tim_file = data_dir / 'J0613-0200.tim'

print("Loading PINT...")
m, t = get_model_and_toas(str(par_file), str(tim_file))
# PINT barycentric freqs?
# Usually t.table['freq'] is coord frame?
# Need to check how PINT computes barycentric freq for FD.
# It seems FD_delay uses `toas.get_freqs()` which might be topo?
# The definition of FD is usually log(f_bary / 1 GHz) or f_topo?
# Tempo2 uses f_bary. PINT should too.

pint_freqs = t.table['freq'] # MHz
# Check if PINT converts to barycentric inside standard models?
# Actually 'freq' column is usually topocentric.

print("Loading JUG...")
jug_res = compute_residuals_simple(str(par_file), str(tim_file), verbose=False)
jug_freqs_bary = jug_res['freq_bary_mhz']

# We need to know what freq PINT used for FD.
# Let's inspect the first few values of JUG bary frequencies vs PINT 'freq' column.
print("\n--- Frequency Comparison (First 5) ---")
print(f"{'Idx':<5} {'JUG Bary (MHz)':<15} {'PINT Col (MHz)':<15} {'Diff':<15}")
for i in range(5):
    print(f"{i:<5} {jug_freqs_bary[i]:<15.6f} {pint_freqs[i]:<15.6f} {jug_freqs_bary[i]-pint_freqs[i]:<15.6f}")

print("\n--- FD Delay Re-Check ---")
# If differences are large (doppler shift), then that explains it.
# If PINT uses Topo freq for FD, that's a bug/feature difference.
# Or if JUG's bary freq calc is wrong.

