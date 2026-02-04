#!/usr/bin/env python
import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs

par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

# 1. Ordered delay component list
print("1. Ordered delay component list:")
print([c.__class__.__name__ for c in model.DelayComponent_list])

# 2. Breakdown of tdbld - get_barycentric_toas vs DM delay
print("\n2. Breakdown for first 5 TOAs:")
print("   tdbld - get_barycentric_toas(toas) vs DM delay\n")

tdbld = toas.table['tdbld']
bary_toas = model.get_barycentric_toas(toas)
diff_sec = (tdbld - np.array([b.value for b in bary_toas])) * 86400  # seconds

# Get DM delay using dispersion_time_delay with frequency
freqs = toas.table['freq']
dm_comp = model.components['DispersionDM']
dm_delay_sec = dm_comp.dispersion_time_delay(model.DM.quantity, freqs).to('s').value

print(f"{'TOA#':<6} {'tdbld-bary (s)':<18} {'DM delay (s)':<18} {'Ratio':<10}")
print("-" * 55)
for i in range(5):
    ratio = diff_sec[i] / dm_delay_sec[i] if dm_delay_sec[i] != 0 else 0
    print(f"{i:<6} {diff_sec[i]:<18.9f} {dm_delay_sec[i]:<18.9f} {ratio:<10.4f}")
