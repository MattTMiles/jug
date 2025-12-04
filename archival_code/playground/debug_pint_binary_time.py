"""Debug what exact time PINT uses for binary delay calc."""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs

PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("Investigating PINT binary time reference...")

model = get_model(PAR_FILE)
toas = get_TOAs(TIM_FILE, planets=True)

# Get binary component
binary_comp = model.get_components_by_category()['pulsar_system'][0]

# Update binary object
binary_comp.update_binary_object(toas, None)

# Check what time it's using
idx = 0
tdb_mjd = float(toas.table['tdbld'][idx])
barycentric_time_used = binary_comp.barycentric_time[idx]

print(f"\nTOA #{idx}:")
print(f"  TDB (raw): {tdb_mjd:.15f} MJD")
print(f"  Barycentric time used by binary model: {barycentric_time_used:.15f}")
print(f"  Units: {barycentric_time_used.unit}")

# Convert to MJD for comparison
bary_time_mjd = float(barycentric_time_used.to_value('day'))
diff_days = bary_time_mjd - tdb_mjd

print(f"\nDifference:")
print(f"  {diff_days:.15f} days = {diff_days * 86400:.9f} s")

# This difference should be the sum of delays BEFORE the binary component
# Get the delays before binary
delay_before_binary = model.get_barycentric_toas(toas)[idx] - toas.table['tdbld'][idx]
print(f"\nDelay before binary (from get_barycentric_toas):")
print(f"  {float(delay_before_binary.to_value('day')):.15f} days")
print(f"  {float(delay_before_binary.to_value('s')):.9f} s")
