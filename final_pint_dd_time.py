#!/usr/bin/env python
"""
Final verification: Exactly what time does PINT DD use?
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("Loading PINT model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"Number of TOAs: {len(toas)}")

# Get the raw TDB time
tdbld = toas.table['tdbld']

# PINT source shows:
#   corr = self.delay(toas, cutoff_component, False)  # cutoff_component = BinaryDD
#   return tbl["tdbld"] * u.day - corr
#
# So bary_toas = tdbld - (sum of delays BEFORE binary)
# The "cutoff_component=False" means DO NOT include the cutoff component itself

# Let's compute the delay sum up to (but not including) binary
print("\n" + "="*80)
print("Computing delay with cutoff at BinaryDD:")
print("="*80)

# Use PINT's delay method with cutoff
delay_before_binary = model.delay(toas, "BinaryDD", False)
print(f"delay(cutoff=BinaryDD, include_last=False):")
print(f"  [0] = {delay_before_binary[0].to('s').value:.12f} s")
print(f"  [100] = {delay_before_binary[100].to('s').value:.12f} s")

# Compute bary_toas manually
manual_bary = (tdbld * u.day) - delay_before_binary
manual_bary_mjd = np.array([m.to('d').value for m in manual_bary])

# Get PINT's bary_toas
pint_bary = model.get_barycentric_toas(toas)
pint_bary_mjd = np.array([p.value for p in pint_bary])

# Compare
diff = (manual_bary_mjd - pint_bary_mjd) * 86400
print(f"\nVerification: manual_bary vs pint_bary:")
print(f"  diff[0] = {diff[0]:.15f} s")
print(f"  diff[100] = {diff[100]:.15f} s")
print(f"  max |diff| = {np.abs(diff).max():.15f} s")

# Now let's understand what delays are included in delay_before_binary
print("\n" + "="*80)
print("What delays are SUBTRACTED to get barycentric TOA:")
print("="*80)

# Compute individual delays
ss_geo = model.solar_system_geometric_delay(toas, None)
ss_shapiro = model.solar_system_shapiro_delay(toas, None)
tropo = model.troposphere_delay(toas)
sw = model.solar_wind_delay(toas, None)
dm = model.constant_dispersion_delay(toas, None)

print(f"\nIndividual pre-binary delays at i=0:")
print(f"  solar_system_geometric: {ss_geo[0].to('s').value:.12f} s")
print(f"  troposphere:            {tropo[0].to('s').value:.12f} s")
print(f"  solar_system_shapiro:   {ss_shapiro[0].to('s').value:.12f} s")
print(f"  solar_wind:             {sw[0].to('s').value:.12f} s")
print(f"  constant_dispersion:    {dm[0].to('s').value:.12f} s")

sum_pre_binary = ss_geo[0] + tropo[0] + ss_shapiro[0] + sw[0] + dm[0]
print(f"\n  SUM of pre-binary delays: {sum_pre_binary.to('s').value:.12f} s")
print(f"  delay(cutoff=BinaryDD):   {delay_before_binary[0].to('s').value:.12f} s")

diff_sum = (sum_pre_binary - delay_before_binary[0]).to('s').value
print(f"  Difference: {diff_sum:.12f} s")

# Get the binary instance
binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

print("\n" + "="*80)
print("CRITICAL FINDING - Time variable used by PINT DD:")
print("="*80)

tdbld_float = np.array([float(t) for t in toas.table['tdbld']])
bo_t = np.array([t.value for t in bo.t])

print(f"\nbo.t[0] = {bo_t[0]:.15f} MJD")
print(f"pint_bary[0] = {pint_bary_mjd[0]:.15f} MJD")
print(f"tdbld[0] = {tdbld_float[0]:.15f} MJD")

diff_bo_bary = (bo_t[0] - pint_bary_mjd[0]) * 86400
diff_bo_tdbld = (bo_t[0] - tdbld_float[0]) * 86400
print(f"\nbo.t vs bary_toas: {diff_bo_bary:.12f} s")
print(f"bo.t vs tdbld: {diff_bo_tdbld:.12f} s")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print("""
PINT DD uses time variable: bo.t = model.get_barycentric_toas(toas)

FORMULA:
  t_DD = tdbld - (ss_geo + tropo + ss_shapiro + sw + dm)
  
Where:
  - tdbld = TDB time at solar system barycenter (from TOA table)
  - ss_geo = solar_system_geometric_delay (Roemer delay, ~±500s)
  - tropo = troposphere_delay (~10 ns)
  - ss_shapiro = solar_system_shapiro_delay (~few μs)
  - sw = solar_wind_delay (~few ns)
  - dm = constant_dispersion_delay (DM delay, ~0.08s at these freqs)

NOTE: These delays are SUBTRACTED from tdbld!
The negative sign means the DD model sees time ADVANCED by these delays.

For JUG: feed dd_binary_delay() with:
  t_DD = tdbld - (ss_geo + ss_shapiro + dm + sw)
  
Do NOT feed it raw tdbld or tdbld + delays!
""")

# Also print the range of the correction
correction = delay_before_binary.to('s').value
print(f"\nCorrection magnitude (delay_before_binary):")
print(f"  min: {correction.min():.6f} s")
print(f"  max: {correction.max():.6f} s")
print(f"  mean: {correction.mean():.6f} s")
print(f"  range: {correction.max() - correction.min():.6f} s")

# Show the dominant term
print(f"\nDominant term is ss_geo (Roemer delay):")
print(f"  ss_geo min: {ss_geo.to('s').value.min():.6f} s")
print(f"  ss_geo max: {ss_geo.to('s').value.max():.6f} s")
