#!/usr/bin/env python
"""
Trace exactly what model.get_barycentric_toas() computes vs tdbld.
Identify which delays are subtracted to create the "barycentric TOA" for the binary model.
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("="*80)
print("Loading PINT model and TOAs...")
print("="*80)

model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"\nNumber of TOAs: {len(toas)}")

# Get all the key time quantities
print("\n" + "="*80)
print("Time quantities comparison:")
print("="*80)

# Raw TDB from TOA table
tdbld = np.array([float(t) for t in toas.table['tdbld']])
print(f"tdbld[0] = {tdbld[0]:.15f} (MJD TDB)")
print(f"tdbld[100] = {tdbld[100]:.15f} (MJD TDB)")

# get_barycentric_toas
bary_toas = model.get_barycentric_toas(toas)
bary_mjd = np.array([b.value for b in bary_toas])
print(f"\nget_barycentric_toas[0] = {bary_mjd[0]:.15f} (MJD)")
print(f"get_barycentric_toas[100] = {bary_mjd[100]:.15f} (MJD)")

# Difference
diff_sec = (bary_mjd - tdbld) * 86400
print(f"\nDifference (bary_toas - tdbld):")
print(f"  [0]: {diff_sec[0]:.9f} seconds")
print(f"  [100]: {diff_sec[100]:.9f} seconds")
print(f"  min: {diff_sec.min():.9f} seconds")
print(f"  max: {diff_sec.max():.9f} seconds")
print(f"  range: {diff_sec.max() - diff_sec.min():.9f} seconds")

# Now let's compute individual delay components
print("\n" + "="*80)
print("Computing individual delay components:")
print("="*80)

# Solar system geometric delay
ss_geo = model.solar_system_geometric_delay(toas, None)
print(f"\nsolar_system_geometric_delay:")
print(f"  [0] = {ss_geo[0].to('s').value:.9f} s")
print(f"  [100] = {ss_geo[100].to('s').value:.9f} s")

# Solar system Shapiro delay
ss_shapiro = model.solar_system_shapiro_delay(toas, None)
print(f"\nsolar_system_shapiro_delay:")
print(f"  [0] = {ss_shapiro[0].to('s').value:.12f} s")
print(f"  [100] = {ss_shapiro[100].to('s').value:.12f} s")

# Troposphere delay
try:
    tropo = model.troposphere_delay(toas)
    print(f"\ntroposphere_delay:")
    print(f"  [0] = {tropo[0].to('s').value:.12f} s")
    print(f"  [100] = {tropo[100].to('s').value:.12f} s")
except Exception as e:
    print(f"\ntroposphere_delay: {e}")

# DM delay
dm_delay = model.constant_dispersion_delay(toas, None)
print(f"\nconstant_dispersion_delay (DM):")
print(f"  [0] = {dm_delay[0].to('s').value:.9f} s")
print(f"  [100] = {dm_delay[100].to('s').value:.9f} s")

# Solar wind delay
sw_delay = model.solar_wind_delay(toas, None)
print(f"\nsolar_wind_delay:")
print(f"  [0] = {sw_delay[0].to('s').value:.9f} s")
print(f"  [100] = {sw_delay[100].to('s').value:.9f} s")

# FD delay
try:
    fd_delay = model.FD_delay(toas, None)
    print(f"\nFD_delay:")
    print(f"  [0] = {fd_delay[0].to('s').value:.12f} s")
    print(f"  [100] = {fd_delay[100].to('s').value:.12f} s")
except Exception as e:
    print(f"\nFD_delay: {e}")

# Binary delay
binary_delay = model.binarymodel_delay(toas, None)
print(f"\nbinarymodel_delay:")
print(f"  [0] = {binary_delay[0].to('s').value:.9f} s")
print(f"  [100] = {binary_delay[100].to('s').value:.9f} s")

# Check the cumulative delays
print("\n" + "="*80)
print("Understanding get_barycentric_toas():")
print("="*80)

# Sum of "pre-binary" delays: ss_geo + ss_shapiro
pre_binary_sum = ss_geo + ss_shapiro
print(f"\nss_geo + ss_shapiro:")
print(f"  [0] = {pre_binary_sum[0].to('s').value:.9f} s")
print(f"  [100] = {pre_binary_sum[100].to('s').value:.9f} s")

# Compare with (bary_toas - tdbld)
print(f"\n(bary_toas - tdbld):")
print(f"  [0] = {diff_sec[0]:.9f} s")
print(f"  [100] = {diff_sec[100]:.9f} s")

# Check if get_barycentric_toas = tdbld + ss_geo + ss_shapiro
test_bary = tdbld + (ss_geo + ss_shapiro).to('d').value
diff_test = (test_bary - bary_mjd) * 86400
print(f"\nTest: tdbld + ss_geo + ss_shapiro matches bary_toas?")
print(f"  Difference [0]: {diff_test[0]:.12f} s")
print(f"  Difference [100]: {diff_test[100]:.12f} s")

# Let's look at the actual source of get_barycentric_toas
print("\n" + "="*80)
print("Checking get_barycentric_toas() source:")
print("="*80)

import inspect
src = inspect.getsource(model.get_barycentric_toas)
print(src[:2000])

# Check which delays are in delay_funcs before binary
print("\n" + "="*80)
print("Delays before binary in model.delay_funcs:")
print("="*80)

for i, func in enumerate(model.delay_funcs):
    func_name = func.__name__
    print(f"  {i}: {func_name}")
    if 'binary' in func_name.lower():
        print(f"     ^^^ BINARY MODEL HERE ^^^")
        break

# Check what add_delay_to_toas does
print("\n" + "="*80)
print("Testing with PINT's own add_delay_to_toas:")
print("="*80)

# PINT computes delays cumulatively
# Let's see what the accumulated delay is at the point where binary model sees it
from pint import toa as pint_toa
import astropy.units as u

# Manual computation of barycentric TOAs following PINT's logic
# tdbld is the input time, then we ADD the "earlier" delays to get barycentric TOA

# The delays before binary are: ss_geo, tropo, ss_shapiro, sw
# But wait - let's check if troposphere is there
pre_binary_delays = ss_geo + ss_shapiro
try:
    tropo = model.troposphere_delay(toas)
    pre_binary_delays = pre_binary_delays + tropo
    print("Including troposphere delay")
except:
    print("No troposphere delay")

# Check if solar wind comes before or after binary
print("\nDelay ordering check:")
for i, func in enumerate(model.delay_funcs):
    print(f"  {i}: {func.__name__}")
