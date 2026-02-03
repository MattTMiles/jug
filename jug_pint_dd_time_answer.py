#!/usr/bin/env python
"""
FINAL ANSWER: What time does PINT DD use vs what JUG uses?
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

# ============================================================================
# PINT's barycentric time for DD
# ============================================================================
print("\n" + "="*80)
print("PINT's DD TIME VARIABLE")
print("="*80)

# PINT's formula: t_DD = tdbld - delay_before_binary
# where delay_before_binary = ss_geo + tropo + ss_shapiro + sw + dm

delay_before_binary = model.delay(toas, "BinaryDD", False)

# Get individual components
ss_geo = model.solar_system_geometric_delay(toas, None)
ss_shapiro = model.solar_system_shapiro_delay(toas, None)
tropo = model.troposphere_delay(toas)
sw = model.solar_wind_delay(toas, None)
dm = model.constant_dispersion_delay(toas, None)

print("\nDelays SUBTRACTED from tdbld before DD sees the time:")
print(f"  solar_system_geometric (Roemer): mean={ss_geo.to('s').value.mean():.6f} s, range=[{ss_geo.to('s').value.min():.6f}, {ss_geo.to('s').value.max():.6f}] s")
print(f"  troposphere:                     mean={tropo.to('s').value.mean():.12f} s")
print(f"  solar_system_shapiro:            mean={ss_shapiro.to('s').value.mean():.9f} s")
print(f"  solar_wind:                      mean={sw.to('s').value.mean():.9f} s")
print(f"  constant_dispersion (DM):        mean={dm.to('s').value.mean():.6f} s")

print(f"\nTotal delay subtracted: mean={delay_before_binary.to('s').value.mean():.6f} s")

# ============================================================================
# JUG's current barycentric time for DD
# ============================================================================
print("\n" + "="*80)
print("JUG's CURRENT DD TIME VARIABLE")
print("="*80)

# JUG uses: t_DD_jug = tdbld - roemer_shapiro_sec / SECS_PER_DAY
# where roemer_shapiro_sec = roemer + sun_shapiro + planet_shapiro

# Roemer = solar_system_geometric_delay
# Sun shapiro = part of solar_system_shapiro_delay
# Planet shapiro = rest of solar_system_shapiro_delay

roemer_shapiro_jug = ss_geo + ss_shapiro  # JUG's current approximation

print("\nDelays JUG CURRENTLY subtracts from tdbld:")
print(f"  roemer (ss_geo):   mean={ss_geo.to('s').value.mean():.6f} s")
print(f"  shapiro:           mean={ss_shapiro.to('s').value.mean():.9f} s")
print(f"\nTotal JUG subtracts: mean={roemer_shapiro_jug.to('s').value.mean():.6f} s")

# ============================================================================
# THE DISCREPANCY
# ============================================================================
print("\n" + "="*80)
print("DISCREPANCY: What JUG is MISSING")
print("="*80)

# JUG is NOT subtracting:
missing = delay_before_binary - roemer_shapiro_jug
missing_sec = missing.to('s').value

print(f"\nJUG's DD time is WRONG by:")
print(f"  mean:  {missing_sec.mean():.6f} s = {missing_sec.mean()*1e6:.3f} μs")
print(f"  range: [{missing_sec.min():.6f}, {missing_sec.max():.6f}] s")
print(f"  std:   {missing_sec.std():.6f} s = {missing_sec.std()*1e6:.3f} μs")

print("\nBreakdown of missing delays:")
print(f"  troposphere:            mean={tropo.to('s').value.mean():.12f} s ({tropo.to('s').value.mean()*1e9:.3f} ns)")
print(f"  solar_wind:             mean={sw.to('s').value.mean():.9f} s ({sw.to('s').value.mean()*1e6:.3f} μs)")  
print(f"  constant_dispersion:    mean={dm.to('s').value.mean():.6f} s ({dm.to('s').value.mean()*1e3:.3f} ms)")

# ============================================================================
# IMPACT ON ORBITAL PHASE
# ============================================================================
print("\n" + "="*80)
print("IMPACT ON BINARY DELAY")
print("="*80)

# Get binary parameters
binary_comp = model.components['BinaryDD']
PB = float(binary_comp.PB.value)  # days

# Time error maps to phase error
time_error_sec = missing_sec
orbital_phase_error = time_error_sec / (PB * 86400) * 360  # degrees

print(f"\nOrbital period: {PB:.6f} days = {PB*86400:.1f} seconds")
print(f"\nTime error in binary input:")
print(f"  mean:  {time_error_sec.mean()*1e3:.3f} ms")
print(f"  range: [{time_error_sec.min()*1e3:.3f}, {time_error_sec.max()*1e3:.3f}] ms")
print(f"\nCorresponding orbital phase error:")
print(f"  mean:  {orbital_phase_error.mean():.6f}°")
print(f"  range: [{orbital_phase_error.min():.6f}°, {orbital_phase_error.max():.6f}°]")

# ============================================================================
# SOLUTION
# ============================================================================
print("\n" + "="*80)
print("SOLUTION FOR JUG")
print("="*80)
print("""
PINT's DD model receives time:
  t_DD = tdbld - (ss_geo + tropo + ss_shapiro + sw + dm)

JUG currently uses:
  t_DD = tdbld - (ss_geo + ss_shapiro)

JUG is MISSING:
  - troposphere delay (~10 ns, negligible)
  - solar_wind delay (varies, can be ~μs near solar conjunction)
  - DM delay (~80 ms at ~1 GHz, VARIES WITH FREQUENCY!)

The DM delay is the dominant missing term and it's FREQUENCY-DEPENDENT.
This means different TOAs at different frequencies will have different
time errors in the binary model input, causing orbital-phase-dependent
residual errors.

FIX OPTIONS:
1. (Correct but complex) Compute binary delay at:
   t_DD = tdbld - (roemer + shapiro + tropo + sw + dm)
   
2. (Simpler) Note that PINT's order of delay_funcs is:
   - solar_system_geometric_delay
   - troposphere_delay
   - solar_system_shapiro_delay
   - solar_wind_delay
   - constant_dispersion_delay
   - binarymodel_delay  <-- DD sees t_DD here
   - FD_delay
   
   So binary delay is computed BEFORE FD is subtracted, but AFTER DM/SW.
   
For apples-to-apples comparison, feed JUG's dd_binary_delay() with:
  t_DD = model.get_barycentric_toas(toas)

Or equivalently:
  t_DD = tdbld - model.delay(toas, "BinaryDD", False)
""")

# Final verification
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

bary_toas = model.get_barycentric_toas(toas)
tdbld = toas.table['tdbld']
computed_delay = (np.array([float(t) for t in tdbld]) - np.array([b.value for b in bary_toas])) * 86400

print(f"\nbary_toas = tdbld - delay_before_binary:")
print(f"  Computed delay[0]: {computed_delay[0]:.9f} s")
print(f"  PINT delay[0]:     {delay_before_binary[0].to('s').value:.9f} s")
print(f"  Match: {np.allclose(computed_delay, delay_before_binary.to('s').value, atol=1e-15)}")
