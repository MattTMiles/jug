#!/usr/bin/env python
"""
Deep dive: Why does JUG DD differ from PINT DD by ~289 μs even at same time,
yet end-to-end residuals only differ by ~1.1 μs?
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

# Import JUG DD
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
from jug.delays.binary_dd import dd_binary_delay

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("Loading PINT model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"Number of TOAs: {len(toas)}")

# Get binary component and update it
binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

# Get PINT's barycentric time
bary_toas = model.get_barycentric_toas(toas)
t_dd = np.array([float(b.value) for b in bary_toas], dtype=np.float64)

print("\n" + "="*80)
print("Inspecting PINT DD intermediate values:")
print("="*80)

# Check what time PINT is using
print(f"\nbo.t (PINT's internal time):")
print(f"  bo.t[0] = {bo.t[0].value:.15f} d")
print(f"  t_dd[0] = {t_dd[0]:.15f} d")
print(f"  Match: {np.isclose(bo.t[0].value, t_dd[0], rtol=1e-14)}")

# Get PINT intermediates
print(f"\nPINT DD intermediates at i=0:")
print(f"  T0 = {bo.T0.value:.15f} d")
print(f"  tt0 = {bo.tt0[0].to('s').value:.9f} s = {bo.tt0[0].to('d').value:.15f} d")
print(f"  PB = {bo.PB.value:.15f} d")
print(f"  orbits = {bo.orbits()[0]:.15f}")
print(f"  M = {bo.M()[0]:.15f} rad")
print(f"  E = {bo.E()[0]:.15f} rad")
print(f"  nu = {bo.nu()[0]:.15f} rad")
print(f"  ecc = {bo.ecc()[0]:.15f}")
print(f"  omega = {bo.omega()[0]:.15f} rad = {np.rad2deg(bo.omega()[0]):.15f} deg")
print(f"  a1 = {bo.a1()[0]:.15f} ls")

# Get PINT's full delay breakdown
print(f"\nPINT DD delay components:")
print(f"  delayR (Roemer) = {bo.delayR()[0].to('s').value:.15f} s")
print(f"  delayA (aberration) = {bo.delayA()[0].to('s').value:.15f} s")
print(f"  delayE (Einstein) = {bo.delayE()[0].to('s').value:.15f} s")
print(f"  delayS (Shapiro) = {bo.delayS()[0].to('s').value:.15f} s")

pint_total = bo.delayR()[0] + bo.delayA()[0] + bo.delayE()[0] + bo.delayS()[0]
pint_delay_direct = binary_comp.binarymodel_delay(toas, None)
print(f"\nPINT total (R+A+E+S) = {pint_total.to('s').value:.15f} s")
print(f"PINT binarymodel_delay = {pint_delay_direct[0].to('s').value:.15f} s")

# Now compute JUG DD at same time and compare intermediates
print("\n" + "="*80)
print("Computing JUG DD for direct comparison:")
print("="*80)

# Extract parameters exactly as PINT has them (with units)
pb = float(bo.PB.value)  # days
t0 = float(bo.T0.value)  # days  
# PINT uses "ls" (light-second) as a unit
a1_val = bo.a1()[0]
a1 = float(a1_val.value) if hasattr(a1_val, 'value') else float(a1_val)  # light-seconds (time-dependent!)
ecc_val = bo.ecc()[0]
ecc = float(ecc_val.value) if hasattr(ecc_val, 'value') else float(ecc_val)  # (time-dependent!)
om_deg = float(binary_comp.OM.value)  # degrees (initial)
gamma = float(binary_comp.GAMMA.value) if binary_comp.GAMMA.value is not None else 0.0
pbdot = float(binary_comp.PBDOT.value) if binary_comp.PBDOT.value is not None else 0.0
omdot = float(binary_comp.OMDOT.value) if binary_comp.OMDOT.value is not None else 0.0
xdot = float(binary_comp.A1DOT.value) if hasattr(binary_comp, 'A1DOT') and binary_comp.A1DOT.value is not None else 0.0
edot = float(binary_comp.EDOT.value) if hasattr(binary_comp, 'EDOT') and binary_comp.EDOT.value is not None else 0.0
sini = float(binary_comp.SINI.value) if binary_comp.SINI.value is not None else 0.0
m2 = float(binary_comp.M2.value) if binary_comp.M2.value is not None else 0.0

# Note: PINT's values at epoch vs values at TOA
print(f"\nPINT time-dependent values at TOA[0]:")
print(f"  a1(t) = {bo.a1()[0]:.15f} ls")
print(f"  ecc(t) = {bo.ecc()[0]:.15f}")
print(f"  omega(t) = {np.rad2deg(bo.omega()[0]):.15f} deg")
print(f"  pb(t) = {bo.pb()[0].value:.15f} d")

print(f"\nPINT epoch values (from parameter file):")
print(f"  A1 = {float(binary_comp.A1.value):.15f} ls")
print(f"  ECC = {float(binary_comp.ECC.value):.15f}")
print(f"  OM = {float(binary_comp.OM.value):.15f} deg")
print(f"  PB = {float(binary_comp.PB.value):.15f} d")

# JUG uses epoch values and applies time corrections internally
# Let's trace through JUG's computation
t_test = t_dd[0]
pb_epoch = float(binary_comp.PB.value)
a1_epoch = float(binary_comp.A1.value)
ecc_epoch = float(binary_comp.ECC.value)
om_epoch_deg = float(binary_comp.OM.value)

print(f"\n" + "="*80)
print("JUG internal computation trace:")
print("="*80)

# JUG's DD formulas (from binary_dd.py)
SECS_PER_DAY = 86400.0

# Time since T0
tt0_jug = (t_test - t0) * SECS_PER_DAY
print(f"  tt0 = {tt0_jug:.9f} s")
print(f"  PINT tt0 = {bo.tt0[0].to('s').value:.9f} s")
print(f"  Diff = {(tt0_jug - bo.tt0[0].to('s').value)*1e9:.3f} ns")

# Time-dependent orbital period (JUG formula)
pb_sec = pb_epoch * SECS_PER_DAY
pb_current_jug = pb_sec * (1 + 0.5 * pbdot * tt0_jug / pb_sec)
print(f"\n  pb_current (JUG) = {pb_current_jug:.15f} s = {pb_current_jug/SECS_PER_DAY:.15f} d")
print(f"  pb(t) (PINT) = {bo.pb()[0].value:.15f} d")

# Mean anomaly
n_jug = 2 * np.pi / pb_current_jug
M_jug = n_jug * tt0_jug  # radians
# Reduce to [0, 2π]
M_jug_wrapped = M_jug % (2*np.pi)
print(f"\n  M (JUG, wrapped) = {M_jug_wrapped:.15f} rad")
print(f"  M (PINT) = {bo.M()[0]:.15f} rad")

# Number of orbits
orbits_jug = M_jug / (2*np.pi)
print(f"\n  orbits (JUG) = {orbits_jug:.15f}")
print(f"  orbits (PINT) = {bo.orbits()[0]:.15f}")

# Time-dependent eccentricity
ecc_jug = ecc_epoch + edot * tt0_jug
print(f"\n  ecc (JUG) = {ecc_jug:.15f}")
print(f"  ecc (PINT) = {bo.ecc()[0]:.15f}")

# Time-dependent a1
a1_jug = a1_epoch + xdot * tt0_jug
print(f"\n  a1 (JUG) = {a1_jug:.15f} ls")
print(f"  a1 (PINT) = {bo.a1()[0]:.15f} ls")

# Time-dependent omega
# JUG: omega_current = omega_epoch + omdot * (tt0 / SECS_PER_YEAR)
# where SECS_PER_YEAR = 365.25 * SECS_PER_DAY
SECS_PER_YEAR = 365.25 * SECS_PER_DAY
om_current_deg_jug = om_epoch_deg + omdot * (tt0_jug / SECS_PER_YEAR)
om_current_rad_jug = np.deg2rad(om_current_deg_jug)
print(f"\n  omega (JUG) = {om_current_deg_jug:.15f} deg = {om_current_rad_jug:.15f} rad")
print(f"  omega (PINT) = {np.rad2deg(bo.omega()[0]):.15f} deg = {bo.omega()[0]:.15f} rad")

# Now let's compute JUG binary delay
jug_delay = float(dd_binary_delay(
    t_test, pb_epoch, a1_epoch, ecc_epoch, om_epoch_deg, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
))

print(f"\n" + "="*80)
print("Final comparison:")
print("="*80)
print(f"  JUG DD delay = {jug_delay:.15f} s")
print(f"  PINT DD delay = {pint_delay_direct[0].to('s').value:.15f} s")
print(f"  Difference = {(jug_delay - pint_delay_direct[0].to('s').value)*1e6:.6f} μs")

# Check PINT's delay breakdown method
print("\n" + "="*80)
print("PINT DDdelay method check:")
print("="*80)

# DDdelay returns the full delay
pint_dd = bo.DDdelay()
print(f"  bo.DDdelay()[0] = {pint_dd[0].to('s').value:.15f} s")

# Check all the components again
print(f"\nDetailed PINT components:")
Dre = bo.Dre()[0]
Drep = bo.Drep()[0]
Drepp = bo.Drepp()[0]
nhat = bo.nhat()[0]
print(f"  Dre = {Dre:.15f}")
print(f"  Drep = {Drep:.15f}") 
print(f"  Drepp = {Drepp:.15f}")
print(f"  nhat = {nhat.to('1/s').value:.15e} /s")

# The key question: what exactly does PINT's DD return?
# Let's check if there's a unit mismatch or formula difference
print("\n" + "="*80)
print("Checking for systematic offset:")
print("="*80)

# Compute mean offset
jug_delays = np.array([float(dd_binary_delay(
    t, pb_epoch, a1_epoch, ecc_epoch, om_epoch_deg, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
)) for t in t_dd[:100]])
pint_delays = pint_delay_direct[:100].to('s').value

diff = jug_delays - pint_delays
print(f"  Mean offset (JUG - PINT): {diff.mean()*1e6:.3f} μs")
print(f"  Offset std: {diff.std()*1e6:.3f} μs")

# Check if the difference correlates with orbital phase
print("\n" + "="*80)
print("Correlation with orbital phase:")
print("="*80)

# Orbital phase from PINT
phase = (bo.orbits()[:100] % 1)
print(f"  Phase range: [{phase.min():.3f}, {phase.max():.3f}]")

# Simple correlation
from scipy.stats import pearsonr
corr, pval = pearsonr(phase, diff)
print(f"  Correlation with phase: r={corr:.3f}, p={pval:.3e}")
