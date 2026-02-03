"""Debug DD binary delay calculation in detail."""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.delays.binary_dd import dd_binary_delay

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

print("="*80)
print("DETAILED DD BINARY DELAY DEBUG")
print("="*80)

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

# Get binary component
binary_comp = model.components.get('BinaryDD', None)
print(f"\nPINT Binary model: {type(binary_comp).__name__}")

# Get PINT's barycentric time
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

# Update binary object
binary_comp.update_binary_object(toas, None)

# Get internal orbital parameters from PINT's binary object
bo = binary_comp.binary_instance
print(f"\nPINT Binary Object internal parameters:")
print(f"  PB = {bo.PB.value if hasattr(bo.PB, 'value') else bo.PB} days")
print(f"  A1 = {bo.A1.value if hasattr(bo.A1, 'value') else bo.A1} lt-s")
print(f"  T0 = {bo.T0.value if hasattr(bo.T0, 'value') else bo.T0} MJD")
print(f"  ECC = {bo.ECC.value if hasattr(bo.ECC, 'value') else bo.ECC}")
print(f"  OM = {bo.OM.value if hasattr(bo.OM, 'value') else bo.OM} deg")
print(f"  GAMMA = {bo.GAMMA.value if hasattr(bo.GAMMA, 'value') else bo.GAMMA} s")
print(f"  PBDOT = {bo.PBDOT.value if hasattr(bo.PBDOT, 'value') else bo.PBDOT}")
print(f"  OMDOT = {bo.OMDOT.value if hasattr(bo.OMDOT, 'value') else bo.OMDOT} deg/yr")
print(f"  M2 = {bo.M2.value if hasattr(bo.M2, 'value') else bo.M2} M_sun")

# Check SINI handling
sini_param = model.SINI
print(f"  SINI = {sini_param.value if hasattr(sini_param, 'value') else sini_param}")

# Get the delay components from PINT
# For a single TOA
idx = 500
t_bary = bary_mjd_pint[idx]

# Get PINT's delays
pint_total = binary_comp.binarymodel_delay(toas, None).to_value(u.s)[idx]

# Try to get individual components from PINT
try:
    pint_roemer = bo.delayR()[idx] if hasattr(bo.delayR(), '__len__') else bo.delayR()
    pint_einstein = bo.delayE()[idx] if hasattr(bo.delayE(), '__len__') else bo.delayE()
    pint_shapiro = bo.delayS()[idx] if hasattr(bo.delayS(), '__len__') else bo.delayS()
    pint_inverse = bo.delayInverse()[idx] if hasattr(bo.delayInverse(), '__len__') else bo.delayInverse()

    print(f"\nPINT delay components at TOA {idx}:")
    print(f"  Roemer (Dre): {pint_roemer*1e6:.6f} μs")
    print(f"  Einstein:     {pint_einstein*1e6:.6f} μs")
    print(f"  Shapiro:      {pint_shapiro*1e6:.6f} μs")
    print(f"  Inverse:      {pint_inverse*1e6:.6f} μs")
    print(f"  Total:        {pint_total*1e6:.6f} μs")
except Exception as e:
    print(f"  Error getting PINT components: {e}")

# JUG parameters
params = parse_par_file(par_file)
pb = float(params.get('PB', 0.0))
a1 = float(params.get('A1', 0.0))
ecc = float(params.get('ECC', 0.0))
om = float(params.get('OM', 0.0))
t0 = float(params.get('T0', 0.0))
gamma = float(params.get('GAMMA', 0.0))
pbdot = float(params.get('PBDOT', 0.0))
omdot = float(params.get('OMDOT', 0.0))
xdot = float(params.get('XDOT', 0.0))
edot = float(params.get('EDOT', 0.0))
sini = float(params.get('SINI', 0.0)) if not isinstance(params.get('SINI', 0.0), str) else 0.0
m2 = float(params.get('M2', 0.0))

print(f"\nJUG parameters:")
print(f"  PB = {pb} days")
print(f"  A1 = {a1} lt-s")
print(f"  T0 = {t0} MJD")
print(f"  ECC = {ecc}")
print(f"  OM = {om} deg")
print(f"  GAMMA = {gamma} s")
print(f"  PBDOT = {pbdot}")
print(f"  OMDOT = {omdot} deg/yr")
print(f"  XDOT = {xdot}")
print(f"  EDOT = {edot}")
print(f"  M2 = {m2} M_sun")
print(f"  SINI = {sini}")

# Compute JUG delay
jug_total = float(dd_binary_delay(
    t_bary, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
    sini, m2, 0.0, 0.0, 0.0
))

print(f"\nComparison at TOA {idx} (t_bary = {t_bary:.9f} MJD):")
print(f"  PINT delay: {pint_total*1e6:.6f} μs")
print(f"  JUG delay:  {jug_total*1e6:.6f} μs")
print(f"  Difference: {(jug_total - pint_total)*1e6:.6f} μs")

# Check unit conversion for OMDOT
print(f"\n" + "="*80)
print("UNIT CHECK FOR OMDOT")
print("="*80)
print(f"  PINT OMDOT value: {bo.OMDOT.value if hasattr(bo.OMDOT, 'value') else bo.OMDOT}")
print(f"  PINT OMDOT unit: {bo.OMDOT.unit if hasattr(bo.OMDOT, 'unit') else 'unknown'}")
print(f"  JUG OMDOT: {omdot} (assumed deg/yr)")

# Check conversion in JUG DD code
# OMDOT should be converted from deg/yr to rad/day for omega calculation
omdot_rad_day_jug = omdot * (np.pi / 180.0) / 365.25
omdot_rad_day_pint = (bo.OMDOT.to(u.rad / u.day).value if hasattr(bo.OMDOT, 'to') else
                      bo.OMDOT.value * np.pi / 180.0 / 365.25)
print(f"  JUG OMDOT in rad/day: {omdot_rad_day_jug}")
print(f"  PINT OMDOT in rad/day: {omdot_rad_day_pint}")

# Check T0
print(f"\n" + "="*80)
print("T0 CHECK")
print("="*80)
print(f"  PINT T0: {bo.T0.value if hasattr(bo.T0, 'value') else bo.T0}")
print(f"  JUG T0:  {t0}")
print(f"  Diff:    {(t0 - (bo.T0.value if hasattr(bo.T0, 'value') else bo.T0))*86400:.6f} seconds")

# Check for any timing in computing orbital phase
dt = t_bary - t0
pb_sec = pb * 86400
n0 = 2 * np.pi / pb_sec
print(f"\n" + "="*80)
print("ORBITAL PHASE CHECK")
print("="*80)
print(f"  Time since T0: {dt:.6f} days = {dt*86400:.6f} seconds")
print(f"  Orbital period: {pb} days = {pb_sec:.6f} seconds")
print(f"  n0 (mean motion): {n0:.9e} rad/s")
print(f"  Orbital cycles since T0: {dt / pb:.6f}")
print(f"  Orbital phase: {(dt / pb) % 1.0:.6f}")
