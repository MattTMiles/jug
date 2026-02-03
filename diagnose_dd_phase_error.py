"""Check if DD binary delay has orbital-phase-dependent error."""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.delays.binary_dd import dd_binary_delay
from jug.utils.constants import SECS_PER_DAY

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

print("="*80)
print("DD BINARY DELAY PHASE-DEPENDENT ERROR ANALYSIS")
print("="*80)

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')
binary_comp = model.components.get('BinaryDD', None)

# Get PINT's barycentric time
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

# Update binary object and get PINT binary delays
binary_comp.update_binary_object(toas, None)
pint_binary_delay = binary_comp.binarymodel_delay(toas, None).to_value(u.s)

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

# Compute JUG binary delay at PINT's barycentric time (to isolate the DD function difference)
print("\nComputing JUG binary delays at PINT's barycentric time...")
jug_binary_delay = np.array([
    float(dd_binary_delay(t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0))
    for t in bary_mjd_pint
])

# Difference
delay_diff = (jug_binary_delay - pint_binary_delay) * 1e6  # in μs

print(f"\nBinary delay difference (JUG - PINT) at same time:")
print(f"  Mean: {np.mean(delay_diff):.6f} μs")
print(f"  RMS:  {np.std(delay_diff):.6f} μs")
print(f"  Range: [{np.min(delay_diff):.6f}, {np.max(delay_diff):.6f}] μs")

# Orbital phase
orbital_phase = ((bary_mjd_pint - t0) / pb) % 1.0

# Phase-binned analysis
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
phase_bin_idx = np.digitize(orbital_phase, bins) - 1

print(f"\nBinary delay diff binned by orbital phase:")
print(f"  Phase     | Mean diff (μs) | Count")
print(f"  ----------|----------------|------")
for i in range(n_bins):
    mask = phase_bin_idx == i
    if np.sum(mask) > 0:
        mean_diff = np.mean(delay_diff[mask])
        count = np.sum(mask)
        bar = "*" * int(abs(mean_diff) * 100) if abs(mean_diff) > 0.001 else ""
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}   |  {mean_diff:+.6f}      | {count:4d}  {bar}")

# Check if error correlates with orbital elements
print(f"\n[CORRELATIONS]")
print(f"  Correlation with orbital phase: {np.corrcoef(orbital_phase, delay_diff)[0,1]:.6f}")

# Check correlation with sin(2π*phase) and cos(2π*phase)
sin_phase = np.sin(2 * np.pi * orbital_phase)
cos_phase = np.cos(2 * np.pi * orbital_phase)
print(f"  Correlation with sin(2π*phase): {np.corrcoef(sin_phase, delay_diff)[0,1]:.6f}")
print(f"  Correlation with cos(2π*phase): {np.corrcoef(cos_phase, delay_diff)[0,1]:.6f}")

# Get eccentric anomaly from PINT for analysis
E_pint = np.array([float(bo.E()[i].to(u.rad).value) for i, bo in [(j, binary_comp.binary_instance) for j in range(len(toas))]])
# Actually, E() returns an array, not per-index. Let me fix that.
bo = binary_comp.binary_instance
E_pint = np.array([float(e.to(u.rad).value) if hasattr(e, 'to') else float(e) for e in bo.E()])

sin_E = np.sin(E_pint)
cos_E = np.cos(E_pint)
print(f"  Correlation with sin(E): {np.corrcoef(sin_E, delay_diff)[0,1]:.6f}")
print(f"  Correlation with cos(E): {np.corrcoef(cos_E, delay_diff)[0,1]:.6f}")

# The binary delay is dominated by a1 * sin(omega + E) for small e
# So the derivative is ~ a1 * cos(omega + E)
# If there's a small timing error, it would show up as correlated with cos(omega + E)
omega_pint = np.array([float(o.to(u.rad).value) if hasattr(o, 'to') else float(o) for o in bo.omega()])
sin_omega_plus_E = np.sin(omega_pint + E_pint)
cos_omega_plus_E = np.cos(omega_pint + E_pint)
print(f"  Correlation with sin(ω+E): {np.corrcoef(sin_omega_plus_E, delay_diff)[0,1]:.6f}")
print(f"  Correlation with cos(ω+E): {np.corrcoef(cos_omega_plus_E, delay_diff)[0,1]:.6f}")

print(f"\n[CONCLUSION]")
if np.std(delay_diff) < 0.01:  # < 10 ns
    print(f"✓ DD binary delay matches PINT to < 10 ns RMS")
    print(f"  The orbital-phase-dependent residual error is NOT from the DD function itself.")
else:
    print(f"⚠️ DD binary delay differs by {np.std(delay_diff):.3f} μs RMS")
