"""Compare JUG and PINT DD delays using EXACT same input time."""

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

from jug.delays.binary_dd import dd_binary_delay
from pint.models import get_model
from pint.toa import get_TOAs

PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("DD DELAY COMPARISON AT EXACT SAME TIME")
print("="*80)

# Load PINT
model = get_model(PAR_FILE)
toas = get_TOAs(TIM_FILE, planets=True)

binary_comp = model.get_components_by_category()['pulsar_system'][0]
binary_comp.update_binary_object(toas, None)

# Get parameters
PB = float(binary_comp.PB.value)
A1 = float(binary_comp.A1.value)
ECC = float(binary_comp.ECC.value)
OM = float(binary_comp.OM.value)
T0 = float(binary_comp.T0.value)
GAMMA = float(binary_comp.GAMMA.value) if binary_comp.GAMMA.value is not None else 0.0
PBDOT = float(binary_comp.PBDOT.value)
OMDOT = float(binary_comp.OMDOT.value)
M2 = float(binary_comp.M2.value)
SINI = float(binary_comp.SINI.value)

# Test first TOA
idx = 0

# Get the EXACT time PINT uses
t_bary_pint = float(binary_comp.barycentric_time[idx].to_value('day'))

print(f"\nUsing PINT's barycentric time: {t_bary_pint:.15f} MJD")

# Compute PINT delay
pint_delay = float(binary_comp.binary_instance.DDdelay()[idx].to_value('s'))
print(f"PINT DD delay: {pint_delay:.15f} s")

# Compute JUG delay at EXACT same time
jug_delay = float(dd_binary_delay(
    t_bary_pint, PB, A1, ECC, OM, T0, GAMMA, PBDOT,
    OMDOT, 0.0, 0.0, SINI, M2
))
print(f"JUG DD delay:  {jug_delay:.15f} s")

# Compare
diff_s = jug_delay - pint_delay
diff_us = diff_s * 1e6
diff_ns = diff_s * 1e9

print(f"\nDifference (JUG - PINT):")
print(f"  {diff_s:.15e} s")
print(f"  {diff_us:.9f} μs")
print(f"  {diff_ns:.3f} ns")

print(f"\n" + "="*80)
if abs(diff_ns) < 50:
    print(f"✅ SUCCESS! Difference ({abs(diff_ns):.1f} ns) < 50 ns target")
else:
    print(f"❌ FAILED! Difference ({abs(diff_ns):.1f} ns) > 50 ns target")
    print(f"   Factor over target: {abs(diff_ns) / 50:.1f}×")

# Test on multiple TOAs
print(f"\n" + "="*80)
print("TESTING MULTIPLE TOAs")
print("="*80)

n_test = min(100, len(toas))
diffs_ns = []

for i in range(n_test):
    t_bary = float(binary_comp.barycentric_time[i].to_value('day'))
    pint_d = float(binary_comp.binary_instance.DDdelay()[i].to_value('s'))
    jug_d = float(dd_binary_delay(
        t_bary, PB, A1, ECC, OM, T0, GAMMA, PBDOT,
        OMDOT, 0.0, 0.0, SINI, M2
    ))
    diffs_ns.append((jug_d - pint_d) * 1e9)

diffs_ns = np.array(diffs_ns)

print(f"\nTested {n_test} TOAs:")
print(f"  Mean difference: {np.mean(diffs_ns):.3f} ns")
print(f"  RMS difference: {np.std(diffs_ns):.3f} ns")
print(f"  Max difference: {np.max(np.abs(diffs_ns)):.3f} ns")

if np.std(diffs_ns) < 50:
    print(f"\n✅ SUCCESS! RMS ({np.std(diffs_ns):.1f} ns) < 50 ns")
else:
    print(f"\n❌ FAILED! RMS ({np.std(diffs_ns):.1f} ns) > 50 ns")
