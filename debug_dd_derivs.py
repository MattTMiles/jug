
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
import jax
import jax.numpy as jnp
from jug.fitting.derivatives_dd import (
    _d_delay_d_H3, _d_delay_d_STIG, _d_delay_d_M2, _d_delay_d_SINI,
    solve_kepler, compute_true_anomaly, compute_mean_anomaly_dd,
    T_SUN, DEG_TO_RAD
)

# J1022+1001 parameters
PB = 7.805134794
T0 = 49778.40923
ECC = 9.673e-5
OM_DEG = 97.7065
OMDOT = 0.01
H3 = 6.962436e-07
STIG = 0.338437

# Derived quantities
SINI = 2 * STIG / (1 + STIG**2)
M2 = H3 / (STIG**3 * T_SUN)

print(f"J1022+1001 Parameters:")
print(f"PB: {PB}")
print(f"H3: {H3}")
print(f"STIG: {STIG}")
print(f"Derived SINI: {SINI}")
print(f"Derived M2: {M2}")
print("-" * 50)

# Create dummy TOAs spanning one orbit
# 100 points across 8 days
toas_mjd = np.linspace(T0, T0 + 8.0, 100)
toas_bary_mjd = jnp.array(toas_mjd)

# Compute OM_RAD (approximate, ignoring significant OMDOT for 8 days)
om_rad = jnp.full_like(toas_bary_mjd, OM_DEG * DEG_TO_RAD)
pbdot = 0.0

print("\nComputing derivatives...")

# 1. H3 Derivative
d_delay_d_h3 = _d_delay_d_H3(toas_bary_mjd, PB, T0, ECC, om_rad, pbdot, STIG)
print(f"\nd(delay)/d(H3) stats:")
print(f"Mean: {jnp.mean(d_delay_d_h3)}")
print(f"Max:  {jnp.max(jnp.abs(d_delay_d_h3))}")
print(f"Min:  {jnp.min(jnp.abs(d_delay_d_h3))}")

# Chain rule verification for H3
# d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
d_delay_d_m2_val = _d_delay_d_M2(toas_bary_mjd, PB, T0, ECC, om_rad, pbdot, SINI)
d_m2_d_h3_val = 1.0 / (STIG**3 * T_SUN)
calc_h3_deriv = d_delay_d_m2_val * d_m2_d_h3_val

print(f"\nChain rule H3 components:")
print(f"d(delay)/d(M2) typical: {jnp.mean(d_delay_d_m2_val)}")
print(f"d(M2)/d(H3) constant:   {d_m2_d_h3_val}")
print(f"Product typical:        {jnp.mean(calc_h3_deriv)}")

if jnp.allclose(d_delay_d_h3, calc_h3_deriv):
    print("✓ H3 derivative matches manual calculation")
else:
    print("❌ H3 derivative mismatch!")


# 2. STIG Derivative
d_delay_d_stig = _d_delay_d_STIG(toas_bary_mjd, PB, T0, ECC, om_rad, pbdot, H3, STIG)
print(f"\nd(delay)/d(STIG) stats:")
print(f"Mean: {jnp.mean(d_delay_d_stig)}")
print(f"Max:  {jnp.max(jnp.abs(d_delay_d_stig))}")

# Chain rule verification for STIG
# d(delay)/d(STIG) = d(delay)/d(M2) * d(M2)/d(STIG) + d(delay)/d(SINI) * d(SINI)/d(STIG)
d_delay_d_sini_val = _d_delay_d_SINI(toas_bary_mjd, PB, T0, ECC, om_rad, pbdot, SINI, M2)
d_m2_d_stig_val = -3 * M2 / STIG
d_sini_d_stig_val = 2 * (1 - STIG**2) / (1 + STIG**2)**2

term1 = d_delay_d_m2_val * d_m2_d_stig_val
term2 = d_delay_d_sini_val * d_sini_d_stig_val
calc_stig_deriv = term1 + term2

print(f"\nChain rule STIG components:")
print(f"d(delay)/d(M2) mean:   {jnp.mean(d_delay_d_m2_val)}")
print(f"d(M2)/d(STIG) const:   {d_m2_d_stig_val}")
print(f"Term 1 mean:           {jnp.mean(term1)}")
print(f"d(delay)/d(SINI) mean: {jnp.mean(d_delay_d_sini_val)}")
print(f"d(SINI)/d(STIG) const: {d_sini_d_stig_val}")
print(f"Term 2 mean:           {jnp.mean(term2)}")
print(f"Total STIG deriv mean: {jnp.mean(calc_stig_deriv)}")

if jnp.allclose(d_delay_d_stig, calc_stig_deriv):
    print("✓ STIG derivative matches manual calculation")
else:
    print("❌ STIG derivative mismatch!")

# Check for nan/inf
if jnp.any(jnp.isnan(d_delay_d_h3)) or jnp.any(jnp.isnan(d_delay_d_stig)):
    print("\n❌ NaNs detected in derivatives!")
else:
    print("\n✓ No NaNs detected")

