#!/usr/bin/env python
"""Compare JUG and PINT DD binary delays directly."""

import numpy as np

# Load PINT
import pint
print(f"PINT version: {pint.__version__}")
from pint.models import get_model
from pint import toa as pint_toas
import pint.residuals

# JUG imports
from jug.delays.binary_dd import dd_binary_delay
from jug.residuals.simple_calculator import compute_residuals_simple

par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.tim"

print("=" * 70)
print("Loading PINT model and TOAs...")
print("=" * 70)

# Load model
model = get_model(par_file)
print(f"Binary model: {model.BINARY.value}")

# Load TOAs
toas = pint_toas.get_TOAs(tim_file, model=model)
print(f"Loaded {toas.ntoas} TOAs")

# Get pre-fit residuals
residuals = pint.residuals.Residuals(toas, model)
pint_resids_us = residuals.time_resids.to('us').value

print(f"\nPINT pre-fit residuals:")
print(f"  Mean: {np.mean(pint_resids_us):.3f} μs")
print(f"  Std: {np.std(pint_resids_us):.3f} μs")
print(f"  Weighted RMS: {residuals.rms_weighted().to('us').value:.3f} μs")

# Now compute JUG residuals
print("\n" + "=" * 70)
print("Computing JUG residuals...")
print("=" * 70)

jug_result = compute_residuals_simple(par_file, tim_file, verbose=True)

print("\n" + "=" * 70)
print("Comparison Summary")
print("=" * 70)
print(f"\nPINT weighted RMS: {residuals.rms_weighted().to('us').value:.3f} μs")
print(f"JUG weighted RMS:  {jug_result['weighted_rms_us']:.3f} μs")

# Compare individual residuals
print("\n" + "-" * 70)
print("First 10 residuals comparison:")
print("-" * 70)
print(f"{'TOA':>6} {'PINT (μs)':>12} {'JUG (μs)':>12} {'Diff (μs)':>12}")
for i in range(min(10, len(pint_resids_us))):
    diff = jug_result['residuals_us'][i] - pint_resids_us[i]
    print(f"{i:>6} {pint_resids_us[i]:>12.3f} {jug_result['residuals_us'][i]:>12.3f} {diff:>12.3f}")

# Statistics on differences
diff_all = jug_result['residuals_us'] - pint_resids_us
print(f"\nResidual differences (JUG - PINT):")
print(f"  Mean: {np.mean(diff_all):.3f} μs")
print(f"  Std: {np.std(diff_all):.3f} μs")
print(f"  Max: {np.max(np.abs(diff_all)):.3f} μs")

# Now let's look at what time PINT uses for binary delay
print("\n" + "=" * 70)
print("Investigating PINT binary delay computation...")
print("=" * 70)

# Get the TDB times
tdb_times = toas.get_mjds()  # This gives barycentric TDB times
print(f"\nFirst 3 TDB times: {tdb_times[:3]}")

# Get the binary model component
binary_model = model.components['BinaryDD']
print(f"Binary model type: {type(binary_model)}")

# PINT computes binary delay at the pulsar binary time
# Let's see what method it uses
try:
    # Get the binary delay for all TOAs
    from astropy import units as u

    # The binary delay in PINT
    binary_delay = model.binarymodel_delay(toas, None)
    print(f"\nPINT binary delays (first 5):")
    for i in range(5):
        print(f"  TOA {i}: {binary_delay[i].to('us').value:.3f} μs (MJD {float(tdb_times[i]):.6f})")

    # Now compute JUG binary delay at the same times
    # But what time should we use?
    print("\nComparing binary delays at TDB times...")

    params = {
        'PB': float(model.PB.value),
        'T0': float(model.T0.value),
        'A1': float(model.A1.value),
        'OM': float(model.OM.value),
        'ECC': float(model.ECC.value),
        'SINI': float(model.SINI.value),
        'M2': float(model.M2.value),
        'OMDOT': float(model.OMDOT.value) if hasattr(model, 'OMDOT') else 0.0,
        'PBDOT': float(model.PBDOT.value) if hasattr(model, 'PBDOT') else 0.0,
        'XDOT': float(model.XDOT.value) if hasattr(model, 'XDOT') else 0.0,
        'GAMMA': float(model.GAMMA.value) if hasattr(model, 'GAMMA') else 0.0,
        'EDOT': float(model.EDOT.value) if hasattr(model, 'EDOT') else 0.0,
    }

    print(f"\nBinary parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Compute JUG delays at same TDB times
    print("\n" + "-" * 70)
    print("Binary delay comparison (JUG at TDB time vs PINT):")
    print("-" * 70)
    print(f"{'TOA':>6} {'MJD':>14} {'PINT (μs)':>14} {'JUG (μs)':>14} {'Diff (μs)':>12}")

    for i in range(min(10, len(tdb_times))):
        t = float(tdb_times[i])
        jug_delay = dd_binary_delay(
            t_bary_mjd=t,
            pb_days=params['PB'],
            a1_lt_sec=params['A1'],
            ecc=params['ECC'],
            omega_deg=params['OM'],
            t0_mjd=params['T0'],
            gamma_sec=params['GAMMA'],
            pbdot=params['PBDOT'],
            omdot_deg_yr=params['OMDOT'],
            xdot=params['XDOT'],
            edot=params['EDOT'],
            sini=params['SINI'],
            m2_msun=params['M2'],
        )
        pint_delay_us = binary_delay[i].to('us').value
        jug_delay_us = float(jug_delay) * 1e6
        diff = jug_delay_us - pint_delay_us
        print(f"{i:>6} {t:>14.6f} {pint_delay_us:>14.3f} {jug_delay_us:>14.3f} {diff:>12.3f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
