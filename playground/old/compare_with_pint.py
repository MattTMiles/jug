#!/usr/bin/env python3
"""
Compare JUG's calculations with PINT to identify discrepancies.
This is ONLY for debugging - JUG does not depend on PINT.
"""

import numpy as np
import pint
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Load the same files JUG uses
par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

print("="*80)
print("PINT vs JUG Comparison")
print("="*80)

# Load model and TOAs with PINT
print("\nLoading with PINT...")
model = get_model(par_file)
toas = get_TOAs(tim_file, model=model)

print(f"✓ Loaded {len(toas)} TOAs")
print(f"✓ Binary model: {model.BINARY.value}")
print(f"✓ F0: {model.F0.value} Hz")
print(f"✓ PEPOCH: {model.PEPOCH.value}")

# Compute residuals with PINT
residuals = Residuals(toas, model)

print(f"\nPINT Residuals:")
print(f"  RMS: {residuals.rms_weighted() * 1e6:.3f} μs")
print(f"  Mean: {np.mean(residuals.time_resids.to_value('us')):.6f} μs")
print(f"  First 10: {residuals.time_resids.to_value('us')[:10]}")

# Get barycentric TOAs from PINT
print(f"\nPINT Barycentric TOAs:")
toas_table = toas.table
print(f"  Columns: {toas_table.colnames[:10]}")

# Check if PINT has 'tdbld' column (barycentric time)
if 'tdbld' in toas_table.colnames:
    pint_bary_mjd = toas_table['tdbld']
    print(f"  Found 'tdbld' (barycentric TDB)")
    print(f"  First value: {pint_bary_mjd[0]}")

# Load JUG's tempo2 BAT for comparison
print(f"\nLoading tempo2 components...")
t2_bat = []
with open('temp_pre_components_next.out') as f:
    for line in f:
        if line.startswith('[') or line.startswith('Starting') or 'This' in line or 'Looking' in line or 'under' in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                t2_bat.append(float(parts[1]))
            except:
                pass

t2_bat = np.array(t2_bat)
print(f"✓ Loaded {len(t2_bat)} tempo2 BAT values")

# Compare PINT's barycentric times with tempo2's BAT
if 'tdbld' in toas_table.colnames:
    print(f"\nComparing PINT vs Tempo2 BAT:")
    diff = (pint_bary_mjd - t2_bat) * 86400  # Convert to seconds
    print(f"  Mean difference: {np.mean(diff):.9f} s")
    print(f"  RMS difference: {np.std(diff):.9f} s")
    print(f"  Max difference: {np.max(np.abs(diff)):.9f} s")

    if np.max(np.abs(diff)) < 1e-6:
        print(f"  ✓ PINT's tdbld matches tempo2's BAT perfectly!")

# Now check if PINT can give us delays
print(f"\n" + "="*80)
print("CHECKING DELAY COMPONENTS")
print("="*80)

# Try to get delay components from PINT
try:
    # PINT stores delays in the TOAs object
    print("\nPINT delay components:")

    # Get binary delays if available
    if hasattr(toas, 'get_flag_value'):
        print("  Checking for binary delay info...")

    # Alternative: compute delays directly
    print("\nComputing delays from PINT model...")

    # Binary delay
    if model.BINARY.value:
        print(f"  Binary model: {model.BINARY.value}")
        print(f"  PB: {model.PB.value} days")
        print(f"  A1: {model.A1.value} lt-s")
        print(f"  TASC: {model.TASC.value}")

        # Get PINT's binary delay calculation
        # This requires accessing PINT's internal delay calculation
        from pint.models.binary_dd import BinaryDD
        from pint.models.binary_ell1 import BinaryELL1

        # Get the binary model component
        if hasattr(model, 'binary_instance'):
            binary = model.binary_instance
            print(f"  Binary instance: {binary}")

except Exception as e:
    print(f"  Error accessing PINT delays: {e}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. If PINT's tdbld matches tempo2's BAT:")
print("   → Tempo2's BAT is the correct barycentric time")
print("   → JUG should use this as starting point")
print("\n2. Need to determine if BAT includes binary delays or not")
print("   → Compare JUG's BAT-binary-DM with PINT's infinite-freq time")
print("\n3. Compare residuals:")
print(f"   → PINT RMS: {residuals.rms_weighted() * 1e6:.3f} μs")
print("   → JUG RMS: ~850 μs (currently)")
print("   → Tempo2 RMS: 0.817 μs")
print("\nIf PINT RMS matches Tempo2 → PINT is correct reference")
