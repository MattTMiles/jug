#!/usr/bin/env python
"""
Check what binarymodel_delay returns vs the sum of DD components.
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs

# Dataset paths  
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("Loading PINT model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

print("\n" + "="*80)
print("PINT DD delay breakdown for TOA[0]:")
print("="*80)

# Get all delay components from the binary instance
print(f"\nFrom binary instance (bo):")
print(f"  delayR (Roemer):       {bo.delayR()[0].to('s').value:.15f} s")
print(f"  delayA (aberration):   {bo.delayA()[0].to('s').value:.15f} s")
print(f"  delayE (Einstein):     {bo.delayE()[0].to('s').value:.15f} s")
print(f"  delayS (Shapiro):      {bo.delayS()[0].to('s').value:.15f} s")

# Check for delayI (inverse)
if hasattr(bo, 'delayI'):
    print(f"  delayI (inverse):      {bo.delayI()[0].to('s').value:.15f} s")
if hasattr(bo, 'delayInverse'):
    print(f"  delayInverse:          {bo.delayInverse()[0].to('s').value:.15f} s")

# DDdelay method
dd_delay = bo.DDdelay()
print(f"\n  bo.DDdelay()[0]:       {dd_delay[0].to('s').value:.15f} s")

# Sum components
sum_delay = bo.delayR()[0] + bo.delayA()[0] + bo.delayE()[0] + bo.delayS()[0]
print(f"\n  Sum (R+A+E+S):         {sum_delay.to('s').value:.15f} s")

# binarymodel_delay
bm_delay = binary_comp.binarymodel_delay(toas, None)
print(f"\n  binarymodel_delay[0]:  {bm_delay[0].to('s').value:.15f} s")

# Difference
diff1 = (dd_delay[0] - sum_delay).to('us').value
diff2 = (bm_delay[0] - dd_delay[0]).to('us').value
diff3 = (bm_delay[0] - sum_delay).to('us').value

print(f"\nDifferences:")
print(f"  DDdelay - Sum(R+A+E+S): {diff1:.6f} μs")
print(f"  binarymodel_delay - DDdelay: {diff2:.6f} μs")
print(f"  binarymodel_delay - Sum: {diff3:.6f} μs")

# Check if there's an inverse delay
print("\n" + "="*80)
print("Checking for inverse delay in PINT DD:")
print("="*80)

# Look at all methods that return delays
for attr in dir(bo):
    if 'delay' in attr.lower() and not attr.startswith('_'):
        try:
            method = getattr(bo, attr)
            if callable(method):
                result = method()
                if hasattr(result, '__getitem__') and hasattr(result[0], 'to'):
                    val = result[0].to('s').value
                    if abs(val) > 1e-15:
                        print(f"  bo.{attr}()[0] = {val:.15f} s")
        except Exception as e:
            pass

# Check what DDdelay actually computes
print("\n" + "="*80)
print("Inspecting DDdelay source:")
print("="*80)

import inspect
try:
    src = inspect.getsource(bo.DDdelay)
    print(src[:2000])
except Exception as e:
    print(f"Could not get source: {e}")

# Also check binarymodel_delay
print("\n" + "="*80)
print("Inspecting binarymodel_delay source:")
print("="*80)

try:
    src = inspect.getsource(binary_comp.binarymodel_delay)
    print(src[:2000])
except Exception as e:
    print(f"Could not get source: {e}")
