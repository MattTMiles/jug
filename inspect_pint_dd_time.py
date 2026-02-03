#!/usr/bin/env python
"""
Diagnostic script to identify the exact time variable PINT's BinaryDD uses internally.
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

# Get the BinaryDD component
print("\n" + "="*80)
print("Getting BinaryDD component...")
print("="*80)

binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

print(f"Binary instance type: {type(bo)}")
print(f"Binary instance class name: {bo.__class__.__name__}")

# Print TOA table columns
print("\n" + "="*80)
print("TOA table column names:")
print("="*80)
print(toas.table.colnames)

# Inspect specific TOA indices
for i in [0, 100]:
    print(f"\n" + "="*80)
    print(f"TOA index i={i}:")
    print("="*80)
    
    print(f"\n--- TOA table columns ---")
    if 'tdbld' in toas.table.colnames:
        print(f"  toas.table['tdbld'][{i}] = {toas.table['tdbld'][i]}")
    if 'mjd' in toas.table.colnames:
        print(f"  toas.table['mjd'][{i}] = {toas.table['mjd'][i]}")
    if 'mjd_float' in toas.table.colnames:
        print(f"  toas.table['mjd_float'][{i}] = {toas.table['mjd_float'][i]}")
    
    # Check for any other time-like columns
    for col in toas.table.colnames:
        if any(x in col.lower() for x in ['time', 'tdb', 'toa', 'bary']):
            if col not in ['tdbld']:
                try:
                    val = toas.table[col][i]
                    print(f"  toas.table['{col}'][{i}] = {val}")
                except:
                    pass

# Inspect binary object attributes
print("\n" + "="*80)
print("Binary object (bo) attributes containing time-related keywords:")
print("="*80)

time_keywords = ['t', 'mjd', 'tt0', 'time', 'pb', 'phase', 'T0', 'epoch', 'bary']
all_attrs = dir(bo)

# Find matching attributes
matching_attrs = []
for attr in all_attrs:
    if attr.startswith('_'):
        continue
    attr_lower = attr.lower()
    if any(kw in attr_lower for kw in time_keywords):
        matching_attrs.append(attr)

print(f"\nMatching attributes: {matching_attrs}")

print("\n--- Inspecting each matching attribute ---")
for attr in sorted(matching_attrs):
    try:
        val = getattr(bo, attr)
        if callable(val):
            # Try calling it
            try:
                result = val()
                if hasattr(result, 'shape'):
                    print(f"\n  bo.{attr}() [callable]:")
                    print(f"    type: {type(result)}")
                    print(f"    shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
                    if hasattr(result, 'unit'):
                        print(f"    unit: {result.unit}")
                    print(f"    value[0]: {result[0] if hasattr(result, '__getitem__') else result}")
                    print(f"    value[100]: {result[100] if hasattr(result, '__getitem__') and len(result) > 100 else 'N/A'}")
                elif hasattr(result, 'value'):
                    print(f"\n  bo.{attr}() [callable]: {result}")
                else:
                    print(f"\n  bo.{attr}() [callable]: {result}")
            except Exception as e:
                print(f"\n  bo.{attr}() [callable]: ERROR calling - {e}")
        else:
            # It's a property or attribute
            if hasattr(val, 'shape'):
                print(f"\n  bo.{attr} [attribute]:")
                print(f"    type: {type(val)}")
                print(f"    shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
                if hasattr(val, 'unit'):
                    print(f"    unit: {val.unit}")
                print(f"    value[0]: {val[0] if hasattr(val, '__getitem__') else val}")
                print(f"    value[100]: {val[100] if hasattr(val, '__getitem__') and len(val) > 100 else 'N/A'}")
            elif hasattr(val, 'value'):
                print(f"\n  bo.{attr} [attribute]: {val}")
            elif val is not None:
                print(f"\n  bo.{attr} [attribute]: {val} (type: {type(val).__name__})")
    except Exception as e:
        print(f"\n  bo.{attr}: ERROR - {e}")

# Specifically check for the 't' attribute which is often the key time
print("\n" + "="*80)
print("Specifically checking bo.t (the main time input):")
print("="*80)
if hasattr(bo, 't'):
    t = bo.t
    print(f"  type: {type(t)}")
    if hasattr(t, 'shape'):
        print(f"  shape: {t.shape}")
    if hasattr(t, 'unit'):
        print(f"  unit: {t.unit}")
    if hasattr(t, '__getitem__'):
        print(f"  t[0] = {t[0]}")
        print(f"  t[100] = {t[100]}")
        # Convert to float MJD
        if hasattr(t[0], 'mjd'):
            print(f"  t[0].mjd = {t[0].mjd}")
            print(f"  t[100].mjd = {t[100].mjd}")
        elif hasattr(t[0], 'value'):
            print(f"  t[0].value = {t[0].value}")
            print(f"  t[100].value = {t[100].value}")

# Check PINT binary intermediates
print("\n" + "="*80)
print("PINT DD binary intermediates:")
print("="*80)

for method_name in ['E', 'omega', 'nu', 'orbits', 'M', 'tt0', 'pb', 'pbdot']:
    if hasattr(bo, method_name):
        method = getattr(bo, method_name)
        if callable(method):
            try:
                result = method()
                if hasattr(result, '__getitem__') and len(result) > 100:
                    print(f"\n  bo.{method_name}():")
                    if hasattr(result, 'unit'):
                        print(f"    unit: {result.unit}")
                    print(f"    [0] = {result[0]}")
                    print(f"    [100] = {result[100]}")
                else:
                    print(f"\n  bo.{method_name}(): {result}")
            except Exception as e:
                print(f"\n  bo.{method_name}(): ERROR - {e}")
        else:
            print(f"\n  bo.{method_name} (not callable): {method}")

# Compare with model.get_barycentric_toas()
print("\n" + "="*80)
print("Comparing bo.t with model.get_barycentric_toas(toas):")
print("="*80)

bary_toas = model.get_barycentric_toas(toas)
print(f"  bary_toas type: {type(bary_toas)}")
if hasattr(bary_toas, 'shape'):
    print(f"  bary_toas shape: {bary_toas.shape}")
if hasattr(bary_toas, 'unit'):
    print(f"  bary_toas unit: {bary_toas.unit}")

# Get float MJD values
if hasattr(bary_toas, 'mjd'):
    bary_mjd_0 = bary_toas[0].mjd
    bary_mjd_100 = bary_toas[100].mjd
elif hasattr(bary_toas[0], 'value'):
    bary_mjd_0 = bary_toas[0].value
    bary_mjd_100 = bary_toas[100].value
else:
    bary_mjd_0 = float(bary_toas[0])
    bary_mjd_100 = float(bary_toas[100])

print(f"  bary_toas[0] = {bary_toas[0]} -> MJD = {bary_mjd_0}")
print(f"  bary_toas[100] = {bary_toas[100]} -> MJD = {bary_mjd_100}")

# Get bo.t values
if hasattr(bo, 't'):
    t = bo.t
    if hasattr(t[0], 'mjd'):
        bo_t_mjd_0 = t[0].mjd
        bo_t_mjd_100 = t[100].mjd
    elif hasattr(t[0], 'value'):
        bo_t_mjd_0 = t[0].value
        bo_t_mjd_100 = t[100].value
    else:
        bo_t_mjd_0 = float(t[0])
        bo_t_mjd_100 = float(t[100])
    
    print(f"\n  bo.t[0] = {t[0]} -> MJD = {bo_t_mjd_0}")
    print(f"  bo.t[100] = {t[100]} -> MJD = {bo_t_mjd_100}")
    
    diff_0 = (bo_t_mjd_0 - bary_mjd_0) * 86400  # seconds
    diff_100 = (bo_t_mjd_100 - bary_mjd_100) * 86400
    print(f"\n  Difference (bo.t - bary_toas) at i=0: {diff_0:.9f} seconds")
    print(f"  Difference (bo.t - bary_toas) at i=100: {diff_100:.9f} seconds")

# Check what toas column bo.t comes from
print("\n" + "="*80)
print("Tracing the source of bo.t:")
print("="*80)

# Check toas.table['tdbld']
tdbld_0 = float(toas.table['tdbld'][0])
tdbld_100 = float(toas.table['tdbld'][100])
print(f"  toas.table['tdbld'][0] = {tdbld_0}")
print(f"  toas.table['tdbld'][100] = {tdbld_100}")

if hasattr(bo, 't'):
    diff_tdbld_0 = (bo_t_mjd_0 - tdbld_0) * 86400
    diff_tdbld_100 = (bo_t_mjd_100 - tdbld_100) * 86400
    print(f"\n  Difference (bo.t - tdbld) at i=0: {diff_tdbld_0:.9f} seconds")
    print(f"  Difference (bo.t - tdbld) at i=100: {diff_tdbld_100:.9f} seconds")

# Final summary
print("\n" + "="*80)
print("SUMMARY - PINT DD time variable identification:")
print("="*80)
print("""
Based on the above inspection, we can identify:
1. What time variable bo.t contains
2. Whether it matches get_barycentric_toas() or tdbld or something else
3. What delays (if any) have been subtracted before the binary model sees the time
""")

# Also check what delays are being computed
print("\n" + "="*80)
print("Checking delay components in PINT model:")
print("="*80)

print("\nModel delay funcs (in order):")
for i, func in enumerate(model.delay_funcs):
    print(f"  {i}: {func.__name__}")

# Look at what BinaryDD actually computes
print("\n" + "="*80)
print("BinaryDD binarydelay method signature and source (if available):")
print("="*80)

import inspect
try:
    sig = inspect.signature(binary_comp.binarymodel_delay)
    print(f"  binarymodel_delay signature: {sig}")
except:
    pass

# Check the actual delay value
print("\n" + "="*80)
print("Computing actual binary delay from PINT:")
print("="*80)

binary_delay = binary_comp.binarymodel_delay(toas, None)
print(f"  binary_delay type: {type(binary_delay)}")
print(f"  binary_delay unit: {binary_delay.unit}")
print(f"  binary_delay[0] = {binary_delay[0]}")
print(f"  binary_delay[100] = {binary_delay[100]}")

# Also get the delay from the binary object directly
if hasattr(bo, 'delay'):
    bo_delay = bo.delay()
    print(f"\n  bo.delay()[0] = {bo_delay[0]}")
    print(f"  bo.delay()[100] = {bo_delay[100]}")
