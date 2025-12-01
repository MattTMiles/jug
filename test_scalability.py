#!/usr/bin/env python3
"""
Scalability Test: How does JUG vs PINT scale with number of TOAs?
"""

import numpy as np
import time
import sys
from pathlib import Path
import tempfile
import os

# Test different TOA counts
TOA_COUNTS = [1000, 5000, 10000, 20000, 50000, 100000]

print("="*80)
print("SCALABILITY TEST: JUG vs PINT")
print("="*80)
print()

results = []

for n_toas in TOA_COUNTS:
    print(f"\n{'='*80}")
    print(f"Testing with {n_toas:,} TOAs")
    print(f"{'='*80}")
    
    # Create synthetic TOA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tim', delete=False) as f:
        tim_file = f.name
        f.write("FORMAT 1\n")
        
        # Base this on J1909-3744
        mjd_start = 58526.0
        mjd_span = 2311.0  # ~6 years
        
        for i in range(n_toas):
            mjd = mjd_start + (mjd_span * i / n_toas)
            freq = 900.0 + np.random.randn() * 100  # ~900 MHz ± 100
            error = 0.5 + abs(np.random.randn()) * 0.5  # ~0.5-1.5 μs
            f.write(f"meerkat {freq:.3f} {mjd:.12f} {error:.3f}\n")
    
    par_file = 'data/pulsars/J1909-3744_tdb_wrong.par'
    
    # ========================================================================
    # Test PINT
    # ========================================================================
    print("\nTesting PINT...")
    try:
        start = time.time()
        result = os.popen(f'''python3 -c "
import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
import time

model = pm.get_model('{par_file}')
toas = pt.get_TOAs('{tim_file}', model=model)

model.F0.frozen = False
model.F1.frozen = False
fit_start = time.time()
fitter = pf.WLSFitter(toas, model)
fitter.fit_toas()
fit_time = time.time() - fit_start

print(f'FITTING_TIME={{fit_time:.3f}}')
" 2>&1 | grep FITTING_TIME
''').read()
        
        pint_time = None
        for line in result.split('\n'):
            if 'FITTING_TIME=' in line:
                pint_time = float(line.split('=')[1])
        
        if pint_time:
            print(f"  ✓ PINT fitting time: {pint_time:.2f}s")
        else:
            print(f"  ✗ PINT failed")
            pint_time = None
            
    except Exception as e:
        print(f"  ✗ PINT error: {e}")
        pint_time = None
    
    # ========================================================================
    # Test JUG
    # ========================================================================
    print("\nTesting JUG...")
    try:
        start = time.time()
        result = os.popen(f'''python3 -c "
from jug.fitting import fit_parameters_optimized
from pathlib import Path

result = fit_parameters_optimized(
    par_file=Path('{par_file}'),
    tim_file=Path('{tim_file}'),
    fit_params=['F0', 'F1'],
    verbose=False
)

print(f'TOTAL_TIME={{result[\\"total_time\\"]:.3f}}')
print(f'CACHE_TIME={{result[\\"cache_time\\"]:.3f}}')
print(f'JIT_TIME={{result[\\"jit_time\\"]:.3f}}')
print(f'ITERS={{result[\\"iterations\\"]}}')
" 2>&1 | grep -E 'TOTAL_TIME|CACHE_TIME|JIT_TIME|ITERS'
''').read()
        
        jug_total = jug_cache = jug_jit = jug_iters = None
        for line in result.split('\n'):
            if 'TOTAL_TIME=' in line:
                jug_total = float(line.split('=')[1])
            elif 'CACHE_TIME=' in line:
                jug_cache = float(line.split('=')[1])
            elif 'JIT_TIME=' in line:
                jug_jit = float(line.split('=')[1])
            elif 'ITERS=' in line:
                jug_iters = int(line.split('=')[1])
        
        if jug_total:
            jug_iter = jug_total - jug_cache - jug_jit
            print(f"  ✓ JUG total time: {jug_total:.2f}s")
            print(f"    - Cache: {jug_cache:.2f}s")
            print(f"    - JIT: {jug_jit:.2f}s")
            print(f"    - Iterations: {jug_iter:.2f}s ({jug_iters} iters)")
        else:
            print(f"  ✗ JUG failed")
            jug_total = jug_cache = jug_jit = jug_iter = None
            
    except Exception as e:
        print(f"  ✗ JUG error: {e}")
        jug_total = jug_cache = jug_jit = jug_iter = None
    
    # Clean up
    os.unlink(tim_file)
    
    # Store results
    results.append({
        'n_toas': n_toas,
        'pint_time': pint_time,
        'jug_total': jug_total,
        'jug_cache': jug_cache,
        'jug_jit': jug_jit,
        'jug_iter': jug_iter,
        'jug_iters': jug_iters
    })
    
    # Print comparison
    if pint_time and jug_total:
        print(f"\n  Comparison:")
        print(f"    PINT:           {pint_time:.2f}s")
        print(f"    JUG (total):    {jug_total:.2f}s")
        print(f"    JUG (iter only): {jug_iter:.2f}s")
        print(f"    Speedup (iter): {pint_time/jug_iter:.1f}x")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SCALABILITY SUMMARY")
print("="*80)
print()

print(f"{'TOAs':<10} {'PINT (s)':<12} {'JUG Total (s)':<15} {'JUG Iter (s)':<15} {'Iter Speedup':<15}")
print("-"*80)

for r in results:
    if r['pint_time'] and r['jug_total']:
        speedup = r['pint_time'] / r['jug_iter'] if r['jug_iter'] else 0
        print(f"{r['n_toas']:<10,} {r['pint_time']:<12.2f} {r['jug_total']:<15.2f} {r['jug_iter']:<15.2f} {speedup:<15.1f}x")

print()

# Estimate for 100k TOAs
if len(results) >= 2:
    # Linear fit for iteration time
    valid_results = [r for r in results if r['jug_iter'] is not None and r['pint_time'] is not None]
    if len(valid_results) >= 2:
        n_toas_arr = np.array([r['n_toas'] for r in valid_results])
        jug_iter_arr = np.array([r['jug_iter'] for r in valid_results])
        pint_time_arr = np.array([r['pint_time'] for r in valid_results])
        
        # Fit linear model: time = a * n_toas
        jug_rate = np.mean(jug_iter_arr / n_toas_arr)
        pint_rate = np.mean(pint_time_arr / n_toas_arr)
        
        print(f"Per-TOA timing:")
        print(f"  JUG iterations: {jug_rate*1e6:.2f} μs/TOA")
        print(f"  PINT fitting:   {pint_rate*1e6:.2f} μs/TOA")
        print(f"  Speedup: {pint_rate/jug_rate:.1f}x")

# Save results
with open('SCALABILITY_RESULTS.txt', 'w') as f:
    f.write("SCALABILITY TEST: JUG vs PINT\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'TOAs':<10} {'PINT (s)':<12} {'JUG Total (s)':<15} {'JUG Iter (s)':<15} {'Iter Speedup':<15}\n")
    f.write("-"*80 + "\n")
    
    for r in results:
        if r['pint_time'] and r['jug_total']:
            speedup = r['pint_time'] / r['jug_iter'] if r['jug_iter'] else 0
            f.write(f"{r['n_toas']:<10,} {r['pint_time']:<12.2f} {r['jug_total']:<15.2f} {r['jug_iter']:<15.2f} {speedup:<15.1f}x\n")

print("\n✓ Results saved: SCALABILITY_RESULTS.txt")
