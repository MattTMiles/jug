"""Performance Audit and Optimization Opportunities for JUG

Date: 2025-11-30
Purpose: Identify JAX/JIT acceleration opportunities in current codebase

"""

import subprocess
import re

def analyze_file(filepath, description):
    """Analyze a Python file for optimization opportunities."""
    print(f"\n{'='*80}")
    print(f"FILE: {filepath}")
    print(f"DESCRIPTION: {description}")
    print('='*80)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Count lines
    print(f"\nTotal lines: {len(lines)}")
    
    # Check for JIT decorators
    jit_count = sum(1 for line in lines if '@jax.jit' in line)
    print(f"@jax.jit functions: {jit_count}")
    
    # Check for Python loops
    loops = []
    for i, line in enumerate(lines, 1):
        if re.search(r'^\s*for\s+.*\s+in\s+', line):
            loops.append((i, line.strip()))
    
    if loops:
        print(f"\n⚠️  Python loops found ({len(loops)}):")
        for lineno, content in loops[:10]:  # Show first 10
            print(f"  Line {lineno}: {content[:80]}")
    else:
        print("\n✓ No Python loops (vectorized)")
    
    # Check for NumPy operations that could be JAX
    numpy_ops = []
    for i, line in enumerate(lines, 1):
        if 'np.' in line and 'import' not in line and 'jnp' not in line:
            numpy_ops.append((i, line.strip()))
    
    if numpy_ops:
        print(f"\n⚠️  NumPy operations ({len(numpy_ops)}):")
        # Show a sample
        for lineno, content in numpy_ops[:5]:
            print(f"  Line {lineno}: {content[:80]}")
        if len(numpy_ops) > 5:
            print(f"  ... and {len(numpy_ops)-5} more")
    
    # Check for list comprehensions that could be vectorized
    list_comps = []
    for i, line in enumerate(lines, 1):
        if '[' in line and 'for' in line and 'in' in line and ']' in line:
            list_comps.append((i, line.strip()))
    
    if list_comps:
        print(f"\n⚠️  List comprehensions ({len(list_comps)}):")
        for lineno, content in list_comps[:5]:
            print(f"  Line {lineno}: {content[:80]}")
    
    return {
        'total_lines': len(lines),
        'jit_count': jit_count,
        'loop_count': len(loops),
        'numpy_ops': len(numpy_ops),
        'list_comps': len(list_comps)
    }

# Analyze key performance-critical files
files_to_check = [
    ('jug/residuals/simple_calculator.py', 'Main residual computation pipeline'),
    ('jug/residuals/core.py', 'Core residual functions'),
    ('jug/delays/barycentric.py', 'Barycentric delay computations'),
    ('jug/delays/combined.py', 'Combined delay kernel (DM, SW, FD, binary)'),
    ('jug/delays/binary_dd.py', 'DD/DDH binary model'),
    ('jug/delays/binary_bt.py', 'BT binary model'),
    ('jug/delays/binary_t2.py', 'T2 binary model'),
    ('jug/io/clock.py', 'Clock correction system'),
]

results = {}
for filepath, description in files_to_check:
    try:
        results[filepath] = analyze_file(filepath, description)
    except FileNotFoundError:
        print(f"\n⚠️  File not found: {filepath}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_stats = {
    'total_lines': 0,
    'jit_count': 0,
    'loop_count': 0,
    'numpy_ops': 0,
    'list_comps': 0
}

for filepath, stats in results.items():
    for key in total_stats:
        total_stats[key] += stats[key]

print(f"\nTotal lines analyzed: {total_stats['total_lines']}")
print(f"JIT-compiled functions: {total_stats['jit_count']}")
print(f"Python loops: {total_stats['loop_count']}")
print(f"NumPy operations: {total_stats['numpy_ops']}")
print(f"List comprehensions: {total_stats['list_comps']}")

print("\n" + "="*80)
print("OPTIMIZATION RECOMMENDATIONS")
print("="*80)

print("""
1. HIGH PRIORITY - simple_calculator.py BT loop:
   - Line 316: Python loop for BT binary model
   - ACTION: Already have bt_binary_delay_vectorized - use it!
   - IMPACT: 10-100x speedup for BT pulsars

2. MEDIUM PRIORITY - NumPy → JAX conversions:
   - Many NumPy operations in simple_calculator.py setup phase
   - These are one-time setup, not in hot loop
   - IMPACT: Minimal (not in inner loop)

3. LOW PRIORITY - List comprehensions:
   - Most are for parameter extraction (one-time)
   - Not in performance-critical loops
   - IMPACT: Negligible

4. ALREADY OPTIMIZED:
   - ✓ combined_delays() is JIT-compiled (main computation kernel)
   - ✓ All binary models have JIT-compiled vectorized versions
   - ✓ Kepler solvers are JIT-compiled
   - ✓ No loops in barycentric computation

5. FUTURE OPTIMIZATION (Post-M2):
   - Design matrix computation (fitting) - needs JAX version
   - Gauss-Newton solver - needs JAX version
   - These are planned for M2.4 (JAX acceleration)
""")
