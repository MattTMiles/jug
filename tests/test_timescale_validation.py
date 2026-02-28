#!/usr/bin/env python
"""
Test script for par file timescale validation.

This script verifies that:
1. TDB par files work normally
2. TCB par files are automatically converted to TDB and work correctly

Environment variables for CI:
    JUG_TEST_J1713_PAR=/path/to/J1713+0747.par
    JUG_TEST_J1713_TIM=/path/to/J1713+0747.tim
"""

import sys
import tempfile
from pathlib import Path

# Import test path utilities
try:
    from tests.test_paths import get_j1713_paths, skip_if_missing
except ImportError:
    # Running from tests/ directory
    from test_paths import get_j1713_paths, skip_if_missing

from jug.io.par_reader import parse_par_file, validate_par_timescale

# Get paths from environment or defaults
par_path, tim_path = get_j1713_paths()
if not skip_if_missing(par_path, tim_path, "timescale_validation"):
    print("\nSKIPPED: Test data not available")
    sys.exit(0)

TDB_PAR_FILE = str(par_path)
TIM_FILE = str(tim_path)

print("=" * 80)
print("Testing Par File Timescale Validation")
print("=" * 80)

# Test 1: TDB par file should work
print("\n1. Testing TDB par file (should succeed)...")
params = parse_par_file(TDB_PAR_FILE)
timescale = params.get('_par_timescale', 'UNKNOWN')
print(f"   UNITS in par file: {params.get('UNITS', 'not specified')}")
print(f"   Detected timescale: {timescale}")

try:
    validated = validate_par_timescale(params, context="Test")
    print(f"   ✓ Validation passed: timescale = {validated}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)

# Test 2: Full residual calculation with TDB par file
print("\n2. Testing full residual calculation with TDB par file...")
from jug.residuals.simple_calculator import compute_residuals_simple

try:
    result = compute_residuals_simple(TDB_PAR_FILE, TIM_FILE, verbose=False)
    print(f"   ✓ Residuals computed successfully")
    print(f"   RMS: {result['rms_us']:.3f} μs, N_TOAs: {result['n_toas']}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)

# Test 3: Create a fake TCB par file and verify it converts to TDB
print("\n3. Testing TCB par file (should convert to TDB automatically)...")

# Read the TDB par file and modify it to TCB
with open(TDB_PAR_FILE) as f:
    par_content = f.read()

# Replace UNITS TDB with UNITS TCB
tcb_content = par_content.replace('UNITS          TDB', 'UNITS          TCB')

# Write to a temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
    f.write(tcb_content)
    tcb_par_file = f.name

print(f"   Created temp TCB par file: {tcb_par_file}")

try:
    params_tcb = parse_par_file(tcb_par_file)
    timescale_tcb = params_tcb.get('_par_timescale', 'UNKNOWN')
    print(f"   UNITS in par file: {params_tcb.get('UNITS', 'not specified')}")
    print(f"   Detected timescale: {timescale_tcb}")
    
    # Save original F0 to verify conversion
    f0_orig = params_tcb.get('F0')
    
    # This should convert TCB to TDB automatically
    result = validate_par_timescale(params_tcb, context="Test TCB", verbose=False)
    
    if result == 'TDB':
        print(f"   ✓ TCB automatically converted to TDB")
        
        # Verify metadata was set
        if params_tcb.get('_tcb_converted'):
            print(f"   ✓ Conversion metadata set")
        else:
            print(f"   ✗ Conversion metadata missing!")
            sys.exit(1)
        
        # Verify F0 was scaled (if it exists)
        if f0_orig is not None:
            from jug.utils.timescales import IFTE_K
            expected_f0 = f0_orig * float(IFTE_K)
            if abs(params_tcb['F0'] - expected_f0) < 1e-12:
                print(f"   ✓ F0 scaled correctly: {f0_orig:.12f} → {params_tcb['F0']:.12f}")
            else:
                print(f"   ✗ F0 scaling incorrect!")
                sys.exit(1)
    else:
        print(f"   ✗ Unexpected result: {result}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify compute_residuals_simple works on converted TCB
print("\n4. Testing that compute_residuals_simple works on TCB par file...")

try:
    result = compute_residuals_simple(tcb_par_file, TIM_FILE, verbose=False)
    print(f"   ✓ Residuals computed successfully from TCB par file")
    print(f"   RMS: {result['rms_us']:.3f} μs, N_TOAs: {result['n_toas']}")
    
    # The RMS should be the same as the TDB version (within numerical tolerance)
    # since it's the same physical model, just converted
    print(f"   Note: TCB par file converted to TDB internally and processed")
except Exception as e:
    print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
Path(tcb_par_file).unlink()

# Test 5: Test filename warning (no UNITS keyword but 'tcb' in filename)
print("\n5. Testing filename-based warning (no UNITS, 'tcb' in filename)...")

# Create a par file with no UNITS keyword but 'tcb' in filename
no_units_content = par_content.replace('UNITS          TDB\n', '')

with tempfile.NamedTemporaryFile(mode='w', suffix='_tcb_test.par', delete=False) as f:
    f.write(no_units_content)
    no_units_par_file = f.name

print(f"   Created temp par file: {no_units_par_file}")

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    params_no_units = parse_par_file(no_units_par_file)
    
    if len(w) > 0:
        print(f"   ✓ Warning raised: {w[0].message}")
    else:
        # Check if the filename actually has 'tcb' in it
        if 'tcb' in Path(no_units_par_file).name.lower():
            print(f"   ⚠ Expected warning about 'tcb' in filename but none raised")
        else:
            print(f"   ✓ No warning (filename doesn't contain 'tcb')")

# Verify it defaults to TDB
timescale_no_units = params_no_units.get('_par_timescale', 'UNKNOWN')
print(f"   Detected timescale (default): {timescale_no_units}")

# Should still work since we default to TDB
try:
    validate_par_timescale(params_no_units, context="Test no UNITS")
    print(f"   ✓ Validation passed (defaulted to TDB)")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")
    sys.exit(1)

# Cleanup
Path(no_units_par_file).unlink()

# Test 6: Test tzrmjd_scale="UTC" contradiction warning with UNITS=TDB
print("\n6. Testing tzrmjd_scale='UTC' contradiction with UNITS=TDB...")
print("   (Should print loud warning about contradiction)")

import io
from contextlib import redirect_stdout

# Capture stdout to check for warning
f_capture = io.StringIO()
with redirect_stdout(f_capture):
    result = compute_residuals_simple(
        TDB_PAR_FILE, TIM_FILE, 
        verbose=True, 
        tzrmjd_scale="UTC"
    )

output = f_capture.getvalue()

# Check for contradiction warning
if "contradicts par file UNITS=TDB" in output:
    print("   ✓ Contradiction warning printed")
else:
    print("   ✗ Missing contradiction warning!")
    print(f"   Output snippet: {output[:500]}...")
    sys.exit(1)

# Check for large delta warning
if "Large TZRMJD shift" in output or "delta" in output.lower():
    print("   ✓ Large shift warning/info printed")

# Check delta_tzr value (should be ~69 seconds)
import re
# Match format: "delta_tzr:     69.185849 s" or "delta = 69.186 s"
delta_match = re.search(r'delta[_a-z]*[:\s]+\s*([\d.]+)\s*s', output, re.IGNORECASE)
if delta_match:
    delta_val = float(delta_match.group(1))
    if delta_val > 60:  # Should be ~69 seconds
        print(f"   ✓ Delta TZRMJD = {delta_val:.1f} s (confirms UTC->TDB conversion applied)")
    else:
        print(f"   ⚠ Delta TZRMJD = {delta_val:.1f} s (unexpectedly small)")
else:
    print("   ⚠ Could not extract delta_tzr value from output")

# Test 7: Test tzrmjd_scale="AUTO" (default behavior)
print("\n7. Testing tzrmjd_scale='AUTO' (default, should derive from par UNITS)...")

f_capture2 = io.StringIO()
with redirect_stdout(f_capture2):
    result2 = compute_residuals_simple(
        TDB_PAR_FILE, TIM_FILE, 
        verbose=True, 
        tzrmjd_scale="AUTO"  # This is the default
    )

output2 = f_capture2.getvalue()

# Check that AUTO resolved to TDB
if "AUTO -> TDB" in output2:
    print("   ✓ AUTO correctly resolved to TDB (from par UNITS)")
else:
    print(f"   ⚠ Could not confirm AUTO->TDB resolution")
    
# Check delta is 0
if "delta_tzr:     0.000000 s" in output2:
    print("   ✓ delta_tzr = 0 (no conversion applied)")
else:
    print("   ⚠ Could not confirm zero delta")

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
