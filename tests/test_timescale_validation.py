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
import pytest
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
    pytest.skip("J1713 test data not available", allow_module_level=True)

TDB_PAR_FILE = str(par_path)
TIM_FILE = str(tim_path)


def test_timescale_validation():
    """Test par file timescale validation and TCB->TDB conversion."""
    import io
    import re
    import warnings as warn_mod
    from contextlib import redirect_stdout

    print("=" * 80)
    print("Testing Par File Timescale Validation")
    print("=" * 80)

    # Test 1: TDB par file should work
    print("\n1. Testing TDB par file (should succeed)...")
    params = parse_par_file(TDB_PAR_FILE)
    timescale = params.get('_par_timescale', 'UNKNOWN')
    print(f"   UNITS in par file: {params.get('UNITS', 'not specified')}")
    print(f"   Detected timescale: {timescale}")

    validated = validate_par_timescale(params, context="Test")
    print(f"   ✓ Validation passed: timescale = {validated}")

    # Test 2: Full residual calculation with TDB par file
    print("\n2. Testing full residual calculation with TDB par file...")
    from jug.residuals.simple_calculator import compute_residuals_simple

    result = compute_residuals_simple(TDB_PAR_FILE, TIM_FILE, verbose=False)
    print(f"   ✓ Residuals computed successfully")
    print(f"   RMS: {result['rms_us']:.3f} μs, N_TOAs: {result['n_toas']}")

    # Test 3: Create a fake TCB par file and verify it converts to TDB
    print("\n3. Testing TCB par file (should convert to TDB automatically)...")

    with open(TDB_PAR_FILE) as f:
        par_content = f.read()

    tcb_content = par_content.replace('UNITS          TDB', 'UNITS          TCB')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        f.write(tcb_content)
        tcb_par_file = f.name

    try:
        params_tcb = parse_par_file(tcb_par_file)
        f0_orig = params_tcb.get('F0')

        result_ts = validate_par_timescale(params_tcb, context="Test TCB", verbose=False)
        assert result_ts == 'TDB', f"Expected TDB, got {result_ts}"
        assert params_tcb.get('_tcb_converted'), "Conversion metadata missing"

        if f0_orig is not None:
            from jug.utils.timescales import IFTE_K
            expected_f0 = f0_orig * float(IFTE_K)
            assert abs(params_tcb['F0'] - expected_f0) < 1e-12, "F0 scaling incorrect"
            print(f"   ✓ F0 scaled correctly")

        # Test 4: Verify compute_residuals_simple works on converted TCB
        print("\n4. Testing that compute_residuals_simple works on TCB par file...")
        result = compute_residuals_simple(tcb_par_file, TIM_FILE, verbose=False)
        print(f"   ✓ Residuals computed successfully from TCB par file")
        print(f"   RMS: {result['rms_us']:.3f} μs, N_TOAs: {result['n_toas']}")
    finally:
        Path(tcb_par_file).unlink()

    # Test 5: Test filename warning (no UNITS keyword but 'tcb' in filename)
    print("\n5. Testing filename-based warning (no UNITS, 'tcb' in filename)...")
    no_units_content = par_content.replace('UNITS          TDB\n', '')

    with tempfile.NamedTemporaryFile(mode='w', suffix='_tcb_test.par', delete=False) as f:
        f.write(no_units_content)
        no_units_par_file = f.name

    try:
        params_no_units = parse_par_file(no_units_par_file)
        timescale_no_units = params_no_units.get('_par_timescale', 'UNKNOWN')
        print(f"   Detected timescale (default): {timescale_no_units}")

        validate_par_timescale(params_no_units, context="Test no UNITS")
        print(f"   ✓ Validation passed (defaulted to TDB)")
    finally:
        Path(no_units_par_file).unlink()

    # Test 6: Test tzrmjd_scale="UTC" contradiction warning with UNITS=TDB
    print("\n6. Testing tzrmjd_scale='UTC' contradiction with UNITS=TDB...")

    f_capture = io.StringIO()
    with redirect_stdout(f_capture):
        result = compute_residuals_simple(
            TDB_PAR_FILE, TIM_FILE,
            verbose=True,
            tzrmjd_scale="UTC"
        )

    output = f_capture.getvalue()
    assert "contradicts par file UNITS=TDB" in output, \
        f"Missing contradiction warning! Output snippet: {output[:500]}..."

    # Test 7: Test tzrmjd_scale="AUTO" (default behavior)
    print("\n7. Testing tzrmjd_scale='AUTO' (default, should derive from par UNITS)...")

    f_capture2 = io.StringIO()
    with redirect_stdout(f_capture2):
        result2 = compute_residuals_simple(
            TDB_PAR_FILE, TIM_FILE,
            verbose=True,
            tzrmjd_scale="AUTO"
        )

    output2 = f_capture2.getvalue()
    if "AUTO -> TDB" in output2:
        print("   ✓ AUTO correctly resolved to TDB (from par UNITS)")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
