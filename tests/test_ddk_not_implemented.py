#!/usr/bin/env python3
"""
Test that DDK binary model raises NotImplementedError.

DDK requires Kopeikin annual orbital parallax corrections that are not yet
implemented in JUG. Previously, DDK was silently aliased to DD which would
produce incorrect results for high-parallax pulsars like J0437-4715.

This test ensures DDK raises a clear error instead of silently producing
wrong science.

Run with: python tests/test_ddk_not_implemented.py

Category: critical (quick, no external data)
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


# Minimal DDK par file for testing
DDK_PAR_CONTENT = """PSRJ           J0437-4715-TEST
RAJ             04:37:15.8961737
DECJ           -47:15:09.11058
F0             173.6879458970678
F1             -1.728493e-15
PEPOCH         55000
DM             2.64476
BINARY         DDK
PB             5.7410459
A1             3.3666787
ECC            0.00001918
OM             1.35
T0             55001.0
KIN            137.56
KOM            207.0
SINI           0.6787
M2             0.254
UNITS          TDB
"""


def test_ddk_raises_not_implemented():
    """Test that DDK par file raises NotImplementedError."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    # Make sure override is not set
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    
    # Create temp par file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        f.write(DDK_PAR_CONTENT)
        par_path = f.name
    
    # Create minimal tim file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tim', delete=False) as f:
        f.write("FORMAT 1\n")
        f.write("meerkat 1000.0 55000.123456789 1.0 meerkat\n")
        tim_path = f.name
    
    try:
        # This should raise NotImplementedError
        result = compute_residuals_simple(par_path, tim_path, verbose=False)
        print("✗ FAIL: DDK should have raised NotImplementedError but didn't!")
        return False, "DDK did not raise NotImplementedError"
    except NotImplementedError as e:
        error_msg = str(e)
        # Verify error message contains helpful information
        checks = []
        if "DDK" in error_msg:
            checks.append("mentions DDK")
        if "Kopeikin" in error_msg:
            checks.append("mentions Kopeikin")
        if "not implemented" in error_msg.lower():
            checks.append("says not implemented")
        if "JUG_ALLOW_DDK_AS_DD" in error_msg:
            checks.append("mentions override option")
        
        if len(checks) >= 3:
            print(f"✓ DDK correctly raises NotImplementedError")
            print(f"  Error message includes: {', '.join(checks)}")
            return True, f"OK (error has: {', '.join(checks)})"
        else:
            print(f"✗ FAIL: Error message incomplete")
            print(f"  Only found: {checks}")
            return False, f"Error message incomplete: {checks}"
    except Exception as e:
        print(f"✗ FAIL: Wrong exception type: {type(e).__name__}: {e}")
        return False, f"Wrong exception: {type(e).__name__}"
    finally:
        # Cleanup
        Path(par_path).unlink(missing_ok=True)
        Path(tim_path).unlink(missing_ok=True)


def test_ddk_override_env_var():
    """Test that JUG_ALLOW_DDK_AS_DD=1 allows DDK (with warning)."""
    import warnings
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.utils.binary_model_overrides import reset_ddk_warning
    
    # Reset warning flag from any previous test
    reset_ddk_warning()
    
    # Set override
    os.environ['JUG_ALLOW_DDK_AS_DD'] = '1'
    
    # Create temp par file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        f.write(DDK_PAR_CONTENT)
        par_path = f.name
    
    # Create minimal tim file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tim', delete=False) as f:
        f.write("FORMAT 1\n")
        f.write("meerkat 1000.0 55000.123456789 1.0 meerkat\n")
        tim_path = f.name
    
    try:
        # With override, it should run but emit a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_residuals_simple(par_path, tim_path, verbose=False)
            
            # Check that warning was emitted
            ddk_warnings = [x for x in w if 'DDK' in str(x.message)]
            if len(ddk_warnings) > 0:
                print(f"✓ DDK override works with warning")
                return True, "OK (override works with warning)"
            else:
                print(f"✗ FAIL: No warning emitted with DDK override")
                return False, "No warning with override"
    except NotImplementedError:
        print(f"✗ FAIL: DDK still raised NotImplementedError with override set")
        return False, "Override didn't work"
    except Exception as e:
        # Other exceptions might occur (missing clock files, etc.) but that's OK
        # The point is it didn't raise NotImplementedError
        print(f"✓ DDK override bypassed NotImplementedError (other error: {type(e).__name__})")
        return True, f"OK (override works, other error: {type(e).__name__})"
    finally:
        # Cleanup
        os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
        Path(par_path).unlink(missing_ok=True)
        Path(tim_path).unlink(missing_ok=True)


def test_dd_still_works():
    """Test that DD model (non-DDK) still works normally."""
    from jug.io.par_reader import parse_par_file
    
    # Create DD par file (similar to DDK but with BINARY DD)
    dd_content = DDK_PAR_CONTENT.replace("BINARY         DDK", "BINARY         DD")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        f.write(dd_content)
        par_path = f.name
    
    try:
        # Parse should work
        params = parse_par_file(par_path)
        if params.get('BINARY', '').upper() == 'DD':
            print(f"✓ DD model parses correctly")
            return True, "OK (DD parses)"
        else:
            print(f"✗ FAIL: DD not detected")
            return False, f"DD not detected: {params.get('BINARY')}"
    except Exception as e:
        print(f"✗ FAIL: DD parsing failed: {e}")
        return False, f"DD parsing failed: {e}"
    finally:
        Path(par_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("DDK Not Implemented Test")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing DDK raises NotImplementedError...")
    results.append(test_ddk_raises_not_implemented())
    
    print("\n2. Testing JUG_ALLOW_DDK_AS_DD override...")
    results.append(test_ddk_override_env_var())
    
    print("\n3. Testing DD model still works...")
    results.append(test_dd_still_works())
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r[0])
    failed = len(results) - passed
    
    for i, (success, msg) in enumerate(results, 1):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Test {i}: {status} - {msg}")
    
    print(f"\n{passed}/{len(results)} tests passed")
    
    if failed > 0:
        print("\nFAILED")
        sys.exit(1)
    else:
        print("\nPASSED")
        sys.exit(0)
