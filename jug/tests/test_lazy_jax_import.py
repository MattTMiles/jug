"""Tests for lazy JAX import behavior.

These tests verify that JAX is NOT imported at package import time,
which is important for:
1. Faster cold starts for non-JAX operations (codecs, tests, tools)
2. Easier parity/determinism testing in environments without JAX
3. Reduced import overhead for lightweight operations
"""

import subprocess
import sys


def test_import_jug_does_not_import_jax():
    """Verify that 'import jug' doesn't import JAX.

    This test runs in a subprocess to ensure clean module state.
    """
    # Script that checks if JAX is in sys.modules after importing jug
    script = '''
import sys

# Verify JAX is not already imported
assert 'jax' not in sys.modules, "JAX was already imported before test"

# Import jug
import jug

# Check if JAX got imported
if 'jax' in sys.modules:
    print("FAIL: JAX was imported by 'import jug'")
    sys.exit(1)
else:
    print("PASS: JAX not imported by 'import jug'")
    sys.exit(0)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"JAX was imported when 'import jug' was called.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_import_jug_model_does_not_import_jax():
    """Verify that importing jug.model (ParameterSpec, codecs) doesn't import JAX."""
    script = '''
import sys

assert 'jax' not in sys.modules, "JAX was already imported before test"

# Import model modules (should not need JAX)
from jug.model.parameter_spec import get_spec, list_fittable_params
from jug.model.codecs import RAJCodec, DECJCodec

# Check if JAX got imported
if 'jax' in sys.modules:
    print("FAIL: JAX was imported by jug.model imports")
    sys.exit(1)
else:
    print("PASS: JAX not imported by jug.model imports")
    sys.exit(0)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"JAX was imported when importing jug.model.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_import_jug_io_does_not_import_jax():
    """Verify that importing jug.io (par/tim readers) doesn't import JAX."""
    script = '''
import sys

assert 'jax' not in sys.modules, "JAX was already imported before test"

# Import I/O modules (should not need JAX)
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds

# Check if JAX got imported
if 'jax' in sys.modules:
    print("FAIL: JAX was imported by jug.io imports")
    sys.exit(1)
else:
    print("PASS: JAX not imported by jug.io imports")
    sys.exit(0)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"JAX was imported when importing jug.io.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_ensure_jax_x64_is_idempotent():
    """Verify that ensure_jax_x64() can be called multiple times safely."""
    from jug.utils.jax_setup import ensure_jax_x64, is_jax_configured

    # First call
    result1 = ensure_jax_x64()
    assert result1 is True, "First call to ensure_jax_x64() should succeed"
    assert is_jax_configured(), "JAX should be configured after first call"

    # Second call (should be no-op but return True)
    result2 = ensure_jax_x64()
    assert result2 is True, "Second call to ensure_jax_x64() should succeed"

    # Third call
    result3 = ensure_jax_x64()
    assert result3 is True, "Third call to ensure_jax_x64() should succeed"


def test_jax_x64_actually_enabled():
    """Verify that after ensure_jax_x64(), JAX actually has x64 enabled."""
    from jug.utils.jax_setup import ensure_jax_x64

    ensure_jax_x64()

    import jax
    import jax.numpy as jnp

    # Check config
    assert jax.config.jax_enable_x64, "jax_enable_x64 should be True"

    # Verify with actual computation
    x = jnp.array([1.0, 2.0, 3.0])
    assert x.dtype == jnp.float64, f"Expected float64, got {x.dtype}"


def test_assert_jax_x64_raises_before_config():
    """Verify that assert_jax_x64() raises if JAX not configured."""
    # This test needs a subprocess to have clean state
    script = '''
import sys
from jug.utils.jax_setup import _reset_for_testing, assert_jax_x64

# Reset state to simulate unconfigured JAX
_reset_for_testing()

try:
    assert_jax_x64()
    print("FAIL: assert_jax_x64() should have raised RuntimeError")
    sys.exit(1)
except RuntimeError as e:
    if "not configured" in str(e).lower():
        print("PASS: assert_jax_x64() raised RuntimeError as expected")
        sys.exit(0)
    else:
        print(f"FAIL: Wrong error message: {e}")
        sys.exit(1)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"assert_jax_x64() did not raise as expected.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


if __name__ == '__main__':
    # Run tests when executed directly
    print("Running lazy JAX import tests...\n")

    test_import_jug_does_not_import_jax()
    print()

    test_import_jug_model_does_not_import_jax()
    print()

    test_import_jug_io_does_not_import_jax()
    print()

    test_ensure_jax_x64_is_idempotent()
    print("PASS: ensure_jax_x64() is idempotent\n")

    test_jax_x64_actually_enabled()
    print("PASS: JAX x64 actually enabled\n")

    test_assert_jax_x64_raises_before_config()
    print()

    print("All tests passed!")
