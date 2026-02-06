#!/usr/bin/env python
"""Unit tests for jug/utils/binary_model_overrides.py.

Tests the centralized DDK override helper functions:
- is_ddk_override_allowed()
- resolve_binary_model()
- reset_ddk_warning()

Run with: python tests/test_binary_model_overrides.py
"""

import os
import sys
import warnings
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def test_import():
    """Test that the helper module imports correctly."""
    from jug.utils.binary_model_overrides import (
        resolve_binary_model,
        is_ddk_override_allowed,
        reset_ddk_warning,
        DDK_ALIASING_INFO,  # Changed from DDK_NOT_IMPLEMENTED_ERROR
        DDK_OVERRIDE_WARNING,
    )
    return True, "OK"


def test_is_ddk_override_allowed_unset():
    """Test is_ddk_override_allowed returns False when env var unset."""
    from jug.utils.binary_model_overrides import is_ddk_override_allowed
    
    # Ensure env var is unset
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    
    if is_ddk_override_allowed():
        return False, "returned True when env var unset"
    return True, "OK (returns False)"


def test_is_ddk_override_allowed_set():
    """Test is_ddk_override_allowed returns True for various truthy values."""
    from jug.utils.binary_model_overrides import is_ddk_override_allowed
    
    truthy_values = ['1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES']
    
    for val in truthy_values:
        os.environ['JUG_ALLOW_DDK_AS_DD'] = val
        if not is_ddk_override_allowed():
            os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
            return False, f"returned False for '{val}'"
    
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    return True, f"OK (truthy: {truthy_values})"


def test_is_ddk_override_allowed_falsy():
    """Test is_ddk_override_allowed returns False for various falsy values."""
    from jug.utils.binary_model_overrides import is_ddk_override_allowed
    
    falsy_values = ['0', 'false', 'no', '', 'anything']
    
    for val in falsy_values:
        os.environ['JUG_ALLOW_DDK_AS_DD'] = val
        if is_ddk_override_allowed():
            os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
            return False, f"returned True for '{val}'"
    
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    return True, f"OK (falsy: {falsy_values})"


def test_resolve_non_ddk_passthrough():
    """Test that non-DDK models pass through unchanged."""
    from jug.utils.binary_model_overrides import resolve_binary_model
    
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    
    models = ['DD', 'ELL1', 'ELL1H', 'DDH', 'DDGR', 'BT', 'T2', 'dd', 'ell1']
    for model in models:
        result = resolve_binary_model(model)
        if result != model.upper():
            return False, f"'{model}' -> '{result}' (expected '{model.upper()}')"
    
    return True, "OK (non-DDK passthrough)"


def test_resolve_ddk_raises_without_override():
    """Test that DDK returns 'DDK' unchanged when DDK is implemented.
    
    Note: This test was originally for the NotImplementedError behavior.
    Now that DDK is fully implemented, it just returns 'DDK' unchanged.
    """
    from jug.utils.binary_model_overrides import resolve_binary_model
    
    os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
    
    result = resolve_binary_model('DDK')
    if result != 'DDK':
        return False, f"returned '{result}' (expected 'DDK')"
    return True, "OK (returns 'DDK' unchanged - DDK now implemented)"


def test_resolve_ddk_returns_dd_with_override():
    """Test that DDK returns 'DD' when override is set."""
    from jug.utils.binary_model_overrides import resolve_binary_model, reset_ddk_warning
    
    reset_ddk_warning()
    os.environ['JUG_ALLOW_DDK_AS_DD'] = '1'
    
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = resolve_binary_model('DDK')
            if result != 'DD':
                return False, f"returned '{result}' (expected 'DD')"
        return True, "OK (returns 'DD')"
    finally:
        os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
        reset_ddk_warning()


def test_resolve_ddk_warns_once():
    """Test that DDK override warns exactly once per process (dedupe)."""
    from jug.utils.binary_model_overrides import resolve_binary_model, reset_ddk_warning
    
    reset_ddk_warning()
    os.environ['JUG_ALLOW_DDK_AS_DD'] = '1'
    
    try:
        # First call should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_binary_model('DDK', warn=True)
            first_call_warnings = len(w)
        
        if first_call_warnings != 1:
            return False, f"first call issued {first_call_warnings} warnings (expected 1)"
        
        # Second call should NOT warn (dedupe)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_binary_model('DDK', warn=True)
            second_call_warnings = len(w)
        
        if second_call_warnings != 0:
            return False, f"second call issued {second_call_warnings} warnings (expected 0)"
        
        return True, "OK (warns once, dedupe works)"
    finally:
        os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
        reset_ddk_warning()


def test_reset_ddk_warning():
    """Test that reset_ddk_warning restores warning emission."""
    from jug.utils.binary_model_overrides import resolve_binary_model, reset_ddk_warning
    
    reset_ddk_warning()
    os.environ['JUG_ALLOW_DDK_AS_DD'] = '1'
    
    try:
        # First call warns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_binary_model('DDK', warn=True)
        
        if len(w) != 1:
            return False, f"first call: {len(w)} warnings"
        
        # Reset and call again - should warn again
        reset_ddk_warning()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve_binary_model('DDK', warn=True)
        
        if len(w) != 1:
            return False, f"after reset: {len(w)} warnings (expected 1)"
        
        return True, "OK (reset restores warning)"
    finally:
        os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
        reset_ddk_warning()


def test_warn_false_suppresses():
    """Test that warn=False suppresses warning even on first call."""
    from jug.utils.binary_model_overrides import resolve_binary_model, reset_ddk_warning
    
    reset_ddk_warning()
    os.environ['JUG_ALLOW_DDK_AS_DD'] = '1'
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_binary_model('DDK', warn=False)
        
        if len(w) != 0:
            return False, f"warn=False still issued {len(w)} warnings"
        if result != 'DD':
            return False, f"returned '{result}' (expected 'DD')"
        
        return True, "OK (warn=False suppresses)"
    finally:
        os.environ.pop('JUG_ALLOW_DDK_AS_DD', None)
        reset_ddk_warning()


def main():
    """Run all tests."""
    print("=" * 60)
    print("binary_model_overrides helper unit tests")
    print("=" * 60)
    
    tests = [
        ("Import", test_import),
        ("is_ddk_override_allowed (unset)", test_is_ddk_override_allowed_unset),
        ("is_ddk_override_allowed (truthy)", test_is_ddk_override_allowed_set),
        ("is_ddk_override_allowed (falsy)", test_is_ddk_override_allowed_falsy),
        ("resolve non-DDK passthrough", test_resolve_non_ddk_passthrough),
        ("resolve DDK raises without override", test_resolve_ddk_raises_without_override),
        ("resolve DDK returns DD with override", test_resolve_ddk_returns_dd_with_override),
        ("resolve DDK warns once (dedupe)", test_resolve_ddk_warns_once),
        ("reset_ddk_warning restores warning", test_reset_ddk_warning),
        ("warn=False suppresses warning", test_warn_false_suppresses),
    ]
    
    all_passed = True
    
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  [FAIL] {name}: exception: {e}")
            all_passed = False
    
    print()
    if all_passed:
        print("All binary_model_overrides tests PASSED")
        return 0
    else:
        print("Some binary_model_overrides tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
