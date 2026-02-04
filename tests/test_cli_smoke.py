#!/usr/bin/env python3
"""
CLI smoke tests for JUG entry points.

Tests that CLI commands:
1. Are importable (entry points resolve)
2. Respond to --help without crashing
3. Handle missing arguments gracefully

Prefers python -m invocation to avoid PATH issues in editable installs.

Run with: python tests/test_cli_smoke.py
"""

import subprocess
import sys
from pathlib import Path

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Module paths for CLI entry points (preferred - no PATH issues)
# Format: (module_path, function_name)
CLI_MODULES = [
    ("jug.scripts.compute_residuals", "main"),
    ("jug.scripts.fit_parameters", "main"),
    ("jug.gui.main", "main"),
    ("jug.scripts.benchmark_stages", "main"),
    ("jug.scripts.jugd", "main"),
]

# Console script names (may not be installed in all environments)
CLI_COMMANDS = [
    "jug-compute-residuals",
    "jug-fit", 
    "jug-gui",
    "jug-benchmark",
    "jugd",
]


def test_help_flag(cmd: str) -> tuple[bool, str]:
    """Test that command responds to --help."""
    try:
        result = subprocess.run(
            [cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # --help should exit 0
        if result.returncode == 0:
            return True, f"OK (exit 0, {len(result.stdout)} chars)"
        else:
            return False, f"exit {result.returncode}: {result.stderr[:100]}"
    except FileNotFoundError:
        return None, "not installed (editable install?)"  # Skip, not fail
    except subprocess.TimeoutExpired:
        return False, "timeout after 30s"
    except Exception as e:
        return False, str(e)


def test_module_help(module_path: str) -> tuple[bool, str]:
    """Test that module responds to -h via python -m (preferred, no PATH issues)."""
    try:
        # Use python -m to avoid PATH/entry point issues
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_path}; {module_path}.main(['--help'])"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root),
        )
        # --help typically causes SystemExit(0)
        if result.returncode == 0 or "usage:" in result.stdout.lower() or "help" in result.stdout.lower():
            return True, f"OK ({len(result.stdout)} chars)"
        # argparse with --help raises SystemExit(0), but some may exit differently
        if result.returncode == 2 and "error" not in result.stderr.lower():
            return True, "OK (argparse help)"
        return False, f"exit {result.returncode}: {result.stderr[:100]}"
    except Exception as e:
        return False, str(e)


def test_import_module(module_path: str) -> tuple[bool, str]:
    """Test that module is importable."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_path}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, "OK"
        else:
            return False, result.stderr[:200]
    except Exception as e:
        return False, str(e)


def main():
    """Run all CLI smoke tests."""
    print("=" * 60)
    print("CLI Smoke Tests")
    print("=" * 60)
    
    all_passed = True
    skipped = 0
    
    # Test module imports (most reliable)
    print("\n--- Module Imports ---")
    for module_path, _ in CLI_MODULES:
        passed, msg = test_import_module(module_path)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] import {module_path}: {msg}")
        all_passed = all_passed and passed
    
    # Test console scripts (may be skipped if not installed)
    print("\n--- Console Scripts (--help) ---")
    for cmd in CLI_COMMANDS:
        result = test_help_flag(cmd)
        if result[0] is None:  # Skip
            print(f"  [SKIP] {cmd} --help: {result[1]}")
            skipped += 1
        elif result[0]:
            print(f"  [PASS] {cmd} --help: {result[1]}")
        else:
            print(f"  [FAIL] {cmd} --help: {result[1]}")
            all_passed = False
    
    # Test missing args (should fail gracefully)
    print("\n--- Missing Args Handling ---")
    for cmd in ["jug-compute-residuals", "jug-fit"]:
        try:
            result = subprocess.run(
                [cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should exit non-zero when missing required args
            if result.returncode != 0:
                print(f"  [PASS] {cmd}: exits non-zero when args missing")
            else:
                print(f"  [WARN] {cmd}: exits 0 with no args (unexpected)")
        except FileNotFoundError:
            print(f"  [SKIP] {cmd}: not installed")
            skipped += 1
        except Exception as e:
            print(f"  [FAIL] {cmd}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print(f"All CLI smoke tests PASSED ({skipped} skipped)")
        return 0
    else:
        print("Some CLI smoke tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
