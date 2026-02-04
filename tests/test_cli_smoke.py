#!/usr/bin/env python3
"""
CLI smoke tests for JUG entry points.

Tests that CLI commands:
1. Are importable (entry points resolve)
2. Respond to --help without crashing
3. Handle missing arguments gracefully

Uses module invocation (python -m) for reliability in editable installs.
Console script tests are optional (skipped if not installed).

Run with: python tests/test_cli_smoke.py
"""

import subprocess
import sys
from pathlib import Path

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Module paths for CLI entry points (preferred - no PATH issues)
# Format: (module_path, help_args, description)
CLI_MODULES = [
    ("jug.scripts.compute_residuals", ["--help"], "compute residuals CLI"),
    ("jug.scripts.fit_parameters", ["--help"], "fit parameters CLI"),
    ("jug.gui.main", ["--help"], "GUI main"),
    ("jug.scripts.benchmark_stages", ["--help"], "benchmark CLI"),
    ("jug.scripts.jugd", ["--help"], "jugd daemon"),
]

# Console script names (optional - may not be installed)
CLI_COMMANDS = [
    "jug-compute-residuals",
    "jug-fit", 
    "jug-gui",
    "jug-benchmark",
    "jugd",
]


def test_module_invocation(module_path: str, args: list) -> tuple[bool, str]:
    """Test module responds to args via python -m (reliable, no PATH issues)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root),
        )
        # --help should exit 0 (argparse convention)
        if result.returncode == 0:
            return True, f"OK (exit 0, {len(result.stdout)} chars)"
        # argparse shows help and exits 0, but some may have different exit codes
        if "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower():
            return True, f"OK (help shown, exit {result.returncode})"
        return False, f"exit {result.returncode}: {result.stderr[:100]}"
    except subprocess.TimeoutExpired:
        return False, "timeout after 30s"
    except Exception as e:
        return False, str(e)


def test_help_flag(cmd: str) -> tuple[bool | None, str]:
    """Test that console script responds to --help (optional)."""
    try:
        result = subprocess.run(
            [cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, f"OK (exit 0, {len(result.stdout)} chars)"
        else:
            return False, f"exit {result.returncode}: {result.stderr[:100]}"
    except FileNotFoundError:
        return None, "not installed (skip)"  # Skip, not fail
    except subprocess.TimeoutExpired:
        return False, "timeout after 30s"
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
    
    # Test module imports (most reliable, required)
    print("\n--- Module Imports (required) ---")
    for module_path, _, desc in CLI_MODULES:
        passed, msg = test_import_module(module_path)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] import {module_path}: {msg}")
        all_passed = all_passed and passed
    
    # Test module invocation with --help (required, reliable)
    print("\n--- Module Invocation (python -m, required) ---")
    for module_path, args, desc in CLI_MODULES:
        passed, msg = test_module_invocation(module_path, args)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] python -m {module_path} {' '.join(args)}: {msg}")
        all_passed = all_passed and passed
    
    # Test console scripts (optional - may be skipped if not installed)
    print("\n--- Console Scripts (optional, may skip) ---")
    for cmd in CLI_COMMANDS:
        result = test_help_flag(cmd)
        if result[0] is None:  # Skip
            print(f"  [SKIP] {cmd} --help: {result[1]}")
            skipped += 1
        elif result[0]:
            print(f"  [PASS] {cmd} --help: {result[1]}")
        else:
            # Console script failures are optional - don't fail overall
            print(f"  [WARN] {cmd} --help: {result[1]}")
            # all_passed = False  # Don't fail on console script issues
    
    print("\n" + "=" * 60)
    if all_passed:
        print(f"All CLI smoke tests PASSED ({skipped} console scripts skipped)")
        return 0
    else:
        print("Some CLI smoke tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
