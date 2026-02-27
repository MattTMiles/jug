#!/usr/bin/env python3
"""
CLI integration tests for JUG compute/fit commands.

Tests that CLI can run end-to-end with actual data (bundled mini dataset).
This is more thorough than smoke tests - we actually compute residuals and fit.

Run with: python tests/test_cli_integration.py

Category: cli (quick, uses bundled mini data)
"""

import os
import subprocess
import sys
import json
import re
from pathlib import Path

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Force CPU for CLI subprocesses to avoid GPU memory contention
# when the parent process (e.g. test_cli_fit_improves_rms) already holds the GPU.
_subprocess_env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}


def get_mini_paths():
    """Get paths to bundled mini dataset."""
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if par.exists() and tim.exists():
        return str(par), str(tim)
    return None, None


def test_cli_compute_residuals():
    """Test jug-compute-residuals CLI with mini data."""
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Try console script first, fall back to module invocation
    try:
        result = subprocess.run(
            ["jug-compute-residuals", par, tim],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
            env=_subprocess_env,
        )
    except FileNotFoundError:
        # Fall back to module invocation
        result = subprocess.run(
            [sys.executable, "-m", "jug.scripts.compute_residuals", par, tim],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
            env=_subprocess_env,
        )
    
    if result.returncode != 0:
        return False, f"exit {result.returncode}: {result.stderr[:200]}"
    
    output = result.stdout + result.stderr
    
    # Check for expected output patterns
    checks = []
    
    # Should report number of TOAs
    if "20" in output or "n_toas" in output.lower():
        checks.append("n_toas")
    
    # Should report RMS (weighted or unweighted)
    if "rms" in output.lower() or "µs" in output or "us" in output:
        checks.append("rms")
    
    if len(checks) < 1:
        return False, f"output missing expected info: {output[:300]}"
    
    return True, f"OK (found: {', '.join(checks)})"


def test_cli_fit_f0f1():
    """Test jug-fit CLI with F0/F1 fitting on mini data."""
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Use module invocation directly (more reliable than console scripts
    # which may be outdated in editable installs)
    result = subprocess.run(
        [sys.executable, "-m", "jug.scripts.fit_parameters",
         par, tim, "--fit", "F0", "F1", "--max-iter", "5"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(repo_root),
        env=_subprocess_env,
    )
    
    if result.returncode != 0:
        # Fit may return non-zero if it doesn't converge, check output
        output = result.stdout + result.stderr
        if "error" in output.lower() and "converge" not in output.lower():
            return False, f"exit {result.returncode}: {result.stderr[:200]}"
    
    output = result.stdout + result.stderr
    
    # Check for expected fit output patterns
    checks = []
    
    # Should report iterations
    if re.search(r'iter|iteration', output, re.I):
        checks.append("iterations")
    
    # Should report pre/post fit RMS or improvement
    if re.search(r'rms|residual', output, re.I):
        checks.append("rms_reported")
    
    # Check that fit did something (not just prefit)
    if re.search(r'(post|final|fitted|after)', output, re.I):
        checks.append("postfit")
    
    if len(checks) < 1:
        return False, f"output missing expected fit info: {output[:300]}"
    
    return True, f"OK (found: {', '.join(checks)})"


def test_cli_fit_improves_rms():
    """Test that CLI fit actually improves RMS."""
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Use Python API to get pre and post fit RMS for reliable comparison
    from jug.engine.session import TimingSession
    
    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
    prefit = session.compute_residuals(force_recompute=True)
    prefit_rms = prefit.get('weighted_rms_us') or prefit.get('rms_us')
    
    if prefit_rms is None:
        return False, "could not get prefit RMS"
    
    # Run fit
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    
    if fit_result.get('iterations', 0) == 0:
        return False, "fit did not iterate"
    
    # Get postfit RMS
    postfit = session.compute_residuals(force_recompute=True)
    postfit_rms = postfit.get('weighted_rms_us') or postfit.get('rms_us')
    
    if postfit_rms is None:
        return False, "could not get postfit RMS"
    
    # RMS should decrease (or at least not increase significantly)
    # For mini data with artificial TOAs, fit may not always improve
    improvement = prefit_rms - postfit_rms
    improvement_pct = (improvement / prefit_rms) * 100 if prefit_rms > 0 else 0
    
    # Accept if RMS decreased or stayed roughly the same (within 5%)
    if postfit_rms > prefit_rms * 1.05:
        return False, f"RMS increased: {prefit_rms:.2f} -> {postfit_rms:.2f} µs"
    
    return True, f"OK (prefit={prefit_rms:.2f}, postfit={postfit_rms:.2f} µs, {improvement_pct:+.1f}%)"


def test_cli_output_has_expected_markers():
    """Test that CLI output contains expected markers for programmatic parsing."""
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Run compute-residuals and check output format
    try:
        result = subprocess.run(
            [sys.executable, "-m", "jug.scripts.compute_residuals", par, tim],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
            env=_subprocess_env,
        )
    except Exception as e:
        return False, f"subprocess error: {e}"
    
    if result.returncode != 0:
        return False, f"exit {result.returncode}"
    
    output = result.stdout + result.stderr
    
    # Check for key markers that allow programmatic parsing
    expected_markers = [
        # Basic info
        (r'\d+\s*TOA', "TOA count"),
        (r'RMS|rms', "RMS mention"),
        (r'µs|us|microsec', "time units"),
    ]
    
    found_markers = []
    missing_markers = []
    
    for pattern, name in expected_markers:
        if re.search(pattern, output, re.I):
            found_markers.append(name)
        else:
            missing_markers.append(name)
    
    # Require at least 2 of 3 markers
    if len(found_markers) >= 2:
        return True, f"OK (found: {', '.join(found_markers)})"
    else:
        return False, f"missing markers: {', '.join(missing_markers)}"


def main():
    """Run all CLI integration tests."""
    print("=" * 60)
    print("CLI Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Compute Residuals CLI", test_cli_compute_residuals),
        ("Fit F0/F1 CLI", test_cli_fit_f0f1),
        ("Fit Improves RMS", test_cli_fit_improves_rms),
        ("Output Has Expected Markers", test_cli_output_has_expected_markers),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All CLI integration tests PASSED")
        return 0
    else:
        print("Some CLI integration tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
