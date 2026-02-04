#!/usr/bin/env python
"""
JUG Test Runner - One-command test execution.

Run from repo root:
    python tests/run_all.py           # Run all tests
    python tests/run_all.py --quick   # Run quick tests only (skip slow, no external data)
    python tests/run_all.py --full    # Run all tests including optional PINT validation
    python tests/run_all.py --no-gui  # Skip GUI tests (for headless CI)
    python tests/run_all.py -v        # Verbose output

This runner executes script-style tests in a sensible order and provides
a concise PASS/FAIL summary. It exits nonzero on any failure.

Test categories:
- CRITICAL: Must pass (failures are hard errors)
- STANDARD: Should pass (failures are reported)
- SLOW: Take longer to run (skipped with --quick)
- CLI: Command-line interface smoke tests
- API: Python API workflow tests
- GUI: GUI tests (require Qt, skip with --no-gui)
- CORRECTNESS: Golden reference validation
- DATA_REQUIRED: Need external data files (skipped with --quick)

Environment variables for CI:
    JUG_TEST_DATA_DIR=/path/to/data   # Base directory for test data
    JUG_TEST_J1713_PAR=/path/to/par   # Per-pulsar overrides
    JUG_TEST_J1713_TIM=/path/to/tim
    etc.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import os

# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestSpec:
    """Specification for a single test."""
    name: str
    script: str
    category: str = "standard"  # critical, standard, slow, cli, api, gui, correctness
    description: str = ""
    timeout: int = 120  # seconds
    requires_data: bool = False  # True if needs external data files
    requires_gui: bool = False   # True if needs Qt/display


# Tests in execution order
TESTS = [
    # Critical tests - must pass
    TestSpec(
        name="imports",
        script="__inline__:test_imports",
        category="critical",
        description="Verify core imports work",
    ),
    TestSpec(
        name="prebinary_cache",
        script="test_cache_prebinary_regression.py",
        category="critical",
        description="Regression: prebinary_delay_sec in cache path",
    ),
    TestSpec(
        name="ddk_not_implemented",
        script="test_ddk_not_implemented.py",
        category="critical",
        description="DDK raises NotImplementedError (no silent aliasing)",
    ),
    TestSpec(
        name="binary_model_overrides",
        script="test_binary_model_overrides.py",
        category="critical",
        description="Centralized DDK override helper unit tests",
    ),
    
    # CLI smoke tests
    TestSpec(
        name="cli_smoke",
        script="test_cli_smoke.py",
        category="cli",
        description="CLI entry points respond to --help",
    ),
    
    # CLI integration tests (uses bundled mini data)
    TestSpec(
        name="cli_integration",
        script="test_cli_integration.py",
        category="cli",
        description="CLI compute/fit end-to-end with mini data",
    ),
    
    # API tests (use bundled mini data, no external deps)
    TestSpec(
        name="api_workflow",
        script="test_api_workflow.py",
        category="api",
        description="Python API workflow with bundled data",
    ),
    
    # Correctness tests (use bundled mini data)
    TestSpec(
        name="correctness",
        script="test_correctness.py",
        category="correctness",
        description="Residuals match golden reference",
    ),
    
    # Fit correctness tests (use bundled mini data)
    TestSpec(
        name="fit_correctness",
        script="test_fit_correctness_mini.py",
        category="correctness",
        description="Fit reduces RMS, deterministic, finite params",
    ),
    
    # Invariant tests (use bundled mini data)
    TestSpec(
        name="invariants",
        script="test_invariants.py",
        category="correctness",
        description="Prebinary time, fit recovery, gradient sanity",
    ),
    
    # GUI tests (need Qt)
    TestSpec(
        name="gui_smoke",
        script="test_gui_smoke.py",
        category="gui",
        description="GUI initializes, computes, fits headless",
        requires_gui=True,
    ),
    
    # Standard tests (need external data)
    TestSpec(
        name="timescale_validation",
        script="test_timescale_validation.py",
        category="standard",
        description="Par file timescale (TDB/TCB) handling",
        requires_data=True,
    ),
    TestSpec(
        name="binary_patch",
        script="test_binary_patch.py",
        category="standard",
        description="Binary delay patch vs PINT",
        requires_data=True,
    ),
    TestSpec(
        name="astrometry_fitting",
        script="test_astrometry_fitting.py",
        category="standard",
        description="Astrometry parameter fitting",
        requires_data=True,
    ),
    
    # Slow tests
    TestSpec(
        name="j2241_fit",
        script="test_j2241_fit.py",
        category="slow",
        description="J2241-5236 FB parameter fitting",
        timeout=180,
        requires_data=True,
    ),
]


# =============================================================================
# Inline Tests (quick sanity checks)
# =============================================================================

def test_imports():
    """Test that core imports work."""
    errors = []
    
    # Core imports
    try:
        from jug.engine.session import TimingSession
    except ImportError as e:
        errors.append(f"jug.engine.session: {e}")
    
    try:
        from jug.residuals.simple_calculator import compute_residuals_simple
    except ImportError as e:
        errors.append(f"jug.residuals.simple_calculator: {e}")
    
    try:
        from jug.fitting.optimized_fitter import fit_parameters_optimized
    except ImportError as e:
        errors.append(f"jug.fitting.optimized_fitter: {e}")
    
    try:
        from jug.io.par_reader import parse_par_file
    except ImportError as e:
        errors.append(f"jug.io.par_reader: {e}")
    
    # GUI imports (optional - may fail if Qt not available)
    try:
        from jug.gui.main_window import MainWindow
    except ImportError as e:
        print(f"  (GUI import skipped: {e})")
    
    if errors:
        raise RuntimeError("Import failures:\n  " + "\n  ".join(errors))
    
    print("  ✓ All core imports successful")
    return True


def test_session_workflow():
    """Quick session + fitting workflow test."""
    try:
        from tests.test_paths import get_j1713_paths, files_exist
    except ImportError:
        from test_paths import get_j1713_paths, files_exist
    
    par, tim = get_j1713_paths()
    if not files_exist(par, tim):
        print("  SKIP: J1713 test data not available")
        return None  # Skip, not fail
    
    from jug.engine.session import TimingSession
    
    session = TimingSession(par, tim, verbose=False)
    result = session.compute_residuals(force_recompute=True)
    
    assert result['n_toas'] > 0, "No TOAs loaded"
    assert result['rms_us'] > 0, "RMS is zero"
    
    # Quick fit test
    fit = session.fit_parameters(['F0', 'F1'], verbose=False)
    assert fit['iterations'] > 0, "Fit did not iterate"
    
    print(f"  ✓ Session workflow: {result['n_toas']} TOAs, RMS={result['rms_us']:.3f} μs")
    return True


INLINE_TESTS = {
    'test_imports': test_imports,
    'test_session_workflow': test_session_workflow,
}


# =============================================================================
# Test Runner
# =============================================================================

@dataclass
class TestResult:
    """Result of running a test."""
    name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration: float
    message: str = ""


def run_inline_test(test_name: str, verbose: bool = False) -> TestResult:
    """Run an inline test function."""
    if test_name not in INLINE_TESTS:
        return TestResult(test_name, "ERROR", 0.0, f"Unknown inline test: {test_name}")
    
    start = time.time()
    try:
        result = INLINE_TESTS[test_name]()
        duration = time.time() - start
        
        if result is None:
            return TestResult(test_name, "SKIP", duration)
        return TestResult(test_name, "PASS", duration)
    except Exception as e:
        duration = time.time() - start
        return TestResult(test_name, "FAIL", duration, str(e))


def run_script_test(
    script: str,
    timeout: int = 120,
    verbose: bool = False
) -> TestResult:
    """Run a test script as a subprocess."""
    tests_dir = Path(__file__).parent
    script_path = tests_dir / script
    
    if not script_path.exists():
        return TestResult(script, "ERROR", 0.0, f"Script not found: {script_path}")
    
    start = time.time()
    try:
        # Run from repo root so imports work
        repo_root = tests_dir.parent
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONPATH': str(repo_root)},
        )
        duration = time.time() - start
        
        # Check for explicit skip marker (test decided to skip entirely)
        # Don't count "[SKIP]" in optional items as full skip
        if result.returncode == 0:
            return TestResult(script, "PASS", duration)
        elif "SKIP: " in result.stdout and "PASS" not in result.stdout:
            # Full test skip (e.g., "SKIP: data not available")
            return TestResult(script, "SKIP", duration, "Test data not available")
        else:
            # Get last few lines of output for error message
            output = result.stdout + result.stderr
            lines = output.strip().split('\n')
            msg = '\n'.join(lines[-5:]) if len(lines) > 5 else output
            return TestResult(script, "FAIL", duration, msg)
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return TestResult(script, "ERROR", duration, f"Timeout after {timeout}s")
    except Exception as e:
        duration = time.time() - start
        return TestResult(script, "ERROR", duration, str(e))


def run_test(spec: TestSpec, verbose: bool = False) -> TestResult:
    """Run a single test based on its spec."""
    if spec.script.startswith("__inline__:"):
        test_name = spec.script.split(":", 1)[1]
        result = run_inline_test(test_name, verbose)
    else:
        result = run_script_test(spec.script, spec.timeout, verbose)
    
    # Copy name from spec
    result.name = spec.name
    return result


def print_result(result: TestResult, verbose: bool = False):
    """Print a test result."""
    status_symbols = {
        "PASS": "✓",
        "FAIL": "✗",
        "SKIP": "○",
        "ERROR": "!",
    }
    symbol = status_symbols.get(result.status, "?")
    
    print(f"  {symbol} {result.name}: {result.status} ({result.duration:.1f}s)")
    
    if verbose and result.message:
        for line in result.message.split('\n'):
            print(f"      {line}")


def main():
    parser = argparse.ArgumentParser(
        description="JUG Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip slow tests and tests requiring external data"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run all tests including slow and optional PINT validation"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip GUI tests (for headless CI)"
    )
    parser.add_argument(
        "--data-required",
        action="store_true",
        help="Only run tests that require external data files"
    )
    parser.add_argument(
        "--pint",
        action="store_true",
        help="Include PINT cross-validation in correctness tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for failures"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available tests and exit"
    )
    parser.add_argument(
        "--category", "-c",
        choices=["critical", "standard", "slow", "cli", "api", "gui", "correctness"],
        help="Run only tests in this category"
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific tests to run (by name)"
    )
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("Available tests:")
        for spec in TESTS:
            markers = []
            if spec.category == "slow":
                markers.append("slow")
            if spec.category == "critical":
                markers.append("critical")
            if spec.requires_data:
                markers.append("data-required")
            if spec.requires_gui:
                markers.append("gui")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            print(f"  {spec.name} ({spec.category}){marker_str}: {spec.description}")
        return 0
    
    # Filter tests
    tests_to_run = TESTS.copy()
    
    # Filter by specific test names
    if args.tests:
        test_names = set(args.tests)
        tests_to_run = [t for t in tests_to_run if t.name in test_names]
        if not tests_to_run:
            print(f"ERROR: No matching tests found for: {args.tests}")
            return 1
    
    # Filter by category
    if args.category:
        tests_to_run = [t for t in tests_to_run if t.category == args.category]
    
    # Quick mode: skip slow and data-requiring tests
    if args.quick:
        tests_to_run = [t for t in tests_to_run if t.category != "slow" and not t.requires_data]
    
    # Skip GUI tests if requested
    if args.no_gui:
        tests_to_run = [t for t in tests_to_run if not t.requires_gui]
    
    # Only data-required tests
    if args.data_required:
        tests_to_run = [t for t in tests_to_run if t.requires_data]
    
    # Set PINT flag for correctness tests
    if args.pint:
        os.environ['JUG_TEST_PINT'] = '1'
    
    # Run tests
    print("=" * 60)
    print("JUG Test Runner")
    print("=" * 60)
    mode_info = []
    if args.quick:
        mode_info.append("quick")
    if args.no_gui:
        mode_info.append("no-gui")
    if args.pint:
        mode_info.append("+pint")
    mode_str = f" ({', '.join(mode_info)})" if mode_info else ""
    print(f"\nRunning {len(tests_to_run)} tests{mode_str}...")
    
    results: List[TestResult] = []
    start_time = time.time()
    
    for spec in tests_to_run:
        print(f"\n[{spec.name}] {spec.description}...")
        result = run_test(spec, args.verbose)
        results.append(result)
        print_result(result, args.verbose)
    
    total_time = time.time() - start_time
    
    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    errors = sum(1 for r in results if r.status == "ERROR")
    
    # Check for critical failures
    critical_failures = [
        r for r in results 
        if r.status in ("FAIL", "ERROR") 
        and any(t.name == r.name and t.category == "critical" for t in tests_to_run)
    ]
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
    print(f"Total time: {total_time:.1f}s")
    
    if critical_failures:
        print("\n⚠️  CRITICAL TEST FAILURES:")
        for r in critical_failures:
            print(f"  - {r.name}: {r.message[:100]}")
    
    if failed > 0 or errors > 0:
        print("\n" + "=" * 60)
        print("RESULT: FAIL")
        print("=" * 60)
        return 1
    else:
        print("\n" + "=" * 60)
        print("RESULT: PASS")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
