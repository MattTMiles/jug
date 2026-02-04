#!/usr/bin/env python
"""
Benchmark script to compare JUG performance before/after optimizations.

Usage:
    # Test current version
    python benchmark_perf.py

    # Compare with a specific commit
    python benchmark_perf.py --compare 78fc692

    # Test specific par/tim files
    python benchmark_perf.py --par data/pulsars/J1909-3744.par --tim data/toas/J1909-3744.tim
"""

import argparse
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path


def run_benchmark(par_file: str, tim_file: str, n_runs: int = 3) -> dict:
    """Run fitting benchmark and return timing results."""
    import numpy as np

    results = {
        'session_create': [],
        'residuals_compute': [],
        'fit_exact': [],
        'fit_fast': [],
        'n_toas': 0,
    }

    # Import here to measure import time impact
    t0 = time.perf_counter()
    from jug.engine.session import TimingSession
    import_time = time.perf_counter() - t0
    results['import_time'] = import_time

    for run in range(n_runs):
        # Session creation
        t0 = time.perf_counter()
        session = TimingSession(par_file, tim_file, verbose=False)
        results['session_create'].append(time.perf_counter() - t0)
        results['n_toas'] = session.get_toa_count()

        # Residuals computation
        t0 = time.perf_counter()
        res = session.compute_residuals(subtract_tzr=True)
        results['residuals_compute'].append(time.perf_counter() - t0)

        # Fit with EXACT solver
        t0 = time.perf_counter()
        try:
            fit_result = session.fit_parameters(
                fit_params=['F0', 'F1'],
                verbose=False,
                solver_mode='exact'
            )
            results['fit_exact'].append(time.perf_counter() - t0)
        except TypeError:
            # Old version without solver_mode
            fit_result = session.fit_parameters(
                fit_params=['F0', 'F1'],
                verbose=False
            )
            results['fit_exact'].append(time.perf_counter() - t0)

        # Recreate session for fair comparison
        session = TimingSession(par_file, tim_file, verbose=False)
        session.compute_residuals(subtract_tzr=False)

        # Fit with FAST solver (if available)
        t0 = time.perf_counter()
        try:
            fit_result = session.fit_parameters(
                fit_params=['F0', 'F1'],
                verbose=False,
                solver_mode='fast'
            )
            results['fit_fast'].append(time.perf_counter() - t0)
        except TypeError:
            # Old version without solver_mode
            results['fit_fast'].append(None)

    return results


def print_results(results: dict, label: str = ""):
    """Print benchmark results."""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS {label}")
    print(f"{'='*60}")
    print(f"TOAs: {results['n_toas']}")
    print(f"Import time: {results.get('import_time', 0)*1000:.1f} ms")
    print()

    for key in ['session_create', 'residuals_compute', 'fit_exact', 'fit_fast']:
        times = results.get(key, [])
        times = [t for t in times if t is not None]
        if times:
            mean = np.mean(times)
            std = np.std(times)
            print(f"{key:20s}: {mean*1000:8.1f} ± {std*1000:5.1f} ms")
        else:
            print(f"{key:20s}: N/A")

    print(f"{'='*60}")


def compare_commits(current_commit: str, compare_commit: str, par_file: str, tim_file: str):
    """Compare performance between two commits."""
    import numpy as np

    # Get current working directory
    repo_dir = Path(__file__).parent

    # Run benchmark on current commit
    print(f"\nBenchmarking CURRENT commit ({current_commit[:7]})...")
    results_current = run_benchmark(par_file, tim_file)
    print_results(results_current, f"(current: {current_commit[:7]})")

    # Stash any changes, checkout compare commit, run benchmark, restore
    print(f"\nBenchmarking COMPARE commit ({compare_commit[:7]})...")

    # Save current state
    subprocess.run(['git', 'stash', '-u'], cwd=repo_dir, capture_output=True)

    try:
        # Checkout compare commit
        subprocess.run(['git', 'checkout', compare_commit], cwd=repo_dir, capture_output=True)

        # Need to reimport modules from the old commit
        # This is tricky - we'll run a subprocess instead
        benchmark_code = f'''
import sys
sys.path.insert(0, "{repo_dir}")
import numpy as np
import time

results = {{
    'session_create': [],
    'residuals_compute': [],
    'fit_exact': [],
    'n_toas': 0,
}}

from jug.engine.session import TimingSession

for run in range(3):
    t0 = time.perf_counter()
    session = TimingSession("{par_file}", "{tim_file}", verbose=False)
    results['session_create'].append(time.perf_counter() - t0)
    results['n_toas'] = session.get_toa_count()

    t0 = time.perf_counter()
    res = session.compute_residuals(subtract_tzr=True)
    results['residuals_compute'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    fit_result = session.fit_parameters(fit_params=['F0', 'F1'], verbose=False)
    results['fit_exact'].append(time.perf_counter() - t0)

for key in ['session_create', 'residuals_compute', 'fit_exact']:
    times = results.get(key, [])
    if times:
        mean = np.mean(times)
        std = np.std(times)
        print(f"{{key:20s}}: {{mean*1000:8.1f}} +/- {{std*1000:5.1f}} ms")
print(f"n_toas: {{results['n_toas']}}")
'''
        result = subprocess.run(
            [sys.executable, '-c', benchmark_code],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )

        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS (compare: {compare_commit[:7]})")
        print(f"{'='*60}")
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr[:200]}")
        print(f"{'='*60}")

    finally:
        # Restore original state
        subprocess.run(['git', 'checkout', current_commit], cwd=repo_dir, capture_output=True)
        subprocess.run(['git', 'stash', 'pop'], cwd=repo_dir, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description='Benchmark JUG performance')
    parser.add_argument('--par', default='data/pulsars/J1909-3744_tdb.par',
                        help='Path to .par file')
    parser.add_argument('--tim', default='data/pulsars/J1909-3744.tim',
                        help='Path to .tim file')
    parser.add_argument('--compare', default=None,
                        help='Git commit to compare against (e.g., 78fc692)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of benchmark runs')
    args = parser.parse_args()

    # Check files exist
    par_path = Path(args.par)
    tim_path = Path(args.tim)

    if not par_path.exists():
        print(f"Error: Par file not found: {par_path}")
        sys.exit(1)
    if not tim_path.exists():
        print(f"Error: Tim file not found: {tim_path}")
        sys.exit(1)

    # Get current commit
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    current_commit = result.stdout.strip()

    if args.compare:
        compare_commits(current_commit, args.compare, str(par_path), str(tim_path))
    else:
        # Just run benchmark on current version
        print(f"Running benchmark on current commit ({current_commit[:7]})...")
        results = run_benchmark(str(par_path), str(tim_path), n_runs=args.runs)
        print_results(results, f"(commit: {current_commit[:7]})")

        # Show speedup if fast solver is available
        if results['fit_fast'] and results['fit_fast'][0] is not None:
            import numpy as np
            exact_mean = np.mean(results['fit_exact'])
            fast_mean = np.mean(results['fit_fast'])
            speedup = exact_mean / fast_mean
            print(f"\nFast solver speedup: {speedup:.2f}x")


if __name__ == '__main__':
    main()
