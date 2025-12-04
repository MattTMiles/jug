#!/usr/bin/env python
"""
Benchmark JUG vs PINT vs Tempo2 for residual computation.

Tests the speed of computing residuals without fitting on J1909-3744.
"""
import time
import subprocess
import numpy as np
import tempfile
from pathlib import Path

# Paths
par_path = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
tim_path = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("=" * 80)
print("PULSAR TIMING SOFTWARE BENCHMARK")
print("=" * 80)
print(f"\nTest Case: J1909-3744")
print(f"Dataset: 10,408 TOAs")
print(f"Binary: ELL1 (tight MSP binary)")
print()

# Count TOAs
with open(tim_path) as f:
    n_toas = sum(1 for line in f if line.strip() and not line.startswith(('C ', 'FORMAT', '#')))
print(f"Verified: {n_toas} TOAs in tim file\n")

# ============================================================================
# Benchmark 1: JUG
# ============================================================================
print("=" * 80)
print("BENCHMARK 1: JUG (JAX-based)")
print("=" * 80)

import jax
jax.config.update("jax_enable_x64", True)
from jug.residuals.simple_calculator import compute_residuals_simple

# Warmup (JIT compilation)
print("Warming up (JIT compilation)...")
_ = compute_residuals_simple(par_path, tim_path, clock_dir="data/clock", observatory="meerkat")

# Timed runs
n_runs = 5
jug_times = []
print(f"\nRunning {n_runs} timed iterations...")
for i in range(n_runs):
    start = time.time()
    result = compute_residuals_simple(par_path, tim_path, clock_dir="data/clock", observatory="meerkat")
    elapsed = time.time() - start
    jug_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f} seconds")

jug_mean = np.mean(jug_times)
jug_std = np.std(jug_times)
jug_rms = result['rms_us']

print(f"\nJUG Results:")
print(f"  Mean time: {jug_mean:.3f} ± {jug_std:.3f} seconds")
print(f"  Best time: {min(jug_times):.3f} seconds")
print(f"  RMS residual: {jug_rms:.3f} μs")

# ============================================================================
# Benchmark 2: PINT
# ============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 2: PINT")
print("=" * 80)

from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Warmup
print("Warming up...")
_ = get_TOAs(tim_path, planets=True, include_bipm=True, bipm_version="BIPM2024", ephem="DE440")
_ = get_model(par_path)

# Timed runs
pint_times = []
print(f"\nRunning {n_runs} timed iterations...")
for i in range(n_runs):
    start = time.time()
    pint_model = get_model(par_path)
    pint_toas = get_TOAs(tim_path, planets=True, include_bipm=True, bipm_version="BIPM2024", ephem="DE440")
    pint_residuals = Residuals(pint_toas, pint_model)
    elapsed = time.time() - start
    pint_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f} seconds")

pint_mean = np.mean(pint_times)
pint_std = np.std(pint_times)
pint_rms = np.sqrt(np.mean(pint_residuals.time_resids.to_value('us')**2))

print(f"\nPINT Results:")
print(f"  Mean time: {pint_mean:.3f} ± {pint_std:.3f} seconds")
print(f"  Best time: {min(pint_times):.3f} seconds")
print(f"  RMS residual: {pint_rms:.3f} μs")

# ============================================================================
# Benchmark 3: Tempo2
# ============================================================================
print("\n" + "=" * 80)
print("BENCHMARK 3: Tempo2")
print("=" * 80)

# Check if tempo2 is available
try:
    result = subprocess.run(['which', 'tempo2'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tempo2 not found in PATH. Skipping Tempo2 benchmark.")
        tempo2_available = False
    else:
        tempo2_available = True
        print(f"Found tempo2 at: {result.stdout.strip()}")
except Exception as e:
    print(f"Error checking for tempo2: {e}")
    tempo2_available = False

if tempo2_available:
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Timed runs
        tempo2_times = []
        print(f"\nRunning {n_runs} timed iterations...")
        for i in range(n_runs):
            start = time.time()
            result = subprocess.run(
                ['tempo2', '-f', par_path, tim_path, '-nofit'],
                capture_output=True,
                text=True,
                cwd=tmpdir
            )
            elapsed = time.time() - start
            tempo2_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f} seconds")
        
        tempo2_mean = np.mean(tempo2_times)
        tempo2_std = np.std(tempo2_times)
        
        # Extract RMS from last run output
        tempo2_rms = None
        for line in result.stdout.split('\n'):
            if 'RMS pre-fit residual' in line:
                try:
                    # Parse: "RMS pre-fit residual = 0.416 (us)"
                    parts = line.split('=')
                    if len(parts) >= 2:
                        rms_str = parts[1].split('(')[0].strip()
                        tempo2_rms = float(rms_str)
                        break
                except Exception as e:
                    print(f"  Warning: Could not parse RMS: {e}")
        
        print(f"\nTempo2 Results:")
        print(f"  Mean time: {tempo2_mean:.3f} ± {tempo2_std:.3f} seconds")
        print(f"  Best time: {min(tempo2_times):.3f} seconds")
        if tempo2_rms:
            print(f"  RMS residual: {tempo2_rms:.3f} μs")
        else:
            print(f"  RMS residual: (unable to parse from output)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

print(f"\n{'Software':<15} {'Mean Time (s)':<20} {'Best Time (s)':<20} {'RMS (μs)':<15}")
print("-" * 80)
print(f"{'JUG':<15} {jug_mean:>8.3f} ± {jug_std:<7.3f}  {min(jug_times):>8.3f}             {jug_rms:>8.3f}")
print(f"{'PINT':<15} {pint_mean:>8.3f} ± {pint_std:<7.3f}  {min(pint_times):>8.3f}             {pint_rms:>8.3f}")

if tempo2_available:
    tempo2_rms_str = f"{tempo2_rms:.3f}" if tempo2_rms else "N/A"
    print(f"{'Tempo2':<15} {tempo2_mean:>8.3f} ± {tempo2_std:<7.3f}  {min(tempo2_times):>8.3f}             {tempo2_rms_str:>8}")
    print("\n" + "=" * 80)
    print("SPEEDUP vs JUG")
    print("=" * 80)
    print(f"PINT:   {pint_mean/jug_mean:.2f}x slower")
    print(f"Tempo2: {tempo2_mean/jug_mean:.2f}x slower")
else:
    print("\n" + "=" * 80)
    print("SPEEDUP")
    print("=" * 80)
    print(f"JUG vs PINT: {pint_mean/jug_mean:.2f}x faster")

print("\n" + "=" * 80)
print("NOTES")
print("=" * 80)
print("- JUG times include JIT compilation warmup")
print("- PINT times include ephemeris and clock file loading")
print("- Tempo2 times include all file I/O and initialization")
print("- All measurements are for residual computation only (no fitting)")
print("- Test system: " + subprocess.run(['uname', '-s'], capture_output=True, text=True).stdout.strip())
print("=" * 80)
