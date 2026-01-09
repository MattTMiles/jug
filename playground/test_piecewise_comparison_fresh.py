#!/usr/bin/env python3
"""
Fresh comparison of three fitting methods for spin parameters.

Methods tested:
1. Longdouble single PEPOCH (current production method)
2. Piecewise with local PEPOCHs (500-day segments)
3. Hybrid chunked method (100-TOA chunks)

Goal: Determine if piecewise/hybrid can match longdouble precision.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add jug to path
sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble

SECS_PER_DAY = 86400.0

print("="*80)
print("FRESH PIECEWISE vs LONGDOUBLE vs HYBRID COMPARISON")
print("="*80)
print()

# ============================================================================
# Load Data
# ============================================================================

par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

if not par_file.exists() or not tim_file.exists():
    print(f"ERROR: Data files not found!")
    print(f"  Par file: {par_file} (exists: {par_file.exists()})")
    print(f"  Tim file: {tim_file} (exists: {tim_file.exists()})")
    sys.exit(1)

params = parse_par_file(par_file)
pepoch_mjd = float(get_longdouble(params, 'PEPOCH'))
f0 = float(params['F0'])
f1 = float(params['F1'])

print(f"Pulsar: J1909-3744")
print(f"  PEPOCH = {pepoch_mjd:.6f} MJD")
print(f"  F0 = {f0:.15f} Hz")
print(f"  F1 = {f1:.6e} Hz/s")
print()

# Compute residuals using simple_calculator (includes all delays)
result = compute_residuals_simple(
    par_file, tim_file,
    clock_dir="data/clock",
    subtract_tzr=False,
    verbose=False
)

dt_sec = result['dt_sec']  # Emission time from PEPOCH (includes all delays)
tdb_mjd = result['tdb_mjd']
tzr_phase = result['tzr_phase']

n_toas = len(dt_sec)
time_span_years = (tdb_mjd.max() - tdb_mjd.min()) / 365.25

print(f"Data loaded:")
print(f"  N_TOAs = {n_toas}")
print(f"  Time span = {time_span_years:.2f} years")
print(f"  MJD range: [{tdb_mjd.min():.2f}, {tdb_mjd.max():.2f}]")
print(f"  TZR phase = {tzr_phase:.6f} cycles")
print()


# ============================================================================
# METHOD 1: Longdouble Single PEPOCH (Current Production)
# ============================================================================

print("="*80)
print("METHOD 1: LONGDOUBLE SINGLE PEPOCH")
print("="*80)

dt_sec_ld = np.array(dt_sec, dtype=np.longdouble)
f0_ld = np.longdouble(f0)
f1_ld = np.longdouble(f1)

# Compute phase in longdouble
phase_ld = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))

# Subtract TZR
phase_ld_minus_tzr = phase_ld - np.longdouble(tzr_phase)

# Wrap and convert to residuals
phase_wrapped_ld = phase_ld_minus_tzr - np.round(phase_ld_minus_tzr)
residuals_ld_sec = np.array(phase_wrapped_ld / f0_ld, dtype=np.float64)
residuals_ld_us = residuals_ld_sec * 1e6

print(f"  Mean: {np.mean(residuals_ld_us):.6f} μs")
print(f"  RMS:  {np.std(residuals_ld_us):.6f} μs")
print(f"  Range: [{np.min(residuals_ld_us):.3f}, {np.max(residuals_ld_us):.3f}] μs")
print()


# ============================================================================
# METHOD 2: Piecewise with Local PEPOCHs
# ============================================================================

print("="*80)
print("METHOD 2: PIECEWISE WITH LOCAL PEPOCHs")
print("="*80)

def create_time_segments(tdb_mjd, segment_duration_days=500.0):
    """Divide TOAs into temporal segments."""
    t_min = np.min(tdb_mjd)
    t_max = np.max(tdb_mjd)
    n_segments = max(1, int(np.ceil((t_max - t_min) / segment_duration_days)))

    segments = []
    for i in range(n_segments):
        seg_start = t_min + i * segment_duration_days
        seg_end = t_min + (i + 1) * segment_duration_days

        mask = (tdb_mjd >= seg_start) & (tdb_mjd < seg_end)
        if i == n_segments - 1:
            mask = (tdb_mjd >= seg_start) & (tdb_mjd <= seg_end)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        seg_times = tdb_mjd[indices]
        segments.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(seg_times)),
            'n_toas': len(indices)
        })

    return segments

segments = create_time_segments(tdb_mjd, segment_duration_days=500.0)
print(f"  Created {len(segments)} segments")

# Compute piecewise phase
phase_piecewise = np.zeros(n_toas)

for i, seg in enumerate(segments):
    idx = seg['indices']

    # Epoch offset in seconds
    dt_epoch = (seg['local_pepoch_mjd'] - pepoch_mjd) * SECS_PER_DAY

    # Local F0 from continuity constraint
    f0_local = f0 + f1 * dt_epoch

    # Local time coordinates (smaller than global dt)
    dt_local = dt_sec[idx] - dt_epoch

    # Phase in local coordinates
    phase_local = dt_local * (f0_local + dt_local * (f1 / 2.0))

    # Phase offset correction (use longdouble for precision)
    dt_epoch_ld = np.longdouble(dt_epoch)
    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)
    phase_offset = float(f0_ld * dt_epoch_ld + (f1_ld / np.longdouble(2.0)) * dt_epoch_ld**2)

    # Add offset to restore global phase
    phase_corrected = phase_local + phase_offset

    # Store
    phase_piecewise[idx] = phase_corrected

# Subtract TZR
phase_piecewise_minus_tzr = phase_piecewise - tzr_phase

# Wrap and convert to residuals (use global F0!)
phase_wrapped_pw = phase_piecewise_minus_tzr - np.round(phase_piecewise_minus_tzr)
residuals_pw_sec = phase_wrapped_pw / f0
residuals_pw_us = residuals_pw_sec * 1e6

print(f"  Mean: {np.mean(residuals_pw_us):.6f} μs")
print(f"  RMS:  {np.std(residuals_pw_us):.6f} μs")
print(f"  Range: [{np.min(residuals_pw_us):.3f}, {np.max(residuals_pw_us):.3f}] μs")
print()


# ============================================================================
# METHOD 3: Hybrid Chunked Method
# ============================================================================

print("="*80)
print("METHOD 3: HYBRID CHUNKED METHOD")
print("="*80)

def compute_phase_hybrid(dt_sec_array, f0, f1, chunk_size=100):
    """
    Compute phase using hybrid chunked method.

    For each chunk:
    - Use mean time as local reference
    - Compute phase at reference in longdouble
    - Compute local deviations in longdouble
    - Keep everything in longdouble throughout
    """
    n_toas = len(dt_sec_array)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size

    phase_hybrid = np.zeros(n_toas, dtype=np.longdouble)

    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_toas)

        # Get chunk in longdouble
        dt_chunk_ld = np.array(dt_sec_array[start:end], dtype=np.longdouble)

        # Reference time (mean of chunk)
        t_ref_ld = np.mean(dt_chunk_ld)

        # Phase at reference
        phase_ref_ld = f0_ld * t_ref_ld + np.longdouble(0.5) * f1_ld * t_ref_ld**2

        # Local deviations from reference
        dt_local_ld = dt_chunk_ld - t_ref_ld

        # Local phase (includes first-order F1 term)
        phase_local_ld = (f0_ld * dt_local_ld +
                          f1_ld * t_ref_ld * dt_local_ld +
                          np.longdouble(0.5) * f1_ld * dt_local_ld**2)

        # Total phase
        phase_hybrid[start:end] = phase_ref_ld + phase_local_ld

    return phase_hybrid

chunk_size = 100
print(f"  Chunk size = {chunk_size} TOAs")
print(f"  N_chunks = {(n_toas + chunk_size - 1) // chunk_size}")

phase_hybrid_ld = compute_phase_hybrid(dt_sec, f0, f1, chunk_size=chunk_size)

# Subtract TZR
phase_hybrid_minus_tzr = phase_hybrid_ld - np.longdouble(tzr_phase)

# Wrap and convert to residuals
phase_wrapped_hyb = phase_hybrid_minus_tzr - np.round(phase_hybrid_minus_tzr)
residuals_hyb_sec = np.array(phase_wrapped_hyb / f0_ld, dtype=np.float64)
residuals_hyb_us = residuals_hyb_sec * 1e6

print(f"  Mean: {np.mean(residuals_hyb_us):.6f} μs")
print(f"  RMS:  {np.std(residuals_hyb_us):.6f} μs")
print(f"  Range: [{np.min(residuals_hyb_us):.3f}, {np.max(residuals_hyb_us):.3f}] μs")
print()


# ============================================================================
# COMPARISON
# ============================================================================

print("="*80)
print("COMPARISON: Piecewise vs Longdouble")
print("="*80)

diff_pw_ld = residuals_pw_sec - residuals_ld_sec
diff_pw_ld_ns = diff_pw_ld * 1e9

print(f"Difference (Piecewise - Longdouble):")
print(f"  Mean: {np.mean(diff_pw_ld_ns):.3f} ns")
print(f"  Std:  {np.std(diff_pw_ld_ns):.3f} ns")
print(f"  Max:  {np.max(np.abs(diff_pw_ld_ns)):.3f} ns")
print(f"  Range: [{np.min(diff_pw_ld_ns):.3f}, {np.max(diff_pw_ld_ns):.3f}] ns")

# Check for time-dependent spreading
print(f"\nTime-dependent spreading analysis:")
t_norm = (tdb_mjd - tdb_mjd.min()) / (tdb_mjd.max() - tdb_mjd.min())
bins = np.linspace(0, 1, 11)
spreads_pw = []
for i in range(10):
    mask = (t_norm >= bins[i]) & (t_norm < bins[i+1])
    if np.sum(mask) > 0:
        spread = np.std(diff_pw_ld_ns[mask])
        spreads_pw.append(spread)
        time_pct = (bins[i] + bins[i+1]) / 2 * 100
        print(f"  {time_pct:4.0f}% through data: σ = {spread:6.2f} ns")

if len(spreads_pw) > 1:
    spread_ratio = spreads_pw[-1] / spreads_pw[0]
    print(f"\n  Spread ratio (late/early): {spread_ratio:.2f}×")
    if spread_ratio > 1.5:
        print(f"  ⚠️  DRIFT DETECTED: Spread increases {spread_ratio:.1f}× across timespan")
    else:
        print(f"  ✓ No significant drift")

print()

print("="*80)
print("COMPARISON: Hybrid vs Longdouble")
print("="*80)

diff_hyb_ld = residuals_hyb_sec - residuals_ld_sec
diff_hyb_ld_ns = diff_hyb_ld * 1e9

print(f"Difference (Hybrid - Longdouble):")
print(f"  Mean: {np.mean(diff_hyb_ld_ns):.3f} ns")
print(f"  Std:  {np.std(diff_hyb_ld_ns):.3f} ns")
print(f"  Max:  {np.max(np.abs(diff_hyb_ld_ns)):.3f} ns")
print(f"  Range: [{np.min(diff_hyb_ld_ns):.3f}, {np.max(diff_hyb_ld_ns):.3f}] ns")

# Check for time-dependent spreading
print(f"\nTime-dependent spreading analysis:")
spreads_hyb = []
for i in range(10):
    mask = (t_norm >= bins[i]) & (t_norm < bins[i+1])
    if np.sum(mask) > 0:
        spread = np.std(diff_hyb_ld_ns[mask])
        spreads_hyb.append(spread)
        time_pct = (bins[i] + bins[i+1]) / 2 * 100
        print(f"  {time_pct:4.0f}% through data: σ = {spread:6.2f} ns")

if len(spreads_hyb) > 1:
    spread_ratio = spreads_hyb[-1] / spreads_hyb[0]
    print(f"\n  Spread ratio (late/early): {spread_ratio:.2f}×")
    if spread_ratio > 1.5:
        print(f"  ⚠️  DRIFT DETECTED: Spread increases {spread_ratio:.1f}× across timespan")
    else:
        print(f"  ✓ No significant drift")

print()


# ============================================================================
# PLOTS
# ============================================================================

print("="*80)
print("GENERATING PLOTS")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Row 1: All three residual patterns
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tdb_mjd, residuals_ld_us, '.', markersize=1, alpha=0.5, color='blue')
ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Residual (μs)')
ax1.set_title('Method 1: Longdouble Single PEPOCH')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(tdb_mjd, residuals_pw_us, '.', markersize=1, alpha=0.5, color='red')
ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Residual (μs)')
ax2.set_title('Method 2: Piecewise (500-day segments)')
ax2.grid(True, alpha=0.3)
# Mark segment boundaries
for seg in segments[1:]:
    tmin = tdb_mjd[seg['indices'][0]]
    ax2.axvline(tmin, color='orange', linestyle=':', alpha=0.3, linewidth=1)

# Row 2: Hybrid and its difference
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(tdb_mjd, residuals_hyb_us, '.', markersize=1, alpha=0.5, color='green')
ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Residual (μs)')
ax3.set_title('Method 3: Hybrid Chunked (100-TOA chunks)')
ax3.grid(True, alpha=0.3)

# Row 3: Differences from longdouble
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(tdb_mjd, diff_pw_ld_ns, '.', markersize=2, alpha=0.5, color='red', label='Piecewise - Longdouble')
ax4.plot(tdb_mjd, diff_hyb_ld_ns, '.', markersize=2, alpha=0.5, color='green', label='Hybrid - Longdouble')
ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Difference (ns)')
ax4.set_title('Differences from Longdouble Single PEPOCH')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 4: Spreading over time
ax5 = fig.add_subplot(gs[3, 0])
bin_centers = [(bins[i] + bins[i+1]) / 2 * time_span_years for i in range(len(spreads_pw))]
ax5.plot(bin_centers, spreads_pw, 'o-', color='red', linewidth=2, markersize=8, label='Piecewise')
ax5.set_xlabel('Years from start')
ax5.set_ylabel('Spread (ns)')
ax5.set_title('Time-dependent Spreading: Piecewise')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[3, 1])
ax6.plot(bin_centers, spreads_hyb, 'o-', color='green', linewidth=2, markersize=8, label='Hybrid')
ax6.set_xlabel('Years from start')
ax6.set_ylabel('Spread (ns)')
ax6.set_title('Time-dependent Spreading: Hybrid')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.savefig('piecewise_comparison_fresh.png', dpi=150, bbox_inches='tight')
print(f"Saved: piecewise_comparison_fresh.png")
print()


# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("SUMMARY")
print("="*80)
print()

print("Precision comparison:")
print(f"  Longdouble:  BASELINE (ground truth)")
print(f"  Piecewise:   {np.max(np.abs(diff_pw_ld_ns)):.1f} ns max error, {spread_ratio:.2f}× spreading")
print(f"  Hybrid:      {np.max(np.abs(diff_hyb_ld_ns)):.1f} ns max error, {spreads_hyb[-1]/spreads_hyb[0]:.2f}× spreading")
print()

# Verdict
max_err_pw = np.max(np.abs(diff_pw_ld_ns))
max_err_hyb = np.max(np.abs(diff_hyb_ld_ns))

if max_err_pw < 1.0 and spread_ratio < 1.5:
    print("✓ PIECEWISE METHOD: Matches longdouble to sub-ns precision with no drift!")
elif max_err_pw < 50.0:
    print("⚠️  PIECEWISE METHOD: Achieves ~{:.0f} ns precision but shows drift".format(max_err_pw))
else:
    print("✗ PIECEWISE METHOD: Errors exceed 50 ns - needs investigation")

if max_err_hyb < 1.0 and spreads_hyb[-1]/spreads_hyb[0] < 1.5:
    print("✓ HYBRID METHOD: Matches longdouble to sub-ns precision with no drift!")
elif max_err_hyb < 50.0:
    print("⚠️  HYBRID METHOD: Achieves ~{:.0f} ns precision but shows drift".format(max_err_hyb))
else:
    print("✗ HYBRID METHOD: Errors exceed 50 ns - needs investigation")

print()
print("="*80)
print("DONE")
print("="*80)
