#!/usr/bin/env python3
"""
Test: Does chunk size affect piecewise method precision?

Hypothesis: Smaller chunks → smaller dt_epoch → less precision loss in float64

Test chunk sizes from large (500 days) to small (10 TOAs) and measure:
1. Max error vs longdouble baseline
2. Spreading ratio (drift indicator)
3. Performance

Test in both numpy/longdouble AND JAX float64 to see if JAX can work.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import time
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Enable float64 in JAX
jax.config.update("jax_enable_x64", True)

# Add jug to path
sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble

SECS_PER_DAY = 86400.0

print("="*80)
print("PIECEWISE CHUNK SIZE INVESTIGATION")
print("="*80)
print()
print("Hypothesis: Smaller chunks → smaller dt_epoch → better precision")
print()

# ============================================================================
# Load Data
# ============================================================================

par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

if not par_file.exists() or not tim_file.exists():
    print("ERROR: Data files not found!")
    sys.exit(1)

print("Loading J1909-3744 data...")
params = parse_par_file(par_file)
pepoch_mjd = float(get_longdouble(params, 'PEPOCH'))
f0 = float(params['F0'])
f1 = float(params['F1'])

result = compute_residuals_simple(par_file, tim_file, clock_dir="data/clock",
                                 subtract_tzr=False, verbose=False)
dt_sec = result['dt_sec']
tdb_mjd = result['tdb_mjd']
tzr_phase = result['tzr_phase']

n_toas = len(dt_sec)
time_span_years = (tdb_mjd.max() - tdb_mjd.min()) / 365.25

print(f"  N_TOAs: {n_toas}")
print(f"  Time span: {time_span_years:.2f} years")
print(f"  F0: {f0:.15f} Hz")
print(f"  F1: {f1:.6e} Hz/s")
print()


# ============================================================================
# BASELINE: Longdouble Single PEPOCH
# ============================================================================

print("="*80)
print("BASELINE: Longdouble Single PEPOCH")
print("="*80)

dt_sec_ld = np.array(dt_sec, dtype=np.longdouble)
f0_ld = np.longdouble(f0)
f1_ld = np.longdouble(f1)

phase_baseline_ld = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))
phase_baseline = np.array(phase_baseline_ld, dtype=np.float64)

print("Ground truth computed.")
print()


# ============================================================================
# PIECEWISE IMPLEMENTATIONS
# ============================================================================

def create_chunks_by_toa_count(tdb_mjd, toas_per_chunk):
    """Create chunks by TOA count."""
    n_toas = len(tdb_mjd)
    n_chunks = (n_toas + toas_per_chunk - 1) // toas_per_chunk

    chunks = []
    for i in range(n_chunks):
        start = i * toas_per_chunk
        end = min((i + 1) * toas_per_chunk, n_toas)
        indices = np.arange(start, end)

        chunk_mjd = tdb_mjd[indices]
        chunks.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(chunk_mjd)),
            'n_toas': len(indices)
        })

    return chunks


def create_chunks_by_days(tdb_mjd, days_per_chunk):
    """Create chunks by time duration."""
    t_min = tdb_mjd.min()
    t_max = tdb_mjd.max()
    n_chunks = max(1, int(np.ceil((t_max - t_min) / days_per_chunk)))

    chunks = []
    for i in range(n_chunks):
        seg_start = t_min + i * days_per_chunk
        seg_end = t_min + (i + 1) * days_per_chunk

        if i == n_chunks - 1:
            mask = (tdb_mjd >= seg_start) & (tdb_mjd <= seg_end)
        else:
            mask = (tdb_mjd >= seg_start) & (tdb_mjd < seg_end)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        chunk_mjd = tdb_mjd[indices]
        chunks.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(chunk_mjd)),
            'n_toas': len(indices)
        })

    return chunks


def compute_piecewise_numpy_longdouble(dt_sec, tdb_mjd, chunks, f0, f1, pepoch_mjd):
    """Piecewise method in numpy/longdouble."""
    n_toas = len(dt_sec)
    phase_piecewise = np.zeros(n_toas)

    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    for chunk in chunks:
        idx = chunk['indices']

        # Epoch offset
        dt_epoch = (chunk['local_pepoch_mjd'] - pepoch_mjd) * SECS_PER_DAY

        # Local F0
        f0_local = f0 + f1 * dt_epoch

        # Local time coordinates
        dt_local = dt_sec[idx] - dt_epoch

        # Phase in local coordinates
        phase_local = dt_local * (f0_local + dt_local * (f1 / 2.0))

        # Phase offset (use longdouble)
        dt_epoch_ld = np.longdouble(dt_epoch)
        phase_offset = float(f0_ld * dt_epoch_ld + (f1_ld / np.longdouble(2.0)) * dt_epoch_ld**2)

        # Corrected phase
        phase_corrected = phase_local + phase_offset

        phase_piecewise[idx] = phase_corrected

    return phase_piecewise


def compute_piecewise_jax(dt_sec, tdb_mjd, chunks, f0, f1, pepoch_mjd):
    """Piecewise method in JAX float64."""
    n_toas = len(dt_sec)
    phase_piecewise = np.zeros(n_toas)

    for chunk in chunks:
        idx = chunk['indices']

        # Epoch offset
        dt_epoch = (chunk['local_pepoch_mjd'] - pepoch_mjd) * SECS_PER_DAY

        # Local F0
        f0_local = f0 + f1 * dt_epoch

        # Local time coordinates
        dt_local = dt_sec[idx] - dt_epoch

        # Phase in local coordinates (float64)
        phase_local = dt_local * (f0_local + dt_local * (f1 / 2.0))

        # Phase offset (float64)
        phase_offset = f0 * dt_epoch + (f1 / 2.0) * dt_epoch**2

        # Corrected phase
        phase_corrected = phase_local + phase_offset

        phase_piecewise[idx] = phase_corrected

    return phase_piecewise


# ============================================================================
# TEST DIFFERENT CHUNK SIZES
# ============================================================================

print("="*80)
print("TESTING DIFFERENT CHUNK SIZES")
print("="*80)
print()

# Define chunk sizes to test
chunk_configs = [
    ("500 days", "days", 500),
    ("200 days", "days", 200),
    ("100 days", "days", 100),
    ("200 TOAs", "toas", 200),
    ("100 TOAs", "toas", 100),
    ("50 TOAs", "toas", 50),
    ("25 TOAs", "toas", 25),
    ("10 TOAs", "toas", 10),
]

results = []

for name, chunk_type, chunk_size in chunk_configs:
    print(f"Testing: {name}")
    print("-" * 60)

    # Create chunks
    if chunk_type == "days":
        chunks = create_chunks_by_days(tdb_mjd, chunk_size)
    else:
        chunks = create_chunks_by_toa_count(tdb_mjd, chunk_size)

    n_chunks = len(chunks)
    avg_toas_per_chunk = n_toas / n_chunks

    # Compute average dt_epoch
    avg_dt_epoch = np.mean([abs((c['local_pepoch_mjd'] - pepoch_mjd) * SECS_PER_DAY)
                            for c in chunks])

    print(f"  N_chunks: {n_chunks}")
    print(f"  Avg TOAs/chunk: {avg_toas_per_chunk:.1f}")
    print(f"  Avg |dt_epoch|: {avg_dt_epoch:.2e} s")

    # Test numpy/longdouble
    t0 = time.time()
    phase_numpy_ld = compute_piecewise_numpy_longdouble(dt_sec, tdb_mjd, chunks,
                                                         f0, f1, pepoch_mjd)
    t_numpy = time.time() - t0

    diff_numpy = phase_numpy_ld - phase_baseline
    max_err_numpy = np.max(np.abs(diff_numpy)) / f0 * 1e9

    print(f"  Numpy/LD: {max_err_numpy:.3f} ns max error ({t_numpy*1000:.2f} ms)")

    # Test JAX float64
    t0 = time.time()
    phase_jax = compute_piecewise_jax(dt_sec, tdb_mjd, chunks, f0, f1, pepoch_mjd)
    t_jax = time.time() - t0

    diff_jax = phase_jax - phase_baseline
    max_err_jax = np.max(np.abs(diff_jax)) / f0 * 1e9

    print(f"  JAX F64:  {max_err_jax:.3f} ns max error ({t_jax*1000:.2f} ms)")

    # Compute spreading ratio
    time_years = (tdb_mjd - pepoch_mjd) / 365.25
    n_bins = 10
    bin_edges = np.linspace(time_years.min(), time_years.max(), n_bins + 1)

    spreads_numpy = []
    spreads_jax = []

    for i in range(n_bins):
        mask = (time_years >= bin_edges[i]) & (time_years < bin_edges[i+1])
        if np.sum(mask) > 10:
            spreads_numpy.append(np.std(diff_numpy[mask] / f0 * 1e9))
            spreads_jax.append(np.std(diff_jax[mask] / f0 * 1e9))

    if len(spreads_numpy) > 1 and spreads_numpy[0] > 0:
        spread_ratio_numpy = spreads_numpy[-1] / spreads_numpy[0]
        spread_ratio_jax = spreads_jax[-1] / spreads_jax[0]
    else:
        spread_ratio_numpy = 1.0
        spread_ratio_jax = 1.0

    print(f"  Numpy/LD spreading: {spread_ratio_numpy:.2f}×")
    print(f"  JAX F64 spreading:  {spread_ratio_jax:.2f}×")
    print()

    results.append({
        'name': name,
        'chunk_type': chunk_type,
        'chunk_size': chunk_size,
        'n_chunks': n_chunks,
        'avg_toas_per_chunk': avg_toas_per_chunk,
        'avg_dt_epoch': avg_dt_epoch,
        'max_err_numpy': max_err_numpy,
        'max_err_jax': max_err_jax,
        'spread_ratio_numpy': spread_ratio_numpy,
        'spread_ratio_jax': spread_ratio_jax,
        't_numpy': t_numpy,
        't_jax': t_jax,
        'diff_numpy': diff_numpy,
        'diff_jax': diff_jax,
    })


# ============================================================================
# ANALYSIS PLOTS
# ============================================================================

print("="*80)
print("GENERATING ANALYSIS PLOTS")
print("="*80)
print()

time_years = (tdb_mjd - pepoch_mjd) / 365.25

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

# Top row: Error vs chunk size
ax1 = fig.add_subplot(gs[0, 0])
avg_dt_epochs = [r['avg_dt_epoch'] for r in results]
max_errs_numpy = [r['max_err_numpy'] for r in results]
max_errs_jax = [r['max_err_jax'] for r in results]

ax1.semilogy(avg_dt_epochs, max_errs_numpy, 'o-', linewidth=2, markersize=8,
             color='blue', label='Numpy/Longdouble')
ax1.semilogy(avg_dt_epochs, max_errs_jax, 's-', linewidth=2, markersize=8,
             color='red', label='JAX Float64')
ax1.set_xlabel('Average |dt_epoch| (seconds)', fontsize=10)
ax1.set_ylabel('Max Error (ns)', fontsize=10)
ax1.set_title('Precision vs Chunk Size', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='1 ns target')

# Top row middle: Spreading ratio
ax2 = fig.add_subplot(gs[0, 1])
spread_ratios_numpy = [r['spread_ratio_numpy'] for r in results]
spread_ratios_jax = [r['spread_ratio_jax'] for r in results]

ax2.plot(avg_dt_epochs, spread_ratios_numpy, 'o-', linewidth=2, markersize=8,
         color='blue', label='Numpy/Longdouble')
ax2.plot(avg_dt_epochs, spread_ratios_jax, 's-', linewidth=2, markersize=8,
         color='red', label='JAX Float64')
ax2.set_xlabel('Average |dt_epoch| (seconds)', fontsize=10)
ax2.set_ylabel('Spreading Ratio (late/early)', fontsize=10)
ax2.set_title('Drift vs Chunk Size', fontsize=11, fontweight='bold')
ax2.axhline(1.5, color='orange', linestyle='--', alpha=0.5, label='1.5× threshold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Top row right: Performance
ax3 = fig.add_subplot(gs[0, 2])
t_numpy_all = [r['t_numpy']*1000 for r in results]
t_jax_all = [r['t_jax']*1000 for r in results]

ax3.plot(avg_dt_epochs, t_numpy_all, 'o-', linewidth=2, markersize=8,
         color='blue', label='Numpy/Longdouble')
ax3.plot(avg_dt_epochs, t_jax_all, 's-', linewidth=2, markersize=8,
         color='red', label='JAX Float64')
ax3.set_xlabel('Average |dt_epoch| (seconds)', fontsize=10)
ax3.set_ylabel('Time (ms)', fontsize=10)
ax3.set_title('Performance vs Chunk Size', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Remaining rows: Individual residual difference plots for selected chunk sizes
selected_indices = [0, 2, 4, 6, 7]  # 500 days, 100 days, 100 TOAs, 25 TOAs, 10 TOAs
plot_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

for idx, pos in zip(selected_indices, plot_positions):
    if idx >= len(results):
        continue

    r = results[idx]
    ax = fig.add_subplot(gs[pos[0], pos[1]])

    diff_ns = r['diff_numpy'] / f0 * 1e9
    ax.plot(time_years, diff_ns, '.', markersize=1, alpha=0.5, color='blue')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Difference (ns)', fontsize=9)
    ax.set_xlabel('Years from PEPOCH', fontsize=9)
    ax.set_title(f"{r['name']} - Numpy/LD\n{r['n_chunks']} chunks, max={r['max_err_numpy']:.3f} ns",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

# Bottom row: JAX versions of selected
plot_positions_jax = [(2, 2), (3, 0), (3, 1), (3, 2)]
for idx, pos in zip(selected_indices[1:5], plot_positions_jax):
    if idx >= len(results):
        continue

    r = results[idx]
    ax = fig.add_subplot(gs[pos[0], pos[1]])

    diff_ns = r['diff_jax'] / f0 * 1e9
    ax.plot(time_years, diff_ns, '.', markersize=1, alpha=0.5, color='red')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Difference (ns)', fontsize=9)
    ax.set_xlabel('Years from PEPOCH', fontsize=9)
    ax.set_title(f"{r['name']} - JAX F64\n{r['n_chunks']} chunks, max={r['max_err_jax']:.3f} ns",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

plt.savefig('piecewise_chunk_size_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: piecewise_chunk_size_analysis.png")
print()


# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("="*80)
print("SUMMARY TABLE")
print("="*80)
print()

print(f"{'Config':<15} {'N_chunks':<10} {'dt_epoch':<12} {'Numpy/LD':<12} {'JAX F64':<12} {'LD Spread':<12} {'JAX Spread':<12}")
print("-" * 110)

for r in results:
    print(f"{r['name']:<15} {r['n_chunks']:<10} {r['avg_dt_epoch']:>11.2e} "
          f"{r['max_err_numpy']:>10.3f} ns {r['max_err_jax']:>10.3f} ns "
          f"{r['spread_ratio_numpy']:>10.2f}× {r['spread_ratio_jax']:>10.2f}×")

print()


# ============================================================================
# CONCLUSIONS
# ============================================================================

print("="*80)
print("CONCLUSIONS")
print("="*80)
print()

# Find best numpy/LD configuration
best_numpy = min(results, key=lambda r: r['max_err_numpy'])
print(f"Best Numpy/LD configuration: {best_numpy['name']}")
print(f"  Max error: {best_numpy['max_err_numpy']:.3f} ns")
print(f"  Spreading: {best_numpy['spread_ratio_numpy']:.2f}×")
print()

# Find best JAX configuration
best_jax = min(results, key=lambda r: r['max_err_jax'])
print(f"Best JAX F64 configuration: {best_jax['name']}")
print(f"  Max error: {best_jax['max_err_jax']:.3f} ns")
print(f"  Spreading: {best_jax['spread_ratio_jax']:.2f}×")
print()

# Test hypothesis
print("Hypothesis test: Does smaller chunk size → better precision?")
print()

# Check correlation
from scipy.stats import spearmanr
corr_numpy, p_numpy = spearmanr(avg_dt_epochs, max_errs_numpy)
corr_jax, p_jax = spearmanr(avg_dt_epochs, max_errs_jax)

print(f"  Numpy/LD: correlation(dt_epoch, error) = {corr_numpy:.3f} (p={p_numpy:.2e})")
print(f"  JAX F64:  correlation(dt_epoch, error) = {corr_jax:.3f} (p={p_jax:.2e})")
print()

if corr_numpy > 0.5 and p_numpy < 0.05:
    print("  ✓ CONFIRMED for Numpy/LD: Smaller chunks → better precision")
elif corr_numpy > 0.0:
    print("  ⚠️  WEAK CORRELATION for Numpy/LD")
else:
    print("  ✗ REJECTED for Numpy/LD: No correlation found")

if corr_jax > 0.5 and p_jax < 0.05:
    print("  ✓ CONFIRMED for JAX: Smaller chunks → better precision")
elif corr_jax > 0.0:
    print("  ⚠️  WEAK CORRELATION for JAX")
else:
    print("  ✗ REJECTED for JAX: No correlation found")

print()

# Check if JAX can achieve sub-ns precision
if best_jax['max_err_jax'] < 1.0:
    print("🎉 SUCCESS! JAX achieves sub-nanosecond precision!")
    print(f"   Configuration: {best_jax['name']}")
    print("   → Pure JAX fitting is VIABLE!")
elif best_jax['max_err_jax'] < 5.0:
    print("✓ GOOD: JAX achieves <5 ns precision")
    print(f"   Configuration: {best_jax['name']}")
    print("   → Pure JAX fitting may be acceptable")
elif best_jax['max_err_jax'] < 50.0:
    print("⚠️  MARGINAL: JAX achieves <50 ns precision")
    print("   May be acceptable for some science cases")
else:
    print("✗ INSUFFICIENT: JAX errors >50 ns")
    print("   Need alternative approach")

print()
print("="*80)
print("END OF TEST")
print("="*80)
