#!/usr/bin/env python3
"""
Test NEW Hybrid method with per-chunk wrapping in both JAX and numpy/longdouble.

Key insight from notebook: Wrap AFTER each chunk to keep values small!
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
print("NEW HYBRID METHOD TEST - WITH PER-CHUNK WRAPPING")
print("="*80)
print()

# ============================================================================
# Load Real Data
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

t0 = time.time()
phase_baseline_ld = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))
phase_baseline = np.array(phase_baseline_ld, dtype=np.float64)
t_baseline = time.time() - t0

print(f"Time: {t_baseline*1000:.2f} ms")
print()


# ============================================================================
# METHOD 1: Original Numpy/Longdouble Hybrid (NO per-chunk wrapping)
# ============================================================================

print("="*80)
print("METHOD 1: Original Numpy/Longdouble Hybrid (no per-chunk wrap)")
print("="*80)

def compute_phase_hybrid_numpy_old(dt_sec_array, f0, f1, chunk_size=100):
    """Original implementation - no per-chunk wrapping."""
    n_toas = len(dt_sec_array)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    phase_hybrid = np.zeros(n_toas, dtype=np.longdouble)

    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_toas)

        dt_chunk_ld = np.array(dt_sec_array[start:end], dtype=np.longdouble)
        t_ref_ld = np.mean(dt_chunk_ld)

        phase_ref_ld = f0_ld * t_ref_ld + np.longdouble(0.5) * f1_ld * t_ref_ld**2
        dt_local_ld = dt_chunk_ld - t_ref_ld
        phase_local_ld = (f0_ld * dt_local_ld +
                          f1_ld * t_ref_ld * dt_local_ld +
                          np.longdouble(0.5) * f1_ld * dt_local_ld**2)

        phase_hybrid[start:end] = phase_ref_ld + phase_local_ld  # No wrapping!

    return phase_hybrid

chunk_size = 100
print(f"Chunk size: {chunk_size} TOAs")

t0 = time.time()
phase_numpy_old_ld = compute_phase_hybrid_numpy_old(dt_sec, f0, f1, chunk_size)
phase_numpy_old = np.array(phase_numpy_old_ld, dtype=np.float64)
t_numpy_old = time.time() - t0

print(f"Time: {t_numpy_old*1000:.2f} ms")
diff_numpy_old = phase_numpy_old - phase_baseline
print(f"Max error: {np.max(np.abs(diff_numpy_old))/f0*1e9:.3f} ns")
print()


# ============================================================================
# METHOD 2: NEW Numpy/Longdouble Hybrid (WITH per-chunk wrapping!)
# ============================================================================

print("="*80)
print("METHOD 2: NEW Numpy/Longdouble Hybrid (WITH per-chunk wrapping)")
print("="*80)

def compute_phase_hybrid_numpy_NEW(dt_sec_array, f0, f1, chunk_size=100):
    """
    NEW implementation with per-chunk wrapping.

    Key: Wrap phase after each chunk to keep values small!
    """
    n_toas = len(dt_sec_array)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    phase_wrapped = np.zeros(n_toas, dtype=np.longdouble)

    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_toas)

        dt_chunk_ld = np.array(dt_sec_array[start:end], dtype=np.longdouble)
        t_ref_ld = np.mean(dt_chunk_ld)

        # Compute phase at reference
        phase_ref_ld = f0_ld * t_ref_ld + np.longdouble(0.5) * f1_ld * t_ref_ld**2

        # Local deviations
        dt_local_ld = dt_chunk_ld - t_ref_ld
        phase_local_ld = (f0_ld * dt_local_ld +
                          f1_ld * t_ref_ld * dt_local_ld +
                          np.longdouble(0.5) * f1_ld * dt_local_ld**2)

        # Total phase for chunk
        phase_total_ld = phase_ref_ld + phase_local_ld

        # *** KEY: Wrap IN LONGDOUBLE per chunk! ***
        phase_wrapped[start:end] = phase_total_ld - np.round(phase_total_ld)

    return phase_wrapped

t0 = time.time()
phase_numpy_NEW_ld = compute_phase_hybrid_numpy_NEW(dt_sec, f0, f1, chunk_size)
t_numpy_NEW = time.time() - t0

# Unwrap for comparison to baseline (which is unwrapped)
# Actually, we need to be careful here - the baseline is unwrapped, but NEW method returns wrapped
# Let me reconsider - for fair comparison, we need unwrapped phase from baseline
# Actually, let's compute residuals instead

print(f"Time: {t_numpy_NEW*1000:.2f} ms")
print(f"Note: This returns WRAPPED phase (values in [-0.5, 0.5] cycles)")
print()


# ============================================================================
# METHOD 3: JAX Hybrid (NO per-chunk wrapping) - Original
# ============================================================================

print("="*80)
print("METHOD 3: JAX Hybrid (no per-chunk wrapping) - Original")
print("="*80)

def compute_phase_hybrid_jax_old(dt_sec_jax, f0, f1, chunk_size=100):
    """Original JAX - no per-chunk wrapping."""
    n_toas = len(dt_sec_jax)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    dt_padded = jnp.concatenate([dt_sec_jax, jnp.zeros(n_padded - n_toas)])
    dt_chunks = dt_padded.reshape(n_chunks, chunk_size)

    def process_chunk(carry, dt_chunk):
        t_ref = jnp.mean(dt_chunk)
        phase_ref = f0 * t_ref + 0.5 * f1 * t_ref**2
        dt_local = dt_chunk - t_ref
        phase_local = (f0 * dt_local +
                      f1 * t_ref * dt_local +
                      0.5 * f1 * dt_local**2)
        phase_chunk = phase_ref + phase_local  # No wrapping!
        return carry, phase_chunk

    _, phase_chunks = jax.lax.scan(process_chunk, None, dt_chunks)
    phase_flat = phase_chunks.flatten()
    return phase_flat[:n_toas]

dt_jax = jnp.array(dt_sec, dtype=jnp.float64)

compute_phase_jax_old_jit = jit(compute_phase_hybrid_jax_old, static_argnums=(3,))

# Warmup
_ = compute_phase_jax_old_jit(dt_jax, f0, f1, chunk_size)

t0 = time.time()
phase_jax_old = compute_phase_jax_old_jit(dt_jax, f0, f1, chunk_size)
phase_jax_old.block_until_ready()
t_jax_old = time.time() - t0

phase_jax_old_np = np.array(phase_jax_old)
diff_jax_old = phase_jax_old_np - phase_baseline

print(f"Time (JIT): {t_jax_old*1000:.2f} ms")
print(f"Max error: {np.max(np.abs(diff_jax_old))/f0*1e9:.3f} ns")
print()


# ============================================================================
# METHOD 4: NEW JAX Hybrid (WITH per-chunk wrapping!)
# ============================================================================

print("="*80)
print("METHOD 4: NEW JAX Hybrid (WITH per-chunk wrapping) - CRITICAL TEST")
print("="*80)

def compute_phase_hybrid_jax_NEW(dt_sec_jax, f0, f1, chunk_size=100):
    """
    NEW JAX implementation with per-chunk wrapping.

    Key: Wrap phase after each chunk to keep values small!
    This is the critical modification.
    """
    n_toas = len(dt_sec_jax)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    dt_padded = jnp.concatenate([dt_sec_jax, jnp.zeros(n_padded - n_toas)])
    dt_chunks = dt_padded.reshape(n_chunks, chunk_size)

    def process_chunk(carry, dt_chunk):
        t_ref = jnp.mean(dt_chunk)

        # Compute phase at reference
        phase_ref = f0 * t_ref + 0.5 * f1 * t_ref**2

        # Local deviations
        dt_local = dt_chunk - t_ref
        phase_local = (f0 * dt_local +
                      f1 * t_ref * dt_local +
                      0.5 * f1 * dt_local**2)

        # Total phase
        phase_total = phase_ref + phase_local

        # *** KEY: Wrap per chunk in float64! ***
        phase_wrapped = phase_total - jnp.round(phase_total)

        return carry, phase_wrapped

    _, phase_chunks = jax.lax.scan(process_chunk, None, dt_chunks)
    phase_flat = phase_chunks.flatten()
    return phase_flat[:n_toas]

compute_phase_jax_NEW_jit = jit(compute_phase_hybrid_jax_NEW, static_argnums=(3,))

# Warmup
_ = compute_phase_jax_NEW_jit(dt_jax, f0, f1, chunk_size)

t0 = time.time()
phase_jax_NEW = compute_phase_jax_NEW_jit(dt_jax, f0, f1, chunk_size)
phase_jax_NEW.block_until_ready()
t_jax_NEW = time.time() - t0

print(f"Time (JIT): {t_jax_NEW*1000:.2f} ms")
print(f"Note: This returns WRAPPED phase (values in [-0.5, 0.5] cycles)")
print()


# ============================================================================
# METHOD 5: Simple JAX (for comparison)
# ============================================================================

print("="*80)
print("METHOD 5: Simple JAX (no chunking)")
print("="*80)

def compute_phase_jax_simple(dt_sec_jax, f0, f1):
    return dt_sec_jax * (f0 + dt_sec_jax * (f1 / 2.0))

compute_phase_simple_jit = jit(compute_phase_jax_simple)

# Warmup
_ = compute_phase_simple_jit(dt_jax, f0, f1)

t0 = time.time()
phase_jax_simple = compute_phase_simple_jit(dt_jax, f0, f1)
phase_jax_simple.block_until_ready()
t_jax_simple = time.time() - t0

phase_jax_simple_np = np.array(phase_jax_simple)
diff_jax_simple = phase_jax_simple_np - phase_baseline

print(f"Time (JIT): {t_jax_simple*1000:.2f} ms")
print(f"Max error: {np.max(np.abs(diff_jax_simple))/f0*1e9:.3f} ns")
print()


# ============================================================================
# RESIDUAL COMPUTATION FOR WRAPPED METHODS
# ============================================================================

print("="*80)
print("COMPUTING RESIDUALS (for fair comparison)")
print("="*80)

# For NEW methods that return wrapped phase, compute residuals directly
# Baseline unwrapped → wrap → residuals
phase_baseline_wrapped = phase_baseline - np.round(phase_baseline)
residuals_baseline = phase_baseline_wrapped / f0

# NEW numpy/LD (already wrapped)
residuals_numpy_NEW = np.array(phase_numpy_NEW_ld / f0_ld, dtype=np.float64)

# NEW JAX (already wrapped)
residuals_jax_NEW = np.array(phase_jax_NEW / f0)

print(f"Baseline residuals: mean={np.mean(residuals_baseline)*1e6:.6f} μs")
print(f"NEW numpy residuals: mean={np.mean(residuals_numpy_NEW)*1e6:.6f} μs")
print(f"NEW JAX residuals: mean={np.mean(residuals_jax_NEW)*1e6:.6f} μs")
print()

# Compute differences in residuals
diff_numpy_NEW_ns = (residuals_numpy_NEW - residuals_baseline) / f0 * 1e9 * f0  # Wait this is wrong
# Actually the phase difference is what we want
# residuals = phase_wrapped / f0
# diff in residuals = diff in phase_wrapped / f0

# Let me recalculate - for NEW methods, compare wrapped phases
# Baseline wrapped phase
phase_baseline_wrapped = phase_baseline - np.round(phase_baseline)

# NEW numpy wrapped phase (already wrapped)
phase_numpy_NEW = np.array(phase_numpy_NEW_ld, dtype=np.float64)

# NEW JAX wrapped phase (already wrapped)
phase_jax_NEW_np = np.array(phase_jax_NEW)

# Differences in wrapped phase (cycles)
diff_numpy_NEW_wrapped = phase_numpy_NEW - phase_baseline_wrapped
diff_jax_NEW_wrapped = phase_jax_NEW_np - phase_baseline_wrapped

print("Wrapped phase differences:")
print(f"  NEW numpy vs baseline: max={np.max(np.abs(diff_numpy_NEW_wrapped))/f0*1e9:.3f} ns")
print(f"  NEW JAX vs baseline: max={np.max(np.abs(diff_jax_NEW_wrapped))/f0*1e9:.3f} ns")
print()


# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================

print("="*80)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*80)

time_years = (tdb_mjd - pepoch_mjd) / 365.25

# Convert all phase differences to nanoseconds
diff_numpy_old_ns = diff_numpy_old / f0 * 1e9
diff_numpy_NEW_ns = diff_numpy_NEW_wrapped / f0 * 1e9
diff_jax_old_ns = diff_jax_old / f0 * 1e9
diff_jax_NEW_ns = diff_jax_NEW_wrapped / f0 * 1e9
diff_jax_simple_ns = diff_jax_simple / f0 * 1e9

fig = plt.figure(figsize=(18, 16))
gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3)

methods = [
    ("Numpy/LD Hybrid (old)", diff_numpy_old_ns, 'blue', 0),
    ("Numpy/LD Hybrid (NEW)", diff_numpy_NEW_ns, 'darkblue', 1),
    ("JAX Hybrid (old)", diff_jax_old_ns, 'red', 2),
    ("JAX Hybrid (NEW)", diff_jax_NEW_ns, 'darkred', 3),
    ("Simple JAX", diff_jax_simple_ns, 'green', 4),
]

# Individual difference plots
for name, diff_ns, color, row in methods:
    ax = fig.add_subplot(gs[row, 0])
    ax.plot(time_years, diff_ns, '.', markersize=1, alpha=0.5, color=color)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Difference (ns)', fontsize=10)
    ax.set_title(f'{name} - Longdouble Single PEPOCH', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    stats_text = f"Max: {np.max(np.abs(diff_ns)):.3f} ns\nStd: {np.std(diff_ns):.3f} ns"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

# Time-dependent spreading analysis
ax_spread = fig.add_subplot(gs[:, 1])

n_bins = 20
bin_edges = np.linspace(np.min(time_years), np.max(time_years), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

for name, diff_ns, color, _ in methods:
    spreads = []
    for i in range(n_bins):
        mask = (time_years >= bin_edges[i]) & (time_years < bin_edges[i+1])
        if np.sum(mask) > 10:
            spreads.append(np.std(diff_ns[mask]))
        else:
            spreads.append(np.nan)

    ax_spread.plot(bin_centers, spreads, 'o-', linewidth=2, markersize=6,
                   color=color, label=name, alpha=0.7)

ax_spread.set_xlabel('Years from PEPOCH', fontsize=11)
ax_spread.set_ylabel('Spread σ (ns)', fontsize=11)
ax_spread.set_title('Time-Dependent Spreading - ALL METHODS', fontsize=12, fontweight='bold')
ax_spread.legend(fontsize=9)
ax_spread.grid(True, alpha=0.3)
ax_spread.axhline(0, color='k', linestyle='--', alpha=0.3)

plt.savefig('hybrid_NEW_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hybrid_NEW_comparison.png")
print()


# ============================================================================
# SPREADING ANALYSIS
# ============================================================================

print("="*80)
print("SPREADING ANALYSIS")
print("="*80)
print()

for name, diff_ns, color, _ in methods:
    spreads = []
    for i in range(n_bins):
        mask = (time_years >= bin_edges[i]) & (time_years < bin_edges[i+1])
        if np.sum(mask) > 10:
            spreads.append(np.std(diff_ns[mask]))

    if len(spreads) > 1:
        spread_ratio = spreads[-1] / spreads[0]
        print(f"{name}:")
        print(f"  Early spread: {spreads[0]:.3f} ns")
        print(f"  Late spread: {spreads[-1]:.3f} ns")
        print(f"  Ratio: {spread_ratio:.2f}×")

        if spread_ratio < 1.5:
            print(f"  ✓ NO DRIFT (ratio < 1.5)")
        else:
            print(f"  ⚠️  DRIFT DETECTED (ratio = {spread_ratio:.2f})")
        print()


# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("FINAL SUMMARY")
print("="*80)
print()

print(f"Performance:")
print(f"  Baseline (LD single):      {t_baseline*1000:.2f} ms")
print(f"  Numpy/LD Hybrid (old):     {t_numpy_old*1000:.2f} ms")
print(f"  Numpy/LD Hybrid (NEW):     {t_numpy_NEW*1000:.2f} ms")
print(f"  JAX Hybrid (old):          {t_jax_old*1000:.2f} ms")
print(f"  JAX Hybrid (NEW):          {t_jax_NEW*1000:.2f} ms")
print(f"  Simple JAX:                {t_jax_simple*1000:.2f} ms")
print()

print(f"Precision (max error from baseline):")
print(f"  Numpy/LD Hybrid (old):     {np.max(np.abs(diff_numpy_old_ns)):.3f} ns")
print(f"  Numpy/LD Hybrid (NEW):     {np.max(np.abs(diff_numpy_NEW_ns)):.3f} ns")
print(f"  JAX Hybrid (old):          {np.max(np.abs(diff_jax_old_ns)):.3f} ns")
print(f"  JAX Hybrid (NEW):          {np.max(np.abs(diff_jax_NEW_ns)):.3f} ns")
print(f"  Simple JAX:                {np.max(np.abs(diff_jax_simple_ns)):.3f} ns")
print()

# Compute improvement from per-chunk wrapping
improvement_numpy = np.max(np.abs(diff_numpy_old_ns)) - np.max(np.abs(diff_numpy_NEW_ns))
improvement_jax = np.max(np.abs(diff_jax_old_ns)) - np.max(np.abs(diff_jax_NEW_ns))

print(f"Improvement from per-chunk wrapping:")
print(f"  Numpy/LD: {improvement_numpy:.3f} ns better ({100*improvement_numpy/np.max(np.abs(diff_numpy_old_ns)):.1f}%)")
print(f"  JAX:      {improvement_jax:.3f} ns better ({100*improvement_jax/np.max(np.abs(diff_jax_old_ns)):.1f}%)")
print()

# Final verdict
if np.max(np.abs(diff_jax_NEW_ns)) < 5.0:
    print("🎉 SUCCESS! NEW JAX Hybrid achieves <5 ns precision!")
    print("   Per-chunk wrapping WORKS in float64!")
elif np.max(np.abs(diff_jax_NEW_ns)) < 50.0:
    print("✓ GOOD: NEW JAX Hybrid achieves sub-50 ns precision")
else:
    print("⚠️  Per-chunk wrapping helps but precision still >50 ns")

print()
print("="*80)
print("END OF TEST")
print("="*80)
