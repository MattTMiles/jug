#!/usr/bin/env python3
"""
Test if hybrid chunked method can be implemented in pure JAX.

Critical questions:
1. Can we implement chunking with JAX primitives (jax.lax.scan)?
2. Does JAX float64 maintain the precision we saw with longdouble?
3. Can it JIT compile?
4. Can we autodiff through it for derivatives?
5. Is it faster than numpy/longdouble?

If ALL answers are YES, we can use pure JAX fitting with longdouble-equivalent precision!
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
import time
from pathlib import Path
import sys

# Enable float64 in JAX
jax.config.update("jax_enable_x64", True)

# Add jug to path
sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble

print("="*80)
print("JAX HYBRID METHOD COMPATIBILITY TEST")
print("="*80)
print()

# ============================================================================
# Load Real Data for Testing
# ============================================================================

par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

if not par_file.exists() or not tim_file.exists():
    print("ERROR: Data files not found. Using synthetic data instead.")
    # Synthetic data
    n_toas = 10000
    dt_sec = np.linspace(-1e8, 1e8, n_toas)
    f0 = 339.315691919040830
    f1 = -1.614750e-15
    tzr_phase = 0.0
else:
    print("Loading J1909-3744 data...")
    params = parse_par_file(par_file)
    f0 = float(params['F0'])
    f1 = float(params['F1'])

    result = compute_residuals_simple(par_file, tim_file, clock_dir="data/clock",
                                     subtract_tzr=False, verbose=False)
    dt_sec = result['dt_sec']
    tzr_phase = result['tzr_phase']
    n_toas = len(dt_sec)

    print(f"  N_TOAs: {n_toas}")
    print(f"  F0: {f0:.15f} Hz")
    print(f"  F1: {f1:.6e} Hz/s")
    print()


# ============================================================================
# REFERENCE: Numpy/Longdouble Hybrid (Known to Work)
# ============================================================================

print("="*80)
print("REFERENCE: Numpy/Longdouble Hybrid")
print("="*80)

def compute_phase_hybrid_numpy(dt_sec_array, f0, f1, chunk_size=100):
    """Reference implementation using numpy/longdouble."""
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

        phase_hybrid[start:end] = phase_ref_ld + phase_local_ld

    return phase_hybrid

chunk_size = 100
print(f"Chunk size: {chunk_size} TOAs")

t0 = time.time()
phase_numpy_ld = compute_phase_hybrid_numpy(dt_sec, f0, f1, chunk_size=chunk_size)
t_numpy = time.time() - t0

phase_numpy = np.array(phase_numpy_ld, dtype=np.float64)
print(f"Time: {t_numpy*1000:.2f} ms")
print(f"Phase range: [{np.min(phase_numpy):.3e}, {np.max(phase_numpy):.3e}] cycles")
print()


# ============================================================================
# ATTEMPT 1: JAX with jax.lax.scan
# ============================================================================

print("="*80)
print("ATTEMPT 1: JAX with jax.lax.scan")
print("="*80)

def compute_phase_hybrid_jax_scan(dt_sec_jax, f0, f1, chunk_size=100):
    """
    JAX implementation using jax.lax.scan for chunking.

    Note: JAX doesn't support longdouble, only float64.
    Question: Is float64 precise enough when using small chunks?
    """
    n_toas = len(dt_sec_jax)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size

    # Pad to make evenly divisible by chunk_size
    n_padded = n_chunks * chunk_size
    dt_padded = jnp.concatenate([dt_sec_jax, jnp.zeros(n_padded - n_toas)])

    # Reshape into chunks: (n_chunks, chunk_size)
    dt_chunks = dt_padded.reshape(n_chunks, chunk_size)

    def process_chunk(carry, dt_chunk):
        """Process one chunk."""
        # Reference time (mean of chunk)
        t_ref = jnp.mean(dt_chunk)

        # Phase at reference
        phase_ref = f0 * t_ref + 0.5 * f1 * t_ref**2

        # Local deviations
        dt_local = dt_chunk - t_ref

        # Local phase
        phase_local = (f0 * dt_local +
                      f1 * t_ref * dt_local +
                      0.5 * f1 * dt_local**2)

        # Total phase for this chunk
        phase_chunk = phase_ref + phase_local

        return carry, phase_chunk

    # Scan over all chunks
    _, phase_chunks = jax.lax.scan(process_chunk, None, dt_chunks)

    # Flatten and trim padding
    phase_flat = phase_chunks.flatten()
    phase_trimmed = phase_flat[:n_toas]

    return phase_trimmed

try:
    print("Testing JAX scan implementation...")
    dt_jax = jnp.array(dt_sec, dtype=jnp.float64)

    # Test without JIT first
    print("  Without JIT...", end=" ")
    t0 = time.time()
    phase_jax_scan = compute_phase_hybrid_jax_scan(dt_jax, f0, f1, chunk_size=chunk_size)
    t_jax_nojit = time.time() - t0
    print(f"✓ {t_jax_nojit*1000:.2f} ms")

    # Test with JIT
    print("  With JIT...", end=" ")
    compute_phase_jit = jit(compute_phase_hybrid_jax_scan, static_argnums=(3,))
    t0 = time.time()
    phase_jax_scan_jit = compute_phase_jit(dt_jax, f0, f1, chunk_size)
    phase_jax_scan_jit.block_until_ready()  # Wait for GPU/async completion
    t_jax_jit_compile = time.time() - t0
    print(f"✓ {t_jax_jit_compile*1000:.2f} ms (includes compilation)")

    # Run again (compiled)
    t0 = time.time()
    phase_jax_scan_jit = compute_phase_jit(dt_jax, f0, f1, chunk_size)
    phase_jax_scan_jit.block_until_ready()
    t_jax_jit = time.time() - t0
    print(f"  With JIT (2nd run)... ✓ {t_jax_jit*1000:.2f} ms (cached)")

    # Compare to numpy reference
    phase_jax_np = np.array(phase_jax_scan_jit)
    diff_jax = phase_jax_np - phase_numpy

    print()
    print("Precision vs numpy/longdouble:")
    print(f"  Mean diff: {np.mean(diff_jax):.6e} cycles = {np.mean(diff_jax)/f0*1e9:.3f} ns")
    print(f"  Std diff:  {np.std(diff_jax):.6e} cycles = {np.std(diff_jax)/f0*1e9:.3f} ns")
    print(f"  Max diff:  {np.max(np.abs(diff_jax)):.6e} cycles = {np.max(np.abs(diff_jax))/f0*1e9:.3f} ns")

    jax_scan_works = True
    max_err_ns = np.max(np.abs(diff_jax)) / f0 * 1e9

    if max_err_ns < 1.0:
        print(f"  ✓ SUCCESS: Sub-nanosecond precision! ({max_err_ns:.3f} ns)")
    elif max_err_ns < 50.0:
        print(f"  ⚠️  ACCEPTABLE: {max_err_ns:.1f} ns precision (target: <50 ns)")
    else:
        print(f"  ✗ FAIL: {max_err_ns:.1f} ns precision (too large!)")
        jax_scan_works = False

    print()
    print(f"Performance:")
    print(f"  Numpy/longdouble: {t_numpy*1000:.2f} ms")
    print(f"  JAX (no JIT):     {t_jax_nojit*1000:.2f} ms ({t_numpy/t_jax_nojit:.2f}× vs numpy)")
    print(f"  JAX (JIT):        {t_jax_jit*1000:.2f} ms ({t_numpy/t_jax_jit:.1f}× speedup!)")

    print()

except Exception as e:
    print(f"✗ FAILED: {e}")
    jax_scan_works = False
    phase_jax_scan_jit = None
    print()


# ============================================================================
# ATTEMPT 2: Test Autodiff Capability
# ============================================================================

if jax_scan_works:
    print("="*80)
    print("ATTEMPT 2: Test Autodiff for Derivatives")
    print("="*80)

    try:
        print("Testing autodiff through hybrid method...")

        def residual_func(f0_param, f1_param):
            """Compute mean squared residual (for testing autodiff)."""
            phase = compute_phase_hybrid_jax_scan(dt_jax, f0_param, f1_param, chunk_size)
            phase_wrapped = phase - jnp.round(phase)
            residuals = phase_wrapped / f0_param
            return jnp.mean(residuals**2)

        # Compute gradient
        print("  Computing gradient...", end=" ")
        grad_func = jit(grad(residual_func, argnums=(0, 1)))

        t0 = time.time()
        grads = grad_func(f0, f1)
        grads[0].block_until_ready()
        t_grad = time.time() - t0

        print(f"✓ {t_grad*1000:.2f} ms")
        print(f"  d(residual²)/d(F0) = {grads[0]:.6e}")
        print(f"  d(residual²)/d(F1) = {grads[1]:.6e}")

        # Test if gradients are reasonable (non-zero, finite)
        if np.isfinite(grads[0]) and np.isfinite(grads[1]) and (grads[0] != 0 or grads[1] != 0):
            print("  ✓ Gradients are finite and non-zero")
            autodiff_works = True
        else:
            print("  ✗ Gradients are zero or non-finite")
            autodiff_works = False

        print()

    except Exception as e:
        print(f"✗ FAILED: {e}")
        autodiff_works = False
        print()
else:
    autodiff_works = False


# ============================================================================
# ATTEMPT 3: Simpler JAX Implementation (Direct Loop)
# ============================================================================

print("="*80)
print("ATTEMPT 3: Simpler JAX Implementation (for comparison)")
print("="*80)

def compute_phase_jax_simple(dt_sec_jax, f0, f1):
    """
    Simple JAX implementation without chunking.
    This is what we'd use if chunking doesn't help.
    """
    return dt_sec_jax * (f0 + dt_sec_jax * (f1 / 2.0))

try:
    print("Testing simple JAX implementation...")

    # Test with JIT
    compute_phase_simple_jit = jit(compute_phase_jax_simple)

    t0 = time.time()
    phase_jax_simple = compute_phase_simple_jit(dt_jax, f0, f1)
    phase_jax_simple.block_until_ready()
    t_simple = time.time() - t0

    print(f"  Time (JIT): {t_simple*1000:.2f} ms")

    # Compare to numpy reference
    phase_simple_np = np.array(phase_jax_simple)
    diff_simple = phase_simple_np - phase_numpy

    print()
    print("Precision vs numpy/longdouble:")
    print(f"  Max diff: {np.max(np.abs(diff_simple)):.6e} cycles = {np.max(np.abs(diff_simple))/f0*1e9:.3f} ns")

    max_err_simple_ns = np.max(np.abs(diff_simple)) / f0 * 1e9

    if max_err_simple_ns < 50.0:
        print(f"  ⚠️  Simple JAX: {max_err_simple_ns:.1f} ns (may degrade for longer baselines)")
    else:
        print(f"  ✗ Simple JAX: {max_err_simple_ns:.1f} ns (too large!)")

    print()

except Exception as e:
    print(f"✗ FAILED: {e}")
    print()


# ============================================================================
# COMPUTE LONGDOUBLE SINGLE PEPOCH BASELINE
# ============================================================================

print("="*80)
print("BASELINE: Longdouble Single PEPOCH")
print("="*80)

# Compute ground truth using single PEPOCH longdouble
dt_sec_ld = np.array(dt_sec, dtype=np.longdouble)
f0_ld = np.longdouble(f0)
f1_ld = np.longdouble(f1)

phase_baseline_ld = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))
phase_baseline = np.array(phase_baseline_ld, dtype=np.float64)

print(f"Time: (instant - single formula)")
print(f"Phase range: [{np.min(phase_baseline):.3e}, {np.max(phase_baseline):.3e}] cycles")
print()

# Compute differences from baseline for plotting
diff_numpy_hybrid = phase_numpy - phase_baseline
diff_jax_hybrid = phase_jax_np - phase_baseline if jax_scan_works else np.zeros_like(phase_baseline)
diff_jax_simple = phase_simple_np - phase_baseline


# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================

print("="*80)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*80)

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if par_file.exists():
        # Get MJD for x-axis
        tdb_mjd = result['tdb_mjd']
        pepoch_mjd = float(get_longdouble(params, 'PEPOCH'))
        time_years = (tdb_mjd - pepoch_mjd) / 365.25
        x_data = time_years
        x_label = 'Years from PEPOCH'
    else:
        # Synthetic data - use dt_sec
        x_data = dt_sec / (365.25 * 86400)  # Convert to years
        x_label = 'Time (years from reference)'

    # Convert phase differences to nanoseconds
    diff_numpy_ns = diff_numpy_hybrid / f0 * 1e9
    diff_jax_ns = diff_jax_hybrid / f0 * 1e9
    diff_simple_ns = diff_jax_simple / f0 * 1e9

    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Numpy/Longdouble Hybrid vs Baseline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_data, diff_numpy_ns, '.', markersize=1, alpha=0.6, color='blue')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Difference (ns)', fontsize=11)
    ax1.set_title('Numpy/Longdouble Hybrid - Longdouble Single PEPOCH', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Max: {np.max(np.abs(diff_numpy_ns)):.3f} ns\nStd: {np.std(diff_numpy_ns):.3f} ns\nMean: {np.mean(diff_numpy_ns):.3f} ns"
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    # Panel 2: JAX Hybrid vs Baseline
    ax2 = fig.add_subplot(gs[1, :])
    if jax_scan_works:
        ax2.plot(x_data, diff_jax_ns, '.', markersize=1, alpha=0.6, color='red')
        ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Difference (ns)', fontsize=11)
        ax2.set_title('JAX Float64 Hybrid - Longdouble Single PEPOCH', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        stats_text = f"Max: {np.max(np.abs(diff_jax_ns)):.3f} ns\nStd: {np.std(diff_jax_ns):.3f} ns\nMean: {np.mean(diff_jax_ns):.3f} ns"
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'JAX Hybrid Failed', transform=ax2.transAxes,
                ha='center', va='center', fontsize=16, color='red')

    # Panel 3: Simple JAX vs Baseline
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(x_data, diff_simple_ns, '.', markersize=1, alpha=0.6, color='green')
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Difference (ns)', fontsize=11)
    ax3.set_xlabel(x_label, fontsize=11)
    ax3.set_title('Simple JAX Float64 - Longdouble Single PEPOCH', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    stats_text = f"Max: {np.max(np.abs(diff_simple_ns)):.3f} ns\nStd: {np.std(diff_simple_ns):.3f} ns\nMean: {np.mean(diff_simple_ns):.3f} ns"
    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    # Panel 4 & 5: Time-dependent spreading analysis
    # Bin data by time and compute spread in each bin
    n_bins = 20
    bin_edges = np.linspace(np.min(x_data), np.max(x_data), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    spread_numpy = []
    spread_jax = []
    spread_simple = []

    for i in range(n_bins):
        mask = (x_data >= bin_edges[i]) & (x_data < bin_edges[i+1])
        if np.sum(mask) > 10:  # Need enough points
            spread_numpy.append(np.std(diff_numpy_ns[mask]))
            spread_jax.append(np.std(diff_jax_ns[mask]) if jax_scan_works else 0)
            spread_simple.append(np.std(diff_simple_ns[mask]))
        else:
            spread_numpy.append(np.nan)
            spread_jax.append(np.nan)
            spread_simple.append(np.nan)

    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(bin_centers, spread_numpy, 'o-', linewidth=2, markersize=6,
             color='blue', label='Numpy/LD Hybrid')
    if jax_scan_works:
        ax4.plot(bin_centers, spread_jax, 's-', linewidth=2, markersize=6,
                 color='red', label='JAX Hybrid')
    ax4.plot(bin_centers, spread_simple, '^-', linewidth=2, markersize=6,
             color='green', label='Simple JAX')
    ax4.set_xlabel(x_label, fontsize=11)
    ax4.set_ylabel('Spread σ (ns)', fontsize=11)
    ax4.set_title('Time-Dependent Spreading', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 6: Histogram of errors
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(diff_numpy_ns, bins=50, alpha=0.5, color='blue',
             label=f'Numpy/LD ({np.std(diff_numpy_ns):.3f} ns σ)')
    if jax_scan_works:
        ax5.hist(diff_jax_ns, bins=50, alpha=0.5, color='red',
                 label=f'JAX Hybrid ({np.std(diff_jax_ns):.3f} ns σ)')
    ax5.hist(diff_simple_ns, bins=50, alpha=0.5, color='green',
             label=f'Simple JAX ({np.std(diff_simple_ns):.3f} ns σ)')
    ax5.set_xlabel('Difference from Baseline (ns)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_yscale('log')

    plt.savefig('hybrid_jax_diagnostics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: hybrid_jax_diagnostics.png")
    print()

except Exception as e:
    print(f"✗ Failed to generate plots: {e}")
    print()


# ============================================================================
# FINAL VERDICT
# ============================================================================

print("="*80)
print("FINAL VERDICT")
print("="*80)
print()

print("Success Criteria:")
print(f"  [{'✓' if jax_scan_works else '✗'}] JAX scan implementation works")
print(f"  [{'✓' if jax_scan_works and max_err_ns < 1.0 else '✗'}] Precision < 1 ns (hybrid)")
print(f"  [{'✓' if autodiff_works else '✗'}] Autodiff works through hybrid method")
print(f"  [{'✓' if jax_scan_works and t_jax_jit < t_numpy else '✗'}] Faster than numpy/longdouble")
print()

if jax_scan_works and autodiff_works and max_err_ns < 1.0:
    print("🎉 SUCCESS! Pure JAX hybrid method is VIABLE!")
    print()
    print("What this means:")
    print("  ✅ Can replace longdouble single PEPOCH with JAX hybrid")
    print("  ✅ Maintain sub-nanosecond precision")
    print("  ✅ Enable JAX autodiff for derivatives (no manual implementation)")
    print(f"  ✅ Speedup: {t_numpy/t_jax_jit:.1f}× faster than numpy/longdouble")
    print("  ✅ Opens door to GPU acceleration")
    print()
    print("Next steps:")
    print("  1. Integrate into optimized_fitter.py")
    print("  2. Test fitting convergence")
    print("  3. Benchmark on multiple pulsars")
    print("  4. Replace longdouble mode in production")

elif jax_scan_works and max_err_ns < 50.0:
    print("⚠️  PARTIAL SUCCESS - JAX hybrid works but precision degraded")
    print()
    print(f"Precision: {max_err_ns:.1f} ns (target: <1 ns)")
    print()
    print("Options:")
    print("  A. Use numpy/longdouble hybrid (0.022 ns proven)")
    print("  B. Investigate why JAX float64 loses precision vs longdouble")
    print("  C. Try smaller chunk sizes")
    print("  D. Keep current longdouble single PEPOCH")

else:
    print("✗ FAILED - JAX hybrid not viable in current form")
    print()
    print("Issues detected:")
    if not jax_scan_works:
        print("  - JAX scan implementation failed or has large errors")
    if not autodiff_works:
        print("  - Autodiff doesn't work through the hybrid method")
    print()
    print("Fallback options:")
    print("  A. Use numpy/longdouble hybrid (0.022 ns verified)")
    print("     - Still better than single PEPOCH for conditioning")
    print("     - No JAX speedup, but still valid")
    print()
    print("  B. Keep current longdouble single PEPOCH")
    print("     - Already works well")
    print("     - Only 5% slower than float64")
    print()
    print("  C. Investigate and fix JAX implementation")
    print("     - May be possible with different approach")

print()
print("="*80)
print("END OF TEST")
print("="*80)
