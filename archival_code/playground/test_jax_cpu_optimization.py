#!/usr/bin/env python3
"""
Test Priority #1 Optimization: Force JAX to use CPU
===================================================

Quick test of forcing JAX arrays to CPU device to avoid
expensive GPU memory transfers.

Expected result: 350ms → 1ms for array conversion
"""

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

N = 10408  # Same size as J1909-3744

print("="*80)
print("JAX CPU OPTIMIZATION TEST")
print("="*80)

# Create test data
print(f"\nCreating test array ({N} float64 values)...")
test_array = np.random.randn(N)

# =============================================================================
# BASELINE: Default JAX device
# =============================================================================
print("\n" + "="*80)
print("BASELINE: Default JAX Device")
print("="*80)

times_default = []
for i in range(5):
    start = time.time()
    jax_array = jnp.array(test_array)
    jax_array.block_until_ready()  # Force completion
    elapsed = time.time() - start
    times_default.append(elapsed)

mean_default = np.mean(times_default)
print(f"\nArray conversion time (default):")
print(f"  Mean: {mean_default*1000:.1f} ms")
print(f"  Min:  {np.min(times_default)*1000:.1f} ms")
print(f"  Max:  {np.max(times_default)*1000:.1f} ms")
print(f"  Device: {jax_array.device}")

# =============================================================================
# OPTIMIZED: Force CPU device
# =============================================================================
print("\n" + "="*80)
print("OPTIMIZED: Force CPU Device")
print("="*80)

times_cpu = []
with jax.default_device(jax.devices('cpu')[0]):
    for i in range(5):
        start = time.time()
        jax_array_cpu = jnp.array(test_array)
        jax_array_cpu.block_until_ready()
        elapsed = time.time() - start
        times_cpu.append(elapsed)

mean_cpu = np.mean(times_cpu)
print(f"\nArray conversion time (CPU):")
print(f"  Mean: {mean_cpu*1000:.1f} ms")
print(f"  Min:  {np.min(times_cpu)*1000:.1f} ms")
print(f"  Max:  {np.max(times_cpu)*1000:.1f} ms")
print(f"  Device: {jax_array_cpu.device}")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

speedup = mean_default / mean_cpu
time_saved = (mean_default - mean_cpu) * 1000

print(f"\nDefault device: {mean_default*1000:.1f} ms")
print(f"CPU device:     {mean_cpu*1000:.1f} ms")
print(f"\nSpeedup: {speedup:.1f}x")
print(f"Time saved: {time_saved:.1f} ms per conversion")

if speedup > 10:
    print(f"\n🚀 HUGE WIN! {speedup:.0f}x faster with CPU device!")
elif speedup > 2:
    print(f"\n✅ Good speedup: {speedup:.1f}x faster")
else:
    print(f"\n⚠️  Minimal benefit: {speedup:.2f}x")

# Verify accuracy
print("\n" + "="*80)
print("ACCURACY CHECK")
print("="*80)

max_diff = np.max(np.abs(np.array(jax_array) - np.array(jax_array_cpu)))
print(f"\nMax difference: {max_diff:.2e}")
if max_diff == 0:
    print("✅ Identical results!")
else:
    print(f"⚠️  Small difference (expected due to different compute paths)")

# =============================================================================
# RECOMMENDATION
# =============================================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"""
If speedup > 10x:
-----------------
STRONGLY RECOMMEND using CPU device for JUG fitting!

Implementation:
```python
# In optimized_fitter.py, wrap JAX operations:
with jax.default_device(jax.devices('cpu')[0]):
    dt_sec_jax = jnp.array(dt_sec_cached)
    errors_jax = jnp.array(errors_sec)
    weights_jax = jnp.array(weights)
```

Expected impact:
- Saves ~{time_saved:.0f} ms per fit
- Zero accuracy loss
- Zero risk
- 5 minute implementation

If speedup < 5x:
----------------
Check if GPU is actually being used. May not have CUDA available,
in which case optimization is already applied by default.
""")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
