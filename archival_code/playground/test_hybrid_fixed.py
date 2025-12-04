# Corrected hybrid method: pure numpy/longdouble for phase calc

import numpy as np

def compute_phase_hybrid_corrected(dt_sec_f64, f0, f1, chunk_size=100):
    """Compute phase using hybrid chunked method - CORRECTED VERSION.
    
    Key fix: Keep phase accumulation in longdouble, not float64.
    """
    n_toas = len(dt_sec_f64)
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    
    # Accumulate in LONGDOUBLE
    phase_ld = np.zeros(n_toas, dtype=np.longdouble)
    
    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_toas)
        
        # Get chunk in longdouble
        dt_chunk_ld = np.array(dt_sec_f64[start:end], dtype=np.longdouble)
        
        # Compute offset
        t_offset_ld = np.mean(dt_chunk_ld)
        
        # Phase at offset (longdouble)
        phase_offset_ld = f0_ld * t_offset_ld + np.longdouble(0.5) * f1_ld * t_offset_ld**2
        
        # Local deviations (longdouble)
        dt_local_ld = dt_chunk_ld - t_offset_ld
        
        # Local phase (longdouble)
        phase_local_ld = (f0_ld * dt_local_ld + 
                          f1_ld * t_offset_ld * dt_local_ld + 
                          np.longdouble(0.5) * f1_ld * dt_local_ld**2)
        
        # Total phase (longdouble)
        phase_ld[start:end] = phase_offset_ld + phase_local_ld
    
    return phase_ld

# Test it
n_toas = 1000
dt_sec = np.linspace(-1e8, 1e8, n_toas)
f0 = 339.0
f1 = -1.6e-15

# Ground truth: direct longdouble
dt_sec_ld = np.array(dt_sec, dtype=np.longdouble)
f0_ld = np.longdouble(f0)
f1_ld = np.longdouble(f1)
phase_truth = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))

# Hybrid method
phase_hybrid = compute_phase_hybrid_corrected(dt_sec, f0, f1, chunk_size=100)

# Compare
diff = (phase_hybrid - phase_truth).astype(np.float64)
rms_cycles = np.sqrt(np.mean(diff**2))
rms_ns = rms_cycles / f0 * 1e9

print("="*80)
print("CORRECTED HYBRID METHOD")
print("="*80)
print(f"RMS difference from longdouble: {rms_cycles:.6e} cycles")
print(f"RMS timing error: {rms_ns:.3f} ns")
print(f"Max error: {np.max(np.abs(diff))/f0*1e9:.3f} ns")

if rms_ns < 1.0:
    print("\n✓ SUCCESS: Sub-nanosecond precision achieved!")
else:
    print(f"\n✗ PROBLEM: {rms_ns:.1f} ns error (target: <1 ns)")
