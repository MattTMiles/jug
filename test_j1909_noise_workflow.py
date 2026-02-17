"""
Test the exact workflow reported by the user with J1909-3744_tdb_test.par.

Steps:
1. Open session with J1909-3744_tdb_test.par (has extensive noise)
2. Fit with all noise enabled → expect wRMS ~ 0.565 μs
3. Disable DM noise and red noise → expect wRMS ~ 0.115 μs
4. Fit again → should get ~ 0.115 μs, NOT 0.565 μs (THIS IS THE BUG)
"""

from pathlib import Path
from jug.engine.session import TimingSession
from jug.engine.noise_mode import NoiseConfig

par_file = Path("data/pulsars/MPTA_data/pulsars_w_noise/J1909-3744_tdb_test.par")
tim_file = Path("data/pulsars/MPTA_data/pulsars_w_noise/J1909-3744.tim")

if not par_file.exists() or not tim_file.exists():
    print("Data files not found")
    exit(1)

print("=" * 80)
print("Testing J1909 noise workflow bug")
print("=" * 80)

# Step 1: Create session
print("\nStep 1: Opening session...")
session = TimingSession(par_file, tim_file, verbose=False)
print(f"  Loaded {len(session.toas_data)} TOAs")

# Step 2: Fit with ALL noise enabled
print("\nStep 2: Fit with ALL noise enabled...")
noise_all = NoiseConfig.from_par(session.params)
print(f"  Noise processes detected: {list(noise_all.enabled.keys())}")
print(f"  Enabled states: {noise_all.enabled}")

fit_params = ['F0', 'F1', 'DM', 'DM1', 'DM2', 'PMRA', 'PMDEC', 'PX']

result1 = session.fit_parameters(
    fit_params=fit_params,
    max_iter=10,
    noise_config=noise_all,
    verbose=True
)

print(f"\n  Result 1:")
print(f"    wRMS: {result1['final_rms']:.6f} μs")
print(f"    Converged: {result1['converged']}")
print(f"    Iterations: {result1['iterations']}")

# Step 3: Disable DM noise and red noise
print("\nStep 3: Creating noise config with DM noise and RedNoise DISABLED...")
noise_reduced = NoiseConfig.from_par(session.params)
# Disable the achromatic red noise and DM noise
noise_reduced.disable('RedNoise')
noise_reduced.disable('DMNoise')
print(f"  Enabled states after disabling: {noise_reduced.enabled}")

# Step 4: Fit again with reduced noise
print("\nStep 4: Fit with DM noise and RedNoise disabled...")
result2 = session.fit_parameters(
    fit_params=fit_params,
    max_iter=10,
    noise_config=noise_reduced,
    verbose=True
)

print(f"\n  Result 2:")
print(f"    wRMS: {result2['final_rms']:.6f} μs")
print(f"    Converged: {result2['converged']}")
print(f"    Iterations: {result2['iterations']}")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print(f"wRMS with all noise:     {result1['final_rms']:.6f} μs")
print(f"wRMS with reduced noise: {result2['final_rms']:.6f} μs")

if abs(result2['final_rms'] - result1['final_rms']) < 0.01:
    print("\n❌ BUG CONFIRMED: wRMS did not change when noise was disabled!")
    print("   Expected: wRMS should INCREASE when noise is disabled")
    print("   Actual: wRMS stayed the same, suggesting noise is still being applied")
    exit(1)
else:
    print("\n✓ CORRECT: wRMS changed when noise config changed")
    diff = result2['final_rms'] - result1['final_rms']
    print(f"   Difference: {diff:+.6f} μs")
    if diff > 0:
        print("   ✓ wRMS increased (expected when removing noise whitening)")
    exit(0)
