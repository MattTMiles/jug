from jug.residuals.simple_calculator import compute_residuals_simple
import numpy as np
import os

# Silence JAX warnings
os.environ["JAX_PLATFORM_NAME"] = "cpu"

print("Test: Verify ELL1 -> Keplerian conversion for BINARY T2")

# Paths
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236.tim"
par_ell1 = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236_tdb.par"
par_t2 = "/home/mattm/soft/JUG/tests/J2241_T2.par"

print("\n1. Running Original (ELL1)...")
res_ell1 = compute_residuals_simple(par_ell1, tim_file, verbose=False)
print(f"   RMS: {res_ell1['rms_us']:.6f} us")

print("\n2. Running Modified (T2 with ELL1 params)...")
res_t2 = compute_residuals_simple(par_t2, tim_file, verbose=False)
print(f"   RMS: {res_t2['rms_us']:.6f} us")

# Compare
diff_us = np.abs(res_ell1['residuals_us'] - res_t2['residuals_us'])
max_diff = np.max(diff_us)
mean_diff = np.mean(diff_us)

print("\nResults:")
print(f"Max difference: {max_diff:.6e} us")
print(f"Mean difference: {mean_diff:.6e} us")

if max_diff < 0.001: # Allow 1 ns tolerance (0.001 us)
    print("SUCCESS: T2 model with ELL1 params matches native ELL1 model!")
else:
    print("WARNING: Differences found. Check conversion logic.")
