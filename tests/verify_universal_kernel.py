from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path
import time
import jax
import os

# Silence JAX warnings
os.environ["JAX_PLATFORM_NAME"] = "cpu" # or gpu if available, but let's stick to default
# Actually, JUG sets x64.

print("Test started.")

datasets = [
    {
        "name": "J2241-5236 (ELL1)",
        "par": "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236_tdb.par",
        "tim": "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236_tdb.tim"
    },
    {
        "name": "J1022+1001 (DDH)",
        "par": "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1022+1001_tdb.par",
        "tim": "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1022+1001.tim"
    }
]

for ds in datasets:
    print(f"\n=== Testing {ds['name']} ===")
    try:
        start = time.time()
        res = compute_residuals_simple(ds['par'], ds['tim'], verbose=True)
        end = time.time()
        print(f"Runtime: {end - start:.4f} s")
        print(f"RMS: {res['rms_us']:.4f} us")
        print(f"N_TOAs: {res['n_toas']}")
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
