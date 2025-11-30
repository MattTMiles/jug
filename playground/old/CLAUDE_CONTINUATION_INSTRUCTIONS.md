# Claude Continuation Instructions - JUG MK3 Notebook

## Task Summary
Creating a clean production notebook `residual_maker_playground_MK3.ipynb` that computes pulsar timing residuals independently using JAX. The MK2 notebook was verified working with ~0.3 µs RMS difference from PINT.

## Current Status
- **MK3 notebook created**: `/home/mattm/soft/JUG/residual_maker_playground_MK3.ipynb`
- **GPU memory fix applied**: Cell 1 now has XLA environment variables to prevent GPU memory pre-allocation
- **Cells 1-10 executed successfully** before the GPU error occurred
- **Cell 11 failed** due to CUDA out-of-memory (before the fix was applied)
- **Cells 12-13 not yet executed**

## What To Do After Restart

### Step 1: Open the MK3 notebook
```
/home/mattm/soft/JUG/residual_maker_playground_MK3.ipynb
```

### Step 2: Run all cells from the beginning
Restart kernel and run all cells in order:

| Cell | Description |
|------|-------------|
| 1 | Imports with GPU memory management env vars |
| 2 | Constants |
| 3 | Data paths |
| 4 | IERS setup |
| 5 | Load JPL ephemeris |
| 6 | SSB/delay functions |
| 7 | ELL1 binary delay function |
| 8 | DM delay function |
| 9 | Par file parsing |
| 10 | Load data with PINT |
| 11 | **Main JUG computation** (this failed before, should work now) |
| 12 | Comparison with PINT |
| 13 | Comparison plots |

### Step 3: Verify expected results
From MK2, we expect:
- JUG RMS: ~0.86 µs
- PINT RMS: ~0.82 µs  
- Difference RMS: ~0.32 µs
- Correlation: ~0.93

### Step 4: Success criteria
If all cells run and results match above → MK3 is complete and ready for production.

## Fallback if GPU Error Persists
Add this line after the env vars in cell 1:
```python
jax.config.update('jax_platforms', 'cpu')
```

## Key Files
| File | Description |
|------|-------------|
| `/home/mattm/soft/JUG/residual_maker_playground_MK2.ipynb` | Working reference (verified ~0.3 µs accuracy) |
| `/home/mattm/soft/JUG/residual_maker_playground_MK3.ipynb` | Clean production version (needs verification) |
| `/home/mattm/soft/JUG/residual_maker_playground_MK2_BACKUP.ipynb` | Backup of MK2 |

## GPU Memory Fix Applied
Cell 1 now includes these lines at the top (before any imports):
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_DISABLE_MMAP_CACHE"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
```

## Data Files Used
- Par file: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par`
- Tim file: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim`
- Ephemeris: `/home/mattm/soft/JUG/data/ephemeris/de440s.bsp`
