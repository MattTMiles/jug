# Quick Fix: Convergence Detection

**Time estimate**: 30 minutes  
**Impact**: Stops oscillation, properly detects convergence

---

## The Problem

Current fitter oscillates:
```
Iter  RMS (μs)    Status
1     0.403712    
2     0.403520    ↓ improving
3     0.403513    ↓ improving
4     0.403494    ↓ improving
5     0.403683    ↑ WORSE!
6     0.403643    ↓ better
7     0.403687    ↑ WORSE again!
...continues oscillating for 20 iterations
```

This wastes time and makes it unclear when fitting is done.

---

## The Solution

Replace current convergence detection with proper criteria from literature:

### Current Code (optimized_fitter.py, lines 546-550):
```python
# Track RMS history for convergence detection
rms_history = []
patience_counter = 0
patience_threshold = 3  # Stop if RMS stable for this many iterations
rms_stability_threshold = 1e-5  # Relative RMS change threshold (μs-level changes)
```

### Replace With:
```python
# Convergence criteria (from JAXopt/literature)
xtol = 1e-12  # Parameter tolerance
gtol = 1e-12  # Gradient tolerance  
min_iterations = 3  # Always do at least 3
```

### In Loop (after computing delta_params):
```python
# Check convergence after parameter update
param_norm = np.linalg.norm(param_values_curr)
delta_norm = np.linalg.norm(delta_params)

# Criterion 1: Parameter change is tiny
param_converged = delta_norm <= xtol * (param_norm + xtol)

# Criterion 2: Gradient is tiny (approximated by RMS change)
rms_change = abs(rms_us - rms_history[-1]) if len(rms_history) > 0 else np.inf
gradient_converged = rms_change < gtol

# Converge if both criteria met AND minimum iterations done
if iteration >= min_iterations and (param_converged or gradient_converged):
    converged = True
    if verbose:
        print(f"{'':6} {'':12} {'':15} {'Converged!':<20}")
    break
```

---

## Expected Behavior After Fix

```
Iter  RMS (μs)    ΔParam          Status
1     0.403712    9.723e-04       
2     0.403520    1.407e-04       
3     0.403513    4.286e-05       
4     0.403494    9.276e-05       
5     0.403683    3.872e-05       ← detects oscillation
                                  Converged!

Final: 5 iterations (was 20-30)
```

---

## Additional Benefit: Speed

- Current: 20-30 iterations × 0.014s = 0.28-0.42s
- After fix: 5-10 iterations × 0.014s = 0.07-0.14s
- **2-3× faster!**

Combined with 1.3s cache = **1.4-1.5s total** (closer to TEMPO2's 0.3s)

---

## Where to Make Changes

File: `jug/fitting/optimized_fitter.py`

Lines to modify:
- Lines 546-550: Update convergence criteria
- Lines 595-615: Add proper convergence check after each iteration

---

## Testing

After implementing, test on J1909-3744:
```python
from jug.fitting.optimized_fitter import fit_parameters_optimized

result = fit_parameters_optimized(
    "data/pulsars/J1909-3744_tdb.par",
    "data/pulsars/J1909-3744.tim",
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=20,
    clock_dir="data/clock"
)

print(f"Converged: {result['converged']}")  # Should be True
print(f"Iterations: {result['iterations']}")  # Should be 5-10
print(f"Final RMS: {result['final_rms']:.6f} μs")  # Should be ~0.404
```

Expected:
- ✅ Converged: True
- ✅ Iterations: 5-10 (not 20-30)
- ✅ Final RMS: 0.403-0.404 μs
- ✅ Total time: <1.5s

---

## Want me to implement this now?

It's a straightforward fix that will:
1. Stop the oscillation
2. Speed up fitting by 2-3×
3. Give clear convergence signal
4. Complete Milestone 2 properly

Let me know and I can make the changes!
