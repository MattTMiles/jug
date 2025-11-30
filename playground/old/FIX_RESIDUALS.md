# Fix for JUG Residuals

## Problem Identified

JUG is currently computing residuals at **topocentric** time, but Tempo2/PINT compute them at **barycentric** (or infinite-frequency) time. This causes the ~1000x discrepancy (850 μs vs 0.8 μs).

## Root Cause

In cell 204, the code uses:
```python
res_sec_topo = residuals_seconds_at_topocentric_time(t_topo_jax, model)
```

But from the CLAUDE.md documentation and the timing pipeline:
1. Timing residuals should be computed at the **emission time** (after binary delays)
2. Or at the **infinite-frequency time** (after DM correction)
3. NOT at the raw topocentric observation time

## The Correct Flow

According to CLAUDE.md:
1. Start with topocentric TOA (t_mjd)
2. Add clock corrections → UTC/TT
3. Add barycentric delay → barycentric arrival time (t_bary)
4. Subtract binary delay → emission time (t_em)
5. Subtract DM delay → infinite-frequency TOA (t_inf)
6. **Compute residuals at t_inf**

## Fix Required

The code should use `t_inf_mjd` or `t_em_mjd` (which are computed in cell 200), NOT `t_mjd` (topocentric).

Replace in cell 204:
```python
# OLD - WRONG
t_topo_jax = jnp.array(t_mjd, dtype=jnp.float64)
res_sec_topo = residuals_seconds_at_topocentric_time(t_topo_jax, model)
```

With:
```python
# NEW - CORRECT
# Use infinite-frequency barycentric time (after DM correction)
t_inf_jax = jnp.array(t_inf_mjd, dtype=jnp.float64)

# Compute residuals at infinite-frequency time
res_sec = residuals_seconds_at_topocentric_time(t_inf_jax, model)
# Note: function name is misleading - it actually works for any time input
```

## Additional Issue: phase_offset_cycles

The TZR calculation in cell 200 correctly computes `phase_offset_cycles = 0.08740234375`, but the residual function uses:

```python
phase_diff = phase - phase_ref - model.phase_offset_cycles
```

This should anchor the phases correctly. However, verify that `t_inf_mjd` exists and is properly computed.

## Expected Result

After this fix:
- RMS should drop from ~850 μs to ~0.8-10 μs (similar to Tempo2)
- Residuals should correlate strongly with Tempo2's residuals
- The 1000x discrepancy should disappear

## Implementation Steps

1. Check that `t_inf_mjd` is computed in cell 200
2. Replace `t_mjd` with `t_inf_mjd` in cell 204
3. Re-run cell 204 and check RMS
4. Compare with Tempo2 residuals to verify match
