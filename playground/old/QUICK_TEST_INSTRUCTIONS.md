# Quick Test: Run the Modified Notebook

## TL;DR

The notebook has been updated to use PINT's correct infinite-frequency times instead of tempo2's BAT. This should dramatically improve residuals from ~850 μs to <10 μs.

## What Changed

**File**: `residual_maker_playground_claude_debug.ipynb`

**Change**: Added Cell 17 that:
1. Loads PINT library
2. Imports PINT's correct infinite-frequency barycentric times (`tdbld` column)
3. Extracts PINT's correct TZR infinite-frequency time
4. Overrides the notebook's `t_inf_mjd` variable
5. Updates the model's TZR reference with correct values

## How to Test

### Option 1: Run in Jupyter (Recommended)

```bash
cd /home/mattm/soft/JUG
jupyter notebook residual_maker_playground_claude_debug.ipynb
```

Then:
1. Run cells sequentially from top to bottom
2. Cell 17 will print status about loading PINT
3. Cell 18 (CORRECTED TEST) will show residuals

### Option 2: Run Entire Notebook via Command Line

```bash
cd /home/mattm/soft/JUG
jupyter nbconvert --to notebook --execute residual_maker_playground_claude_debug.ipynb
```

### Option 3: Convert to Python Script and Run

```bash
cd /home/mattm/soft/JUG
jupyter nbconvert --to script residual_maker_playground_claude_debug.ipynb --output temp_notebook_script.py
python3 temp_notebook_script.py
```

## Expected Output

When Cell 17 executes, you should see:
```
================================================================================
IMPORTING PINT'S CORRECT TIMES (TEMPORARY FIX)
================================================================================
✓ Imported 10408 TOAs from PINT
  First t_inf: 58526.2146899...

✓ TZR from PINT:
  TZRMJD (topocentric): 59679.2480619...
  TZR (infinite-freq): 59679.2488627...

✓ Phase at TZR:
  Absolute: ...
  Fractional: ...

✓✓✓ Model updated with PINT's correct TZR!
  phase_ref_mjd: 59679.2488627...
  phase_offset_cycles: ...

⚠️  NOTE: This uses PINT's computed times as a temporary fix.
   Full independence requires implementing barycentric corrections in JUG.
```

When Cell 18 executes, you should see residuals that are much better than ~850 μs.

## What This Proves

If residuals are now <10 μs:
- ✓ JUG's residual calculation logic is CORRECT
- ✓ The problem was NOT the calculation method
- ✓ The problem WAS using wrong input times (tempo2's BAT)
- ✓ JUG can achieve matching tempo2/PINT performance

If residuals are still large:
- The problem might be deeper than expected
- Could be related to other timing parameters
- May need further investigation

## What This DOESN'T Prove

This test uses PINT's computed times, so it doesn't prove independence. To truly show JUG works independently, we need to:

1. Implement observatory position calculation in SSB frame
2. Implement pulsar direction computation
3. Implement Roemer delay calculation
4. Implement Shapiro delay calculation
5. Compute barycentric times from scratch
6. Remove PINT dependency entirely

See `CONTINUATION_PROGRESS.md` for the full implementation plan.

## Troubleshooting

### If Cell 17 Says "PINT not available"

- PINT is not installed
- Install with: `pip install pint-pulsar`
- The notebook will fall back to tempo2-based times (old behavior)

### If Cell 18 Shows No Results

- The notebook may have encountered an error
- Look for error messages in earlier cells
- Check that `temp_pre_general2.out` exists (tempo2 output)

### If Residuals Are Still ~850 μs

- There might be another issue we haven't identified
- The TZR calculation might still be wrong
- Could be related to how the residuals are being computed
- See `PINT_COMPARISON_FINDINGS.md` for more context

## Files to Check

After running, examine these output files:
- `temp_pre_general2.out` - tempo2 residuals (should show ~0.817 μs RMS)
- `temp_pre_components_next.out` - tempo2 delay components (for validation)
- Output from cell 18 - JUG residuals (should now be <10 μs)

## Next Steps After Testing

1. **If successful** (<10 μs error):
   - Document that JUG's calculation method is correct
   - Plan implementation of barycentric corrections
   - Aim for full independence in 1-2 weeks

2. **If unsuccessful** (still large error):
   - Debug the TZR calculation more carefully
   - Check if other timing parameters need adjustment
   - May need to compare against PINT's internal calculations directly

---

**Current Status**: Ready for testing. Expected time to run: 5-10 minutes.
