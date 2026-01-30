# Astrometry Derivatives - Validation Complete

## Summary

The astrometry derivative functions in `jug/fitting/derivatives_astrometry.py` have been **validated against PINT** and match perfectly for all 5 parameters:

- ✅ RAJ (Right Ascension)
- ✅ DECJ (Declination)  
- ✅ PMRA (Proper Motion in RA)
- ✅ PMDEC (Proper Motion in DEC)
- ✅ PX (Parallax)

## Test Results

All parameters show perfect agreement with PINT (mean ratio = 1.0000000002, max differences at floating-point precision):

```
RAJ:    Max abs diff: 7.42e-08 milliseconds  ✓
DECJ:   Max abs diff: 1.71e-09 milliseconds  ✓
PMRA:   Max abs diff: 8.38e-15 milliseconds  ✓
PMDEC:  Max abs diff: 2.20e-15 milliseconds  ✓
PX:     Max abs diff: 8.67e-16 milliseconds  ✓
```

Test data: J1909-3744 (10,408 TOAs)
Test script: `test_astrometry_derivatives.py`

## Units and Conventions

### Input Parameters (Internal JUG representation)
- RAJ, DECJ: radians
- PMRA, PMDEC: rad/yr
- POSEPOCH: MJD

### Output Derivatives (delay derivatives in seconds)
- RAJ: seconds/hourangle
- DECJ: seconds/degree
- PMRA: seconds/(mas/year)
- PMDEC: seconds/(mas/year)
- **PX: seconds/arcsec** (NOTE: PX is stored in mas in par files, but derivatives use arcsec per PINT convention)

### PINT Comparison Notes
- PINT's design matrix uses **milliseconds**, JUG uses **seconds** internally
- For validation, multiply JUG derivatives by 1000
- PINT uses PX derivatives in arcsec, not mas (even though PX parameter is in mas)

## Key Implementation Details

1. **Proper Motion Correction for PX**: The PX derivative includes proper motion correction for accuracy over long time spans (20+ years), matching PINT's behavior

2. **Geometric Formulas**: All formulas follow standard astrometry:
   - RAJ: `∂τ/∂RA = r × cos(dec_e) × cos(dec_p) × sin(ra_p - ra_e) / c`
   - DECJ: Similar but for declination
   - PMRA/PMDEC: Time-dependent extensions of position derivatives
   - PX: `∂τ/∂PX = 0.5 × px_r² / (AU × c)` where px_r is transverse distance

3. **JAX Compatibility**: All functions use NumPy (not JAX) since they're computed once during setup, not in the iterative loop

## Next Steps

To integrate into the fitter:

1. **Add `ssb_obs_pos_lt_sec` to session cache** (currently not cached)
2. **Add astrometry parameter routing** to `_run_general_fit_iterations()` 
3. **Implement parameter limits/constraints** (e.g., DECJ ∈ [-π/2, π/2])
4. **Update GUI** to expose astrometry parameters for fitting
5. **Test end-to-end** fitting with astrometry parameters

## Files Modified

- `jug/fitting/derivatives_astrometry.py` - Fixed PX units (mas → arcsec)
- `test_astrometry_derivatives.py` - Created validation test script

## Known Issues / Limitations

- Astrometry derivatives **not yet integrated into fitter** - currently standalone validated code
- SSB positions not in session cache - would need to load for each fit
- No parameter bounds checking (e.g., |DECJ| ≤ 90°)
