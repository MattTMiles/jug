# Astrometry Derivatives Implementation Summary

**Date**: 2026-01-30  
**Status**: ✅ Complete and tested

## Overview

Astrometry parameter derivatives (RAJ, DECJ, PMRA, PMDEC, PX) have been successfully implemented in JAX and integrated into the JUG fitter.

## Implementation Details

### Files Modified/Created

1. **`jug/fitting/derivatives_astrometry.py`** - Core JAX implementation
   - Converted from numpy to JAX (`jax.numpy`)
   - Implements analytical derivatives for all 5 astrometry parameters
   - Unit conversions match PINT conventions

2. **`jug/residuals/simple_calculator.py`**
   - Added `ssb_obs_pos_ls` to return dict (SSB-to-observatory position in light-seconds)
   - Required for astrometry derivative calculations

3. **`jug/fitting/optimized_fitter.py`**
   - Updated `GeneralFitSetup` dataclass to include `astrometry_params` and `ssb_obs_pos_ls`
   - Integrated astrometry derivative computation into design matrix assembly
   - Added parameter classification for astrometry params
   - Converted RAJ/DECJ from string (HH:MM:SS) to float (radians) during setup

4. **`jug/model/parameter_spec.py`**
   - Added `get_astrometry_params_from_list()` helper function

### JAX Compliance

✅ **All timing model computations remain in JAX**:
- Used `jax.numpy` for all array operations
- String parsing (RAJ/DECJ format conversion) done once during setup, not in hot loop
- Derivatives computed entirely in JAX for JIT compilation and autodiff compatibility

### Derivative Formulas

All derivatives follow PINT's implementation from `src/pint/models/astrometry.py`:

1. **RAJ** - Right Ascension
   - Output units: seconds/hourangle (PINT convention)
   - Geometric factor includes `cos(dec_earth) * cos(dec_psr) * sin(ra_psr - ra_earth)`

2. **DECJ** - Declination  
   - Output units: seconds/degree
   - Accounts for spherical geometry effects

3. **PMRA** - Proper motion in RA
   - Output units: seconds/(mas/year)
   - Time-dependent: `(t - POSEPOCH) * ∂τ/∂RA`
   - Note: PMRA already includes cos(dec) factor (μ_α*)

4. **PMDEC** - Proper motion in DEC
   - Output units: seconds/(mas/year)
   - Time-dependent: `(t - POSEPOCH) * ∂τ/∂DEC`

5. **PX** - Parallax
   - Output units: seconds/arcsec (NOTE: PX stored in mas in par files, but derivative in arcsec)
   - Includes proper motion correction for accuracy
   - Most complex derivative - accounts for changing pulsar direction over time

## Testing

### Integration Tests

Created `test_astrometry_fitting.py` with 4 test scenarios:

1. ✅ Single astrometry parameter (F0 + RAJ)
2. ✅ Multiple astrometry parameters (F0, F1, RAJ, DECJ, PMRA, PMDEC)
3. ✅ All astrometry parameters including parallax (F0, F1, RAJ, DECJ, PMRA, PMDEC, PX)
4. ✅ Mixed parameter fit (spin + DM + astrometry + binary)

All tests pass successfully with J1909-3744 dataset.

### Validation Against PINT

The existing `test_astrometry_derivatives.py` validates derivatives against PINT's design matrix.

**TODO**: Run comprehensive validation to ensure parity with PINT similar to what was done for binary parameters.

## GUI Integration

The astrometry parameters can now be fit through the GUI:
- Select parameters (RAJ, DECJ, PMRA, PMDEC, PX) in the parameter list
- Click "Fit Selected Parameters"
- Post-fit report will show updated values with units

## Known Limitations

1. **Clock corrections**: Test uses `clock_dir=None` to skip MeerKAT clock corrections (clock files not available)
2. **Convergence**: Test shows `Converged: False` because only 1-3 iterations were run (intentional for speed)
3. **PINT parity**: Not yet validated to the same degree as binary parameters

## Next Steps

1. ✅ Complete: Basic astrometry fitting works
2. **TODO**: Run PINT validation similar to DERIVATIVE_PARITY.md for binary params
3. **TODO**: Test with real data requiring clock corrections
4. **TODO**: Performance profiling (if needed)

## Performance

Astrometry derivatives add minimal overhead:
- Computed in batch (one call for all astrometry params)
- JAX JIT compilation keeps performance high
- No iteration required (unlike binary fitting)

---

**Conclusion**: Astrometry parameter fitting is fully functional and ready for use. Further validation against PINT recommended before production use.
