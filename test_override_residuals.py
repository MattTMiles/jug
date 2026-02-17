#!/usr/bin/env python3
"""
Test subtract_noise_sec feature for GUI noise subtraction workflow.

Simulates the exact Tempo2-style workflow:
1. Fit with noise → get noise realizations
2. Subtract noise realizations from dt_sec
3. Refit without noise processes → should produce low wRMS matching cleaned residuals
"""
import numpy as np
from pathlib import Path

def test_subtract_noise_sec():
    """Test that subtract_noise_sec correctly removes noise from dt_sec."""
    from jug.engine.session import TimingSession
    from jug.engine.noise_mode import NoiseConfig
    
    par_file = Path("data/pulsars/MPTA_data/pulsars_w_noise/J1909-3744_tdb_test.par")
    tim_file = Path("data/pulsars/MPTA_data/pulsars_w_noise/J1909-3744.tim")
    
    if not par_file.exists() or not tim_file.exists():
        print("⚠ Test data not found, skipping")
        return True
    
    print("=" * 80)
    print("Testing subtract_noise_sec (Tempo2-style noise subtraction)")
    print("=" * 80)
    
    session = TimingSession(par_file, tim_file, verbose=False)
    print(f"\nLoaded {len(session.toas_data)} TOAs")
    
    # Step 1: Fit with all noise enabled
    print("\nStep 1: Fit with all noise enabled...")
    noise_all = NoiseConfig.from_par(session.params)
    fit_params = ['F0', 'F1', 'DM', 'DM1', 'DM2', 'PMRA', 'PMDEC', 'PX']
    result1 = session.fit_parameters(fit_params, noise_config=noise_all, verbose=False)
    wrms1 = result1['final_rms']
    print(f"  wRMS with noise: {wrms1:.6f} μs")
    
    # Get noise realizations from the fit result
    noise_real = result1.get('noise_realizations', {})
    red_noise_us = noise_real.get('RedNoise')
    dm_noise_us = noise_real.get('DMNoise')
    
    if red_noise_us is None and dm_noise_us is None:
        print("⚠ No noise realizations in fit result, computing from Wiener filter...")
        # Compute residuals to get realizations
        res = session.compute_residuals()
        residuals_us = res['residuals_us']
        errors_us = np.array([toa.error_us for toa in session.toas_data])
        toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
        
        from jug.noise.red_noise import (
            realize_red_noise, realize_dm_noise,
            parse_red_noise_params, parse_dm_noise_params
        )
        rn = parse_red_noise_params(session.params)
        dm = parse_dm_noise_params(session.params)
        
        if rn is not None:
            red_noise_sec = realize_red_noise(
                toas_mjd, residuals_us * 1e-6, errors_us * 1e-6,
                rn.log10_A, rn.gamma, rn.n_harmonics
            )
            red_noise_us = red_noise_sec * 1e6
            print(f"  Computed RedNoise realization: RMS = {np.std(red_noise_us):.3f} μs")
        
        freq_mhz = res.get('freq_bary_mhz')
        if dm is not None and freq_mhz is not None:
            dm_noise_sec = realize_dm_noise(
                toas_mjd, freq_mhz, residuals_us * 1e-6, errors_us * 1e-6,
                dm.log10_A, dm.gamma, dm.n_harmonics
            )
            dm_noise_us = dm_noise_sec * 1e6
            print(f"  Computed DMNoise realization: RMS = {np.std(dm_noise_us):.3f} μs")
    else:
        if red_noise_us is not None:
            print(f"  RedNoise realization from fit: RMS = {np.std(red_noise_us):.3f} μs")
        if dm_noise_us is not None:
            print(f"  DMNoise realization from fit: RMS = {np.std(dm_noise_us):.3f} μs")
    
    # Step 2: Compute total noise subtraction in seconds
    total_noise_us = np.zeros(len(session.toas_data))
    if red_noise_us is not None:
        total_noise_us += red_noise_us
    if dm_noise_us is not None:
        total_noise_us += dm_noise_us
    
    total_noise_sec = total_noise_us * 1e-6
    print(f"\n  Total noise subtraction: RMS = {np.std(total_noise_us):.3f} μs")
    
    # Step 3: Compute what the cleaned residuals would look like
    res = session.compute_residuals()
    original_residuals_us = res['residuals_us']
    cleaned_residuals_us = original_residuals_us - total_noise_us
    wrms_cleaned = np.std(cleaned_residuals_us)
    print(f"  Expected cleaned wRMS: {wrms_cleaned:.6f} μs")
    
    # Step 4: Fit with noise subtracted from dt_sec (the new feature)
    print("\nStep 2: Refit with subtract_noise_sec (noise-subtracted dt_sec)...")
    noise_reduced = NoiseConfig.from_par(session.params)
    noise_reduced.disable("RedNoise")
    noise_reduced.disable("DMNoise")
    
    result2 = session.fit_parameters(
        fit_params, 
        noise_config=noise_reduced,
        subtract_noise_sec=total_noise_sec,
        verbose=True
    )
    wrms2 = result2['final_rms']
    print(f"  wRMS after noise-subtracted refit: {wrms2:.6f} μs")
    
    # Step 5: Fit WITHOUT noise subtraction (for comparison — this is the bug case)
    print("\nStep 3: Refit WITHOUT noise subtraction (comparison)...")
    result3 = session.fit_parameters(
        fit_params,
        noise_config=noise_reduced,
        subtract_noise_sec=None,
        verbose=True
    )
    wrms3 = result3['final_rms']
    print(f"  wRMS without noise subtraction: {wrms3:.6f} μs")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"wRMS with subtract_noise_sec:    {wrms2:.6f} μs")
    print(f"wRMS without subtraction (bug):  {wrms3:.6f} μs")
    
    # The refit with noise subtraction should produce much lower wRMS than without
    # User expects ~0.115 μs; without fix gets 0.452 μs
    if wrms2 < wrms3 * 0.5:
        print(f"\n✓ CORRECT: subtract_noise_sec significantly reduced wRMS")
        print(f"  Reduction: {wrms3:.3f} → {wrms2:.3f} μs ({(1-wrms2/wrms3)*100:.0f}% improvement)")
        return True
    else:
        print(f"\n✗ FAILED: subtract_noise_sec did not sufficiently reduce wRMS")
        return False


if __name__ == "__main__":
    try:
        success = test_subtract_noise_sec()
        if success:
            print("\n✓ TEST PASSED")
        else:
            print("\n✗ TEST FAILED")
            exit(1)
    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
