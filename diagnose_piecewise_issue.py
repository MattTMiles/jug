"""
Diagnose the piecewise residual issue.

The problem: We need to verify that piecewise residuals match global residuals
when using the SAME parameters.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble

SECS_PER_DAY = 86400.0


def compute_residuals_piecewise(dt_sec_global, pepoch_global_mjd, segments, 
                                f0_global, f1_global):
    """
    Compute phase residuals using local coordinates per segment.
    
    CORRECTED VERSION: Use dt_sec directly (it already has delays applied).
    """
    n_toas = len(dt_sec_global)
    residuals_sec = np.zeros(n_toas)
    
    for seg in segments:
        idx = seg['indices']
        
        # Epoch offset in seconds
        # This is a coordinate transformation, not a physical time
        dt_epoch = (seg['local_pepoch_mjd'] - pepoch_global_mjd) * SECS_PER_DAY
        
        # Local F0 from continuity constraint
        f0_local = f0_global + f1_global * dt_epoch
        
        # Local time coordinates (SMALL!)
        # dt_sec_global is already emission time from global PEPOCH
        dt_local = dt_sec_global[idx] - dt_epoch
        
        # Phase computation
        phase = dt_local * (f0_local + dt_local * (f1_global / 2.0))
        
        # Wrap to nearest integer pulse
        phase_wrapped = phase - np.round(phase)
        
        # Convert to time residuals
        residuals_sec[idx] = phase_wrapped / f0_local
    
    return residuals_sec


def create_time_segments(tdb_mjd, segment_duration_days=500.0):
    """Divide TOAs into temporal segments."""
    t_min = np.min(tdb_mjd)
    t_max = np.max(tdb_mjd)
    
    n_segments = max(1, int(np.ceil((t_max - t_min) / segment_duration_days)))
    
    segments = []
    for i in range(n_segments):
        seg_start = t_min + i * segment_duration_days
        seg_end = t_min + (i + 1) * segment_duration_days
        
        mask = (tdb_mjd >= seg_start) & (tdb_mjd < seg_end)
        if i == n_segments - 1:
            mask = (tdb_mjd >= seg_start) & (tdb_mjd <= seg_end)
        
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
            
        seg_times = tdb_mjd[indices]
        
        segments.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(seg_times)),
            'tmin_mjd': float(np.min(seg_times)),
            'tmax_mjd': float(np.max(seg_times)),
            'n_toas': len(indices)
        })
    
    return segments


def main():
    print("="*80)
    print("DIAGNOSING PIECEWISE RESIDUAL COMPUTATION")
    print("="*80)
    
    # Load data
    par_file = Path("data/pulsars/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    params = parse_par_file(par_file)
    pepoch_global_mjd = float(get_longdouble(params, 'PEPOCH'))
    f0_initial = float(params['F0'])
    f1_initial = float(params['F1'])
    
    print(f"\nParameters:")
    print(f"  PEPOCH = {pepoch_global_mjd:.6f} MJD")
    print(f"  F0 = {f0_initial:.15f} Hz")
    print(f"  F1 = {f1_initial:.6e} Hz/s")
    
    # Compute delays
    result = compute_residuals_simple(
        par_file, tim_file,
        clock_dir="data/clock",
        subtract_tzr=False,
        verbose=False
    )
    
    dt_sec_global = result['dt_sec']
    tdb_mjd = result['tdb_mjd']
    residuals_standard = result['residuals_us'] * 1e-6
    
    print(f"\nData: {len(dt_sec_global)} TOAs")
    print(f"  Time span: {(tdb_mjd.max()-tdb_mjd.min())/365.25:.1f} years")
    
    # Create segments
    segments = create_time_segments(tdb_mjd, segment_duration_days=500.0)
    print(f"\nCreated {len(segments)} segments")
    
    # Method 1: Standard global residuals
    print("\n" + "="*80)
    print("METHOD 1: Standard Global Residuals")
    print("="*80)
    
    phase_global = dt_sec_global * (f0_initial + dt_sec_global * (f1_initial / 2.0))
    phase_wrapped = phase_global - np.round(phase_global)
    residuals_global = phase_wrapped / f0_initial
    
    print(f"  RMS: {np.std(residuals_global)*1e6:.6f} μs")
    print(f"  Range: [{np.min(residuals_global)*1e6:.3f}, {np.max(residuals_global)*1e6:.3f}] μs")
    
    # Method 2: Piecewise residuals
    print("\n" + "="*80)
    print("METHOD 2: Piecewise Residuals")
    print("="*80)
    
    residuals_piecewise = compute_residuals_piecewise(
        dt_sec_global, pepoch_global_mjd, segments,
        f0_initial, f1_initial
    )
    
    print(f"  RMS: {np.std(residuals_piecewise)*1e6:.6f} μs")
    print(f"  Range: [{np.min(residuals_piecewise)*1e6:.3f}, {np.max(residuals_piecewise)*1e6:.3f}] μs")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    diff = residuals_piecewise - residuals_global
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"\nDifference (piecewise - global):")
    print(f"  Max: {max_diff:.3e} s = {max_diff*1e9:.3f} ns")
    print(f"  RMS: {rms_diff:.3e} s = {rms_diff*1e9:.3f} ns")
    
    if max_diff < 1e-15:
        print(f"\n✓ PASS: Methods are numerically identical (< 1 fs)")
    elif max_diff < 1e-12:
        print(f"\n✓ PASS: Methods agree to ps precision")
    else:
        print(f"\n⚠ WARNING: Difference of {max_diff*1e9:.3f} ns detected")
    
    # Also compare with result['residuals_us']
    print("\n" + "="*80)
    print("COMPARISON WITH compute_residuals_simple OUTPUT")
    print("="*80)
    
    diff_std = residuals_piecewise - residuals_standard
    max_diff_std = np.max(np.abs(diff_std))
    
    print(f"\nDifference (piecewise - standard):")
    print(f"  Max: {max_diff_std:.3e} s = {max_diff_std*1e6:.3f} μs")
    
    if max_diff_std < 1e-9:
        print(f"✓ PASS: Agrees with standard residuals (< 1 ns)")
    else:
        print(f"⚠ WARNING: Difference of {max_diff_std*1e6:.3f} μs")
        
        # Debug: check if it's a constant offset
        print(f"\nDebug:")
        print(f"  Mean difference: {np.mean(diff_std)*1e6:.3f} μs")
        print(f"  Std of difference: {np.std(diff_std)*1e6:.6f} μs")


if __name__ == "__main__":
    main()
