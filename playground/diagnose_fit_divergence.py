#!/usr/bin/env python3
"""
Diagnose why fitting all parameters causes divergence on second iteration.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jug.io.par_reader import parse_par_file, format_ra, format_dec
from jug.fitting.optimized_fitter import fit_parameters_optimized
import tempfile
import os

def write_par_file(params, path):
    """Write parameters to a .par file"""
    with open(path, 'w') as f:
        for key, val in params.items():
            f.write(f"{key} {val}\n")

def main():
    # Load data
    par_file = Path("data/pulsars/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    params = parse_par_file(par_file)
    
    # All fittable parameters
    all_params = ['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 
                  'DM', 'DM1', 'DM2',
                  'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT']
    
    print("=" * 80)
    print("TESTING ITERATIVE FITTING - ALL PARAMETERS")
    print("=" * 80)
    
    # Create temp directory for updated par files
    temp_dir = tempfile.mkdtemp()
    current_par = par_file
    
    try:
        for iteration in range(5):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*80}")
            
            # Reload current params
            params = parse_par_file(current_par)
            
            # Show current parameter values
            print("\nCurrent parameters:")
            for param in all_params:
                if param in params:
                    print(f"  {param:10s} = {params[param]}")
            
            # Perform fit
            result = fit_parameters_optimized(
                par_file=current_par,
                tim_file=tim_file,
                fit_params=all_params,
                max_iter=10,
                convergence_threshold=1e-10
            )
            
            print(f"\nFit converged: {result['converged']}")
            print(f"RMS: {result['final_rms']:.6f} μs")
            print(f"Iterations: {result['iterations']}")
            
            # Show parameter updates
            print("\nParameter changes:")
            for param in all_params:
                if param in params and param in result['final_params']:
                    old_val = params[param]
                    new_val = result['final_params'][param]
                    
                    # Handle string (RAJ/DECJ) vs numeric
                    if isinstance(old_val, str) and isinstance(new_val, str):
                        print(f"  {param:10s}: {old_val} -> {new_val}")
                    elif isinstance(old_val, str):
                        print(f"  {param:10s}: {old_val} -> {new_val:.15e}")
                    elif isinstance(new_val, str):
                        print(f"  {param:10s}: {old_val:.15e} -> {new_val}")
                    else:
                        diff = new_val - old_val
                        rel_change = abs(diff / old_val) if old_val != 0 else 0
                        print(f"  {param:10s}: {old_val:.15e} -> {new_val:.15e} (Δ={diff:.3e}, rel={rel_change:.3e})")
            
            # Update params for next iteration
            from jug.io.par_reader import format_ra, format_dec
            updated_params = result['final_params'].copy()
            # Convert RAJ/DECJ from radians back to string format
            if 'RAJ' in updated_params:
                updated_params['RAJ'] = format_ra(updated_params['RAJ'])
            if 'DECJ' in updated_params:
                updated_params['DECJ'] = format_dec(updated_params['DECJ'])
            params.update(updated_params)
            
            # Check for NaNs or extreme values
            has_problem = False
            for param in all_params:
                if param in params:
                    val = params[param]
                    if not isinstance(val, str):
                        if np.isnan(val) or np.isinf(val):
                            print(f"\n⚠️  WARNING: {param} has become NaN or Inf!")
                            has_problem = True
                        elif abs(val) > 1e10:
                            print(f"\n⚠️  WARNING: {param} has extreme value: {val}")
                            has_problem = True
            
            if has_problem:
                print("\n❌ DIVERGENCE DETECTED - STOPPING")
                break
            
            # Write updated par file for next iteration
            next_par = Path(temp_dir) / f"iter{iteration+1}.par"
            write_par_file(params, next_par)
            current_par = next_par
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()
