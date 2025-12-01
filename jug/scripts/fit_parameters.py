#!/usr/bin/env python3
"""
JUG Fitting Command-Line Interface
===================================

Fit pulsar timing parameters from .par and .tim files.

Usage:
    jug-fit J1909.par J1909.tim --fit F0 F1 --device cpu
    jug-fit J1909.par J1909.tim --fit F0 F1 DM --device auto --max-iter 50
    jug-fit J1909.par J1909.tim --fit F0 F1 --device gpu --output fitted.par
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.utils.device import set_device_preference, print_device_info
from jug.io.par_reader import parse_par_file


def main():
    parser = argparse.ArgumentParser(
        description='Fit pulsar timing model parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit F0 and F1 (default: CPU)
  jug-fit J1909.par J1909.tim --fit F0 F1
  
  # Force GPU usage
  jug-fit J1909.par J1909.tim --fit F0 F1 --device gpu
  
  # Automatic device selection based on problem size
  jug-fit J1909.par J1909.tim --fit F0 F1 DM --device auto
  
  # Save fitted parameters to new file
  jug-fit J1909.par J1909.tim --fit F0 F1 --output fitted.par
  
  # More iterations, custom convergence
  jug-fit J1909.par J1909.tim --fit F0 F1 --max-iter 50 --threshold 1e-15

Device Selection:
  cpu  : Use CPU (recommended for <50k TOAs, <20 parameters) [DEFAULT]
  gpu  : Use GPU (recommended for >100k TOAs or >20 parameters)
  auto : Automatic selection based on problem size
  
Environment Variable:
  JUG_DEVICE : Override device selection (cpu, gpu, auto)
  Example: export JUG_DEVICE=gpu
        """
    )
    
    # Positional arguments
    parser.add_argument('par_file', type=str, nargs='?',
                       help='.par file with timing model')
    parser.add_argument('tim_file', type=str, nargs='?',
                       help='.tim file with TOAs')
    
    # Fit parameters
    parser.add_argument('--fit', nargs='+',
                       metavar='PARAM',
                       help='Parameters to fit (e.g., F0 F1 DM)')
    
    # Device selection
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu', 'auto'],
                       default=None,
                       help='Device for computation (default: cpu for typical timing)')
    
    # Fitting options
    parser.add_argument('--max-iter', type=int, default=25,
                       help='Maximum fitting iterations (default: 25)')
    parser.add_argument('--threshold', type=float, default=1e-14,
                       help='Convergence threshold (default: 1e-14)')
    
    # Clock files
    parser.add_argument('--clock-dir', type=str, default='data/clock',
                       help='Directory with clock files (default: data/clock)')
    
    # Output
    parser.add_argument('--output', '-o', type=str,
                       help='Output .par file with fitted parameters')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--plot', action='store_true',
                       help='Generate residual plot (prefit and postfit, or just prefit if no fitting)')
    
    # Device info
    parser.add_argument('--show-devices', action='store_true',
                       help='Show available devices and exit')
    
    args = parser.parse_args()
    
    # Show devices if requested
    if args.show_devices:
        print_device_info(verbose=True)
        return 0
    
    # Validate required arguments
    if not args.par_file or not args.tim_file:
        parser.error("par_file and tim_file are required unless using --show-devices")
        return 1
    
    # If --fit not specified, run in no-fit mode (just compute prefit residuals)
    no_fit_mode = (args.fit is None or len(args.fit) == 0)
    
    # Validate files exist
    par_file = Path(args.par_file)
    tim_file = Path(args.tim_file)
    
    if not par_file.exists():
        print(f"Error: Par file not found: {par_file}", file=sys.stderr)
        return 1
    
    if not tim_file.exists():
        print(f"Error: Tim file not found: {tim_file}", file=sys.stderr)
        return 1
    
    # Set device preference if specified
    if args.device:
        set_device_preference(args.device)
    
    # Run fitting or just compute residuals
    if not args.quiet:
        print("="*80)
        if no_fit_mode:
            print("JUG RESIDUAL COMPUTATION (NO FITTING)")
        else:
            print("JUG PARAMETER FITTING")
        print("="*80)
        print(f"\nPar file: {par_file}")
        print(f"Tim file: {tim_file}")
        
        if no_fit_mode:
            print(f"Mode: Prefit residuals only")
        else:
            print(f"Fit parameters: {', '.join(args.fit)}")
        
        if args.device:
            print(f"Device preference: {args.device}")
    
    try:
        if no_fit_mode:
            # No-fit mode: Just compute prefit residuals
            from jug.residuals.simple_calculator import compute_residuals_simple
            result = compute_residuals_simple(
                par_file=par_file,
                tim_file=tim_file,
                clock_dir=args.clock_dir,
                verbose=not args.quiet  # Use verbose unless --quiet specified
            )
            # Reformat to match fitter output
            orig_result = result  # Keep reference to original
            result = {
                'prefit_rms': orig_result['rms_us'],
                'final_rms': orig_result['rms_us'],
                'prefit_residuals_us': orig_result['residuals_us'],
                'postfit_residuals_us': None,  # No postfit in no-fit mode
                'tdb_mjd': orig_result['tdb_mjd'],
                'errors_us': orig_result['errors_us'],  # Add uncertainties
                'final_params': {},
                'uncertainties': {},
                'iterations': 0,
                'converged': True,
                'total_time': 0,
                'cache_time': 0,
                'jit_time': 0,
                'no_fit_mode': True
            }
        else:
            # Normal fitting mode
            result = fit_parameters_optimized(
                par_file=par_file,
                tim_file=tim_file,
                fit_params=args.fit,
                max_iter=args.max_iter,
                convergence_threshold=args.threshold,
                clock_dir=args.clock_dir,
                verbose=not args.quiet,
                device=args.device
            )
            result['no_fit_mode'] = False
    except Exception as e:
        print(f"\nError during fitting: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Print results
    if not args.quiet:
        print("\n" + "="*80)
        print("FITTING COMPLETE")
        print("="*80)
    
    print(f"\nFitted parameters:")
    for param, value in result['final_params'].items():
        unc = result['uncertainties'].get(param, 0.0)
        if param.startswith('F'):
            # Format for frequency parameters
            if param == 'F0':
                print(f"  {param} = {value:.15e} ± {unc:.3e} Hz")
            else:
                print(f"  {param} = {value:.15e} ± {unc:.3e} Hz/s^{int(param[1:])}")
        else:
            print(f"  {param} = {value:.15e} ± {unc:.3e}")
    
    print(f"\nFit quality:")
    print(f"  RMS: {result['final_rms']:.6f} μs")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    
    if not args.quiet:
        print(f"\nPerformance:")
        print(f"  Cache time: {result['cache_time']:.3f} s")
        print(f"  JIT time: {result['jit_time']:.3f} s")
        print(f"  Total time: {result['total_time']:.3f} s")
    
    # Write output file if requested
    if args.output:
        output_path = Path(args.output)
        
        # Read original par file
        params_orig = parse_par_file(par_file)
        
        # Update with fitted values
        params_orig.update(result['final_params'])
        
        # Write new par file
        with open(output_path, 'w') as f:
            # Copy original file, updating fitted parameters
            with open(par_file) as orig:
                for line in orig:
                    parts = line.split()
                    if parts and parts[0] in result['final_params']:
                        # Write fitted value
                        param = parts[0]
                        value = result['final_params'][param]
                        unc = result['uncertainties'].get(param, 0.0)
                        
                        # Try to preserve format
                        if param.startswith('F'):
                            if param == 'F0':
                                f.write(f"{param} {value:.20e} 1 {unc:.3e}\n")
                            else:
                                f.write(f"{param} {value:.20e} 1 {unc:.3e}\n")
                        else:
                            f.write(f"{param} {value:.15e} 1 {unc:.3e}\n")
                    else:
                        # Copy line as-is
                        f.write(line)
        
        print(f"\nFitted parameters written to: {output_path}")
    
    # Generate plot if requested
    if args.plot:
        import matplotlib.pyplot as plt
        
        tdb_mjd = result['tdb_mjd']
        errors_us = result.get('errors_us', None)  # TOA uncertainties
        
        if result.get('no_fit_mode', False):
            # No-fit mode: Just plot prefit residuals
            fig, ax = plt.subplots(figsize=(12, 6))
            
            residuals_us = result['prefit_residuals_us']
            rms = result['prefit_rms']
            
            if errors_us is not None:
                ax.errorbar(tdb_mjd, residuals_us, yerr=errors_us, fmt='o', 
                           markersize=2, alpha=0.5, elinewidth=0.5, capsize=0,
                           label='Prefit residuals')
            else:
                ax.plot(tdb_mjd, residuals_us, 'o', markersize=2, alpha=0.5, label='Prefit residuals')
            
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Time (MJD, TDB)', fontsize=12)
            ax.set_ylabel('Timing Residual (μs)', fontsize=12)
            ax.set_title(f'Timing Residuals (No Fitting)\nWeighted RMS = {rms:.3f} μs', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_file = str(par_file).replace('.par', '_residuals.png')
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            print(f"\nPlot saved to: {plot_file}")
            
        else:
            # Fitting mode: Plot prefit and postfit
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Prefit
            prefit_res = result['prefit_residuals_us']
            prefit_rms = result['prefit_rms']
            
            if errors_us is not None:
                ax1.errorbar(tdb_mjd, prefit_res, yerr=errors_us, fmt='o', 
                            markersize=2, alpha=0.5, color='red', elinewidth=0.5, 
                            capsize=0, label='Prefit')
            else:
                ax1.plot(tdb_mjd, prefit_res, 'o', markersize=2, alpha=0.5, color='red', label='Prefit')
            
            ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax1.set_ylabel('Timing Residual (μs)', fontsize=12)
            ax1.set_title(f'Prefit Residuals\nWeighted RMS = {prefit_rms:.3f} μs', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Postfit
            postfit_res = result['postfit_residuals_us']
            postfit_rms = result['final_rms']
            
            if errors_us is not None:
                ax2.errorbar(tdb_mjd, postfit_res, yerr=errors_us, fmt='o', 
                            markersize=2, alpha=0.5, color='blue', elinewidth=0.5, 
                            capsize=0, label='Postfit')
            else:
                ax2.plot(tdb_mjd, postfit_res, 'o', markersize=2, alpha=0.5, color='blue', label='Postfit')
            
            ax2.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.set_xlabel('Time (MJD, TDB)', fontsize=12)
            ax2.set_ylabel('Timing Residual (μs)', fontsize=12)
            ax2.set_title(f'Postfit Residuals\nWeighted RMS = {postfit_rms:.3f} μs', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plot_file = str(par_file).replace('.par', '_prefit_postfit.png')
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            print(f"\nPlot saved to: {plot_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
