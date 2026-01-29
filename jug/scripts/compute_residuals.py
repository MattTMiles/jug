#!/usr/bin/env python3
"""Command-line interface for computing pulsar timing residuals with JUG.

This script provides a simple interface to compute timing residuals from
.par and .tim files using the JUG timing package.
"""

import argparse
import sys
from pathlib import Path

# Configure JAX compilation cache EARLY (before any JIT functions are called)
from jug.utils.jax_cache import configure_jax_compilation_cache
configure_jax_compilation_cache()

# Configure Astropy for deterministic behavior (before any Astropy imports)
from jug.utils.astropy_config import configure_astropy
configure_astropy()

# Enable JAX 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

from jug.residuals.simple_calculator import compute_residuals_simple


def main():
    """Main entry point for jug-compute-residuals CLI."""
    parser = argparse.ArgumentParser(
        description="Compute pulsar timing residuals using JUG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  jug-compute-residuals J1909-3744.par J1909-3744.tim

  # Generate residual plot
  jug-compute-residuals J1909-3744.par J1909-3744.tim --plot

  # Specify clock directory and output location
  jug-compute-residuals J1909-3744.par J1909-3744.tim --clock-dir /path/to/clocks --plot --output-dir ./plots

  # Specify observatory
  jug-compute-residuals J1909-3744.par J1909-3744.tim --observatory parkes
        """
    )

    parser.add_argument(
        "par_file",
        type=Path,
        help="Path to .par file with timing model parameters"
    )

    parser.add_argument(
        "tim_file",
        type=Path,
        help="Path to .tim file with TOAs"
    )

    parser.add_argument(
        "--clock-dir",
        type=Path,
        default="data/clock",
        help="Directory containing clock files (default: data/clock)"
    )

    parser.add_argument(
        "--observatory",
        type=str,
        default="meerkat",
        help="Observatory name (default: meerkat)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate residual plot (saves as <pulsar>_residuals.png)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output plot (default: current directory)"
    )

    args = parser.parse_args()

    # Validate input files
    if not args.par_file.exists():
        print(f"Error: .par file not found: {args.par_file}", file=sys.stderr)
        sys.exit(1)

    if not args.tim_file.exists():
        print(f"Error: .tim file not found: {args.tim_file}", file=sys.stderr)
        sys.exit(1)

    # Compute residuals
    try:
        result = compute_residuals_simple(
            par_file=args.par_file,
            tim_file=args.tim_file,
            clock_dir=args.clock_dir,
            observatory=args.observatory
        )

        # Summary output
        if not args.verbose:
            print(f"\nResults for {args.par_file.stem}:")
            print(f"  Weighted RMS:   {result['rms_us']:8.3f} μs")
            print(f"  Mean:           {result['mean_us']:8.3f} μs")
            print(f"  N_TOAs:         {result['n_toas']:8d}")

        # Generate plot if requested
        if args.plot:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                
                # Create figure with residuals vs time and histogram
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Get data for plotting
                from jug.io.tim_reader import parse_tim_file_mjds
                toas = parse_tim_file_mjds(args.tim_file)
                mjds = [t.mjd_int + t.mjd_frac for t in toas]
                errors_us = [t.error_us for t in toas]
                residuals_us = result['residuals_us']
                
                # Top: Residuals vs MJD with error bars
                ax1.errorbar(mjds, residuals_us, yerr=errors_us, fmt='o', 
                            alpha=0.6, markersize=3, elinewidth=0.5, 
                            capsize=0, color='blue', ecolor='blue')
                ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
                ax1.set_xlabel('MJD', fontsize=12)
                ax1.set_ylabel('Residual (μs)', fontsize=12)
                ax1.set_title(f'{args.par_file.stem} - Timing Residuals (Weighted RMS={result["rms_us"]:.3f} μs)', 
                             fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Bottom: Histogram
                ax2.hist(residuals_us, bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax2.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
                ax2.set_xlabel('Residual (μs)', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title(f'Distribution (Mean={result["mean_us"]:.3f} μs, N={result["n_toas"]})', 
                             fontsize=12)
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                # Save plot
                output_file = args.output_dir / f"{args.par_file.stem}_residuals.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Plot saved to: {output_file}")
                
            except ImportError:
                print("Warning: matplotlib not available, skipping plot generation", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to generate plot: {e}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error computing residuals: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
