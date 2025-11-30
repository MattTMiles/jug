"""
Example usage of the standalone TDB calculation module.

This script demonstrates how to use jug.tdb to compute TDB values
that match PINT exactly, without requiring PINT as a dependency.
"""

import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u
from pathlib import Path

# Import the TDB module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.jug.tdb import (
    parse_clock_file,
    interpolate_clock,
    compute_tdb,
    compute_tdb_batch,
    compute_tdb_with_clocks,
    validate_precision
)


def example_single_toa():
    """Example: Compute TDB for a single TOA."""
    print("=" * 70)
    print("Example 1: Single TOA")
    print("=" * 70)
    
    # Define observatory location (MeerKAT)
    mk_location = EarthLocation.from_geocentric(
        5109360.133, 2006852.586, -3238948.127, unit=u.m
    )
    
    # TOA: MJD 58526.213889148718147
    mjd_int = 58526
    mjd_frac = 0.213889148718147
    
    # Clock corrections (in seconds)
    bipm_corr = 27.674e-6  # ~27.7 µs
    mk_corr = 0.0          # MeerKAT observatory correction
    gps_corr = -3.0e-9     # ~-3 ns
    total_corr = bipm_corr + mk_corr + gps_corr
    
    # Compute TDB
    tdb_mjd = compute_tdb(mjd_int, mjd_frac, total_corr, mk_location)
    
    print(f"UTC MJD:          {mjd_int + mjd_frac:.15f}")
    print(f"Clock correction: {total_corr * 1e6:.3f} µs")
    print(f"TDB MJD:          {tdb_mjd:.15f}")
    print()


def example_with_clock_files():
    """Example: Compute TDB using clock files."""
    print("=" * 70)
    print("Example 2: Using Clock Files")
    print("=" * 70)
    
    # Define data directory
    data_dir = Path(__file__).parent.parent / "data" / "clock"
    
    # Load clock files
    print("Loading clock files...")
    bipm_clock = parse_clock_file(data_dir / "tai2tt_bipm2024.clk")
    mk_clock = parse_clock_file(data_dir / "mk2utc.clk")
    
    # For GPS, you might load from PINT cache or TEMPO2
    # For this example, we'll use an empty clock (no correction)
    gps_clock = {'mjd': np.array([]), 'offset': np.array([])}
    
    print(f"  BIPM clock: {len(bipm_clock['mjd'])} entries")
    print(f"  MK clock:   {len(mk_clock['mjd'])} entries")
    print()
    
    # Define observatory location
    mk_location = EarthLocation.from_geocentric(
        5109360.133, 2006852.586, -3238948.127, unit=u.m
    )
    
    # Example TOAs
    mjds = [
        (58526, 0.213889148718147),
        (59000, 0.5),
        (60000, 0.75)
    ]
    
    print("Computing TDB for example TOAs:")
    for mjd_int, mjd_frac in mjds:
        tdb = compute_tdb_with_clocks(
            mjd_int, mjd_frac,
            bipm_clock, mk_clock, gps_clock,
            mk_location
        )
        print(f"  MJD {mjd_int + mjd_frac:.6f} → TDB {tdb:.15f}")
    print()


def example_batch_processing():
    """Example: Batch processing of multiple TOAs."""
    print("=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    
    # Generate example TOAs
    n_toas = 1000
    mjd_ints = np.random.randint(58000, 61000, size=n_toas)
    mjd_fracs = np.random.uniform(0, 1, size=n_toas)
    
    # Clock corrections (could be interpolated from clock files)
    clock_corrs = np.random.uniform(-50e-6, 50e-6, size=n_toas)  # ±50 µs
    
    # Observatory location
    mk_location = EarthLocation.from_geocentric(
        5109360.133, 2006852.586, -3238948.127, unit=u.m
    )
    
    # Compute TDB for all TOAs
    print(f"Computing TDB for {n_toas} TOAs...")
    tdb_values = compute_tdb_batch(
        mjd_ints, mjd_fracs, clock_corrs,
        mk_location, show_progress=False
    )
    
    print(f"  First TDB:  {tdb_values[0]:.15f}")
    print(f"  Last TDB:   {tdb_values[-1]:.15f}")
    print(f"  Mean TDB:   {np.mean(tdb_values):.15f}")
    print()


def example_validation():
    """Example: Validate against reference values."""
    print("=" * 70)
    print("Example 4: Validation Against Reference")
    print("=" * 70)
    
    # Simulate computed and reference TDB values
    n_toas = 10000
    our_tdb = np.random.uniform(58000, 61000, size=n_toas)
    
    # Add small noise to simulate precision
    noise_ns = np.random.normal(0, 0.0001, size=n_toas)  # 0.1 ps RMS
    noise_ns[100:103] = 628.643  # Simulate 3 outliers
    reference_tdb = our_tdb + noise_ns / (86400e9)
    
    # Validate
    stats = validate_precision(our_tdb, reference_tdb, threshold_ns=0.001)
    
    print(f"Total TOAs:        {stats['n_total']}")
    print(f"Exact matches:     {stats['n_exact']} ({stats['percentage']:.4f}%)")
    print(f"Max difference:    {stats['max_diff_ns']:.6f} ns")
    print(f"Mean difference:   {stats['mean_diff_ns']:.6f} ns")
    print(f"Outliers:          {len(stats['outlier_indices'])}")
    
    if len(stats['outlier_indices']) > 0:
        print(f"Outlier indices:   {stats['outlier_indices'][:10]}")
        print(f"Outlier diffs:     {stats['outlier_diffs_ns'][:10]} ns")
    print()


if __name__ == "__main__":
    print("\n")
    print("JUG Standalone TDB Calculation - Examples")
    print("=" * 70)
    print()
    
    example_single_toa()
    example_with_clock_files()
    example_batch_processing()
    example_validation()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
