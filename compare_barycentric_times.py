"""Compare barycentric time calculation between JUG and PINT for both pulsars."""

import numpy as np
import matplotlib.pyplot as plt
from pint.models import get_model
from pint.toa import get_TOAs

# Test both pulsars
test_cases = [
    {
        'name': 'J1012-4235 (DD)',
        'par': '/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par',
        'tim': '/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim',
    },
    {
        'name': 'J1909-3744 (ELL1)',
        'par': '/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744_tdb.par',
        'tim': '/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim',
    }
]

print("="*80)
print("BARYCENTRIC TIME COMPARISON")
print("="*80)

for case in test_cases:
    print(f"\n{case['name']}")
    print("-" * 80)

    # Load PINT model
    model = get_model(case['par'])
    toas = get_TOAs(case['tim'], planets=True)

    # Get times
    mjds_utc = toas.get_mjds().value
    tdb_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

    # Get barycentric TOAs (after all delays except binary)
    bary_toas = model.get_barycentric_toas(toas)
    bary_mjd_pint = np.array([float(t) for t in bary_toas], dtype=np.float64)

    # Compute the total delay before binary
    delay_before_binary_days = bary_mjd_pint - tdb_pint
    delay_before_binary_us = delay_before_binary_days * 86400 * 1e6

    # Time since first TOA
    dt_years = (mjds_utc - mjds_utc[0]) / 365.25

    # Analyze trend
    coeffs = np.polyfit(dt_years, delay_before_binary_us, 1)
    linear_trend = coeffs[0] * dt_years + coeffs[1]
    detrended = delay_before_binary_us - linear_trend

    print(f"\nDelay before binary (Roemer + Einstein + Shapiro):")
    print(f"  Mean: {np.mean(delay_before_binary_us):.3f} μs")
    print(f"  RMS:  {np.std(delay_before_binary_us):.3f} μs")
    print(f"  Range: [{np.min(delay_before_binary_us):.3f}, {np.max(delay_before_binary_us):.3f}] μs")

    print(f"\nLinear trend in delay:")
    print(f"  Slope: {coeffs[0]:.6f} μs/year")
    print(f"  Offset: {coeffs[1]:.3f} μs")
    print(f"  Detrended RMS: {np.std(detrended):.3f} μs")

    # Check if binary component exists
    binary_comps = model.get_components_by_category().get('pulsar_system', [])
    if binary_comps:
        binary_comp = binary_comps[0]
        binary_comp.update_binary_object(toas, None)

        # Get what PINT uses for binary time
        binary_bary_time = binary_comp.barycentric_time
        binary_bary_mjd = np.array([float(t.to_value('day')) for t in binary_bary_time])

        # This should equal bary_mjd_pint
        diff_mjd = binary_bary_mjd - bary_mjd_pint
        diff_us = diff_mjd * 86400 * 1e6

        print(f"\nBinary barycentric time check:")
        print(f"  Diff (binary_bary - bary_toas): {np.mean(diff_us):.9f} ± {np.std(diff_us):.9f} μs")
        print(f"  Max diff: {np.max(np.abs(diff_us)):.9f} μs")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nThe delay before binary includes:")
print("  1. Roemer delay (geometric light travel time)")
print("  2. Einstein delay (gravitational time dilation)")
print("  3. Shapiro delay (solar system)")
print("\nA linear trend in this delay suggests:")
print("  - Proper motion errors")
print("  - Or systematic error in Roemer/Einstein calculation")
print("  - Or time scale conversion issue")
