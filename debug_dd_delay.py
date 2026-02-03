#!/usr/bin/env python
"""Debug script to compare JUG DD binary delay with PINT's implementation.

This script computes the DD binary delay at specific times using both
JUG and PINT, and identifies where they diverge.
"""

import numpy as np
import jax.numpy as jnp

# JUG imports
from jug.delays.binary_dd import dd_binary_delay, solve_kepler
from jug.utils.constants import SECS_PER_DAY

# PINT imports for comparison
import pint.models as pm
from pint.models.binary_dd import BinaryDD

# Parameters from J1713+0747 par file
params = {
    'PB': 67.825133454705601975,  # days
    'T0': 54303.634565063208914,   # MJD
    'A1': 32.342422258041301913,   # light-seconds
    'OM': 176.20070970009467573,   # degrees
    'ECC': 7.4937105388132851258e-05,
    'SINI': 0.94407439403445780001,
    'M2': 0.30427453004974552999,  # solar masses
    'OMDOT': 0.00010227746700280294005,  # deg/yr
    'PBDOT': 9.1031328852562099998e-15,  # dimensionless
    'XDOT': 5.8165507334348638132e-15,  # lt-s/s
    'GAMMA': 0.0,
    'EDOT': 0.0,
}

def test_jug_dd_delay():
    """Test JUG's DD delay at a single time."""

    # Test time (middle of observation span)
    t_test = 58950.0  # MJD

    print("=" * 70)
    print("DD Binary Delay Comparison: JUG vs PINT")
    print("=" * 70)
    print(f"\nTest time: MJD {t_test}")
    print(f"T0: MJD {params['T0']}")
    print(f"PB: {params['PB']} days")
    print(f"A1: {params['A1']} lt-s")
    print(f"ECC: {params['ECC']}")
    print(f"OM: {params['OM']} deg")
    print()

    # Compute JUG delay
    jug_delay = dd_binary_delay(
        t_bary_mjd=t_test,
        pb_days=params['PB'],
        a1_lt_sec=params['A1'],
        ecc=params['ECC'],
        omega_deg=params['OM'],
        t0_mjd=params['T0'],
        gamma_sec=params['GAMMA'],
        pbdot=params['PBDOT'],
        omdot_deg_yr=params['OMDOT'],
        xdot=params['XDOT'],
        edot=params['EDOT'],
        sini=params['SINI'],
        m2_msun=params['M2'],
    )

    print(f"JUG DD delay: {float(jug_delay):.12f} seconds")
    print(f"             {float(jug_delay) * 1e6:.6f} microseconds")

    # Now let's trace through the JUG calculation step by step
    print("\n" + "-" * 70)
    print("Step-by-step JUG calculation:")
    print("-" * 70)

    t_bary_mjd = t_test
    t0_mjd = params['T0']
    pb_days = params['PB']
    ecc = params['ECC']
    omega_deg = params['OM']
    a1_lt_sec = params['A1']
    omdot_deg_yr = params['OMDOT']
    pbdot = params['PBDOT']
    gamma_sec = params['GAMMA']
    xdot = params['XDOT']
    edot = params['EDOT']
    sini = params['SINI']
    m2_msun = params['M2']

    # Time since periastron
    dt_days = t_bary_mjd - t0_mjd
    dt_sec = dt_days * SECS_PER_DAY
    print(f"\ndt_days = t - T0 = {dt_days:.6f} days")
    print(f"dt_sec = {dt_sec:.3f} seconds")

    # Number of orbits
    n_orbits = dt_days / pb_days
    print(f"\nNumber of orbits: {n_orbits:.6f}")

    # Periastron advance
    dt_years = dt_days / 365.25
    omega_current_deg = omega_deg + omdot_deg_yr * dt_years
    omega_rad = np.deg2rad(omega_current_deg)
    print(f"\nOmega at t: {omega_current_deg:.9f} deg")
    print(f"          = {omega_rad:.12f} rad")

    # Mean anomaly (PINT-style formula)
    pb_sec = pb_days * SECS_PER_DAY
    tt0 = dt_sec
    orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
    norbits = np.floor(orbits)
    frac_orbits = orbits - norbits
    mean_anomaly = frac_orbits * 2.0 * np.pi

    print(f"\nMean anomaly calculation:")
    print(f"  tt0/pb_sec = {tt0/pb_sec:.12f}")
    print(f"  PBDOT term = {0.5 * pbdot * (tt0/pb_sec)**2:.15e}")
    print(f"  orbits = {orbits:.12f}")
    print(f"  frac_orbits = {frac_orbits:.12f}")
    print(f"  mean_anomaly = {mean_anomaly:.12f} rad")
    print(f"               = {np.rad2deg(mean_anomaly):.6f} deg")

    # Solve Kepler's equation
    E = float(solve_kepler(jnp.array(mean_anomaly), ecc))
    print(f"\nEccentric anomaly E = {E:.12f} rad")
    print(f"                    = {np.rad2deg(E):.6f} deg")

    # Apply secular changes
    a1_current = a1_lt_sec + xdot * dt_sec
    ecc_current = ecc + edot * dt_sec
    print(f"\na1_current = {a1_current:.15f} lt-s")
    print(f"ecc_current = {ecc_current:.15e}")

    # Trigonometric functions
    sinE = np.sin(E)
    cosE = np.cos(E)
    sinOm = np.sin(omega_rad)
    cosOm = np.cos(omega_rad)

    print(f"\nTrigonometric values:")
    print(f"  sin(E) = {sinE:.12f}")
    print(f"  cos(E) = {cosE:.12f}")
    print(f"  sin(ω) = {sinOm:.12f}")
    print(f"  cos(ω) = {cosOm:.12f}")

    # Alpha and Beta (D&D equations [46], [47])
    er = ecc_current
    eTheta = ecc_current
    alpha = a1_current * sinOm
    beta = a1_current * np.sqrt(1.0 - eTheta**2) * cosOm

    print(f"\nAlpha/Beta parameters:")
    print(f"  alpha = a1 * sin(ω) = {alpha:.12f} lt-s")
    print(f"  beta = a1 * sqrt(1-e²) * cos(ω) = {beta:.12f} lt-s")

    # Roemer delay
    delayR = alpha * (cosE - er) + beta * sinE
    print(f"\nRoemer delay = alpha*(cos(E)-e) + beta*sin(E)")
    print(f"             = {alpha:.6f} * ({cosE:.9f} - {er:.9e}) + {beta:.6f} * {sinE:.9f}")
    print(f"             = {delayR:.12f} s")
    print(f"             = {delayR * 1e6:.6f} μs")

    # Einstein delay
    delayE = gamma_sec * sinE
    print(f"\nEinstein delay = GAMMA * sin(E) = {delayE:.12e} s")

    # Combined Dre
    Dre = delayR + delayE
    print(f"\nDre (Roemer + Einstein) = {Dre:.12f} s")

    # First and second derivatives for inverse delay transformation
    Drep = -alpha * sinE + (beta + gamma_sec) * cosE
    Drepp = -alpha * cosE - (beta + gamma_sec) * sinE

    # nhat = dE/dt
    pb_prime_sec = pb_sec + pbdot * tt0
    nhat = (2.0 * np.pi / pb_prime_sec) / (1.0 - ecc_current * cosE)

    print(f"\nInverse delay transformation parameters:")
    print(f"  Drep = {Drep:.12f}")
    print(f"  Drepp = {Drepp:.12f}")
    print(f"  nhat = {nhat:.15e} rad/s")

    # Inverse delay transformation (D&D eq [52])
    correction_factor = (
        1.0
        - nhat * Drep
        + (nhat * Drep)**2
        + 0.5 * nhat**2 * Dre * Drepp
        - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat**2 * Dre * Drep
    )

    print(f"\nCorrection factor components:")
    print(f"  1.0")
    print(f"  - nhat * Drep = {-nhat * Drep:.15e}")
    print(f"  + (nhat * Drep)² = {(nhat * Drep)**2:.15e}")
    print(f"  + 0.5 * nhat² * Dre * Drepp = {0.5 * nhat**2 * Dre * Drepp:.15e}")
    last_term = -0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat**2 * Dre * Drep
    print(f"  - last term = {last_term:.15e}")
    print(f"  Total correction = {correction_factor:.15f}")

    delayInverse = Dre * correction_factor
    print(f"\ndelayInverse = Dre * correction = {delayInverse:.12f} s")

    # Shapiro delay
    T_SUN = 4.925490947e-6
    shapiro_arg = (
        1.0
        - ecc_current * cosE
        - sini * (sinOm * (cosE - ecc_current) + np.sqrt(1.0 - ecc_current**2) * cosOm * sinE)
    )
    delayS = -2.0 * T_SUN * m2_msun * np.log(shapiro_arg)

    print(f"\nShapiro delay:")
    print(f"  r = T_sun * M2 = {T_SUN * m2_msun:.15e} s")
    print(f"  shapiro_arg = {shapiro_arg:.15f}")
    print(f"  delayS = -2r * ln(arg) = {delayS:.12f} s")
    print(f"         = {delayS * 1e6:.6f} μs")

    # Total delay
    total = delayInverse + delayS
    print(f"\n" + "=" * 70)
    print(f"TOTAL DD DELAY:")
    print(f"  Inverse (Roemer+Einstein corrected): {delayInverse:.12f} s = {delayInverse*1e6:.6f} μs")
    print(f"  Shapiro:                             {delayS:.12f} s = {delayS*1e6:.6f} μs")
    print(f"  TOTAL:                               {total:.12f} s = {total*1e6:.6f} μs")
    print("=" * 70)

    return float(jug_delay)


def test_pint_dd_delay():
    """Test PINT's DD delay for comparison."""

    print("\n" + "=" * 70)
    print("PINT DD Delay Calculation")
    print("=" * 70)

    # Load PINT model
    par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.par"
    tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.tim"

    try:
        from pint.models import get_model
        from pint.toas import get_TOAs
        import pint.residuals as pr

        model = get_model(par_file)

        # Get TOAs
        toas = get_TOAs(tim_file, model=model)

        # Get pre-fit residuals from PINT
        residuals = pr.Residuals(toas, model)
        pint_resids = residuals.time_resids.to('us').value

        print(f"\nPINT pre-fit residuals:")
        print(f"  Mean: {np.mean(pint_resids):.3f} μs")
        print(f"  RMS: {np.std(pint_resids):.3f} μs")
        print(f"  Weighted RMS: {residuals.rms_weighted().to('us').value:.3f} μs")

        # Get the binary delay component specifically
        # PINT stores the computed delays in the model
        binary_component = model.components.get('BinaryDD', None)
        if binary_component:
            print(f"\nPINT Binary component found: {binary_component}")

            # Get the barycentered times
            tdbld = toas.get_mjds()

            # Compute binary delay at first TOA
            t_test = float(tdbld[0])
            print(f"\nFirst TOA time: MJD {t_test}")

            # Try to get individual delay
            # PINT computes delays via delay() method
            from astropy.time import Time
            from astropy import units as u

            # The binary model uses the TOAs directly
            # Let's compute residuals for a subset
            binary_delays = model.binarymodel_delay(toas, None)

            print(f"\nPINT binary delays (first 5):")
            for i in range(min(5, len(binary_delays))):
                print(f"  TOA {i}: {binary_delays[i].to('us').value:.6f} μs")

            return binary_delays, pint_resids

    except Exception as e:
        print(f"\nError computing PINT delays: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_at_multiple_times():
    """Compare JUG and PINT at multiple orbital phases."""

    print("\n" + "=" * 70)
    print("Comparing JUG vs PINT DD delays at multiple orbital phases")
    print("=" * 70)

    # Test at different orbital phases
    pb = params['PB']
    t0 = params['T0']

    # Test times spanning one orbit from a recent epoch
    t_start = 58950.0
    phases = np.linspace(0, 1, 9)  # 0, 0.125, 0.25, ..., 1.0
    test_times = t_start + phases * pb

    print(f"\nOrbital period: {pb:.6f} days")
    print(f"Test epoch: MJD {t_start}")
    print()

    jug_delays = []
    for i, t in enumerate(test_times):
        delay = dd_binary_delay(
            t_bary_mjd=t,
            pb_days=params['PB'],
            a1_lt_sec=params['A1'],
            ecc=params['ECC'],
            omega_deg=params['OM'],
            t0_mjd=params['T0'],
            gamma_sec=params['GAMMA'],
            pbdot=params['PBDOT'],
            omdot_deg_yr=params['OMDOT'],
            xdot=params['XDOT'],
            edot=params['EDOT'],
            sini=params['SINI'],
            m2_msun=params['M2'],
        )
        jug_delays.append(float(delay))
        print(f"Phase {phases[i]:.3f}: MJD {t:.3f}, JUG delay = {float(delay)*1e6:12.6f} μs")

    return test_times, jug_delays


if __name__ == "__main__":
    # Run step-by-step JUG calculation
    jug_result = test_jug_dd_delay()

    # Compare at multiple phases
    times, delays = compare_at_multiple_times()

    # Compare with PINT
    pint_binary, pint_resids = test_pint_dd_delay()
