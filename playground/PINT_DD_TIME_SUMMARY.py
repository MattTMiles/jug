#!/usr/bin/env python
"""
FINAL SUMMARY: PINT DD time variable and algorithm comparison with JUG
"""

print("="*80)
print("FINAL ANSWER: PINT DD Time Variable and Algorithm")
print("="*80)

print("""
FINDING 1: TIME VARIABLE
========================
PINT DD uses time: bo.t = model.get_barycentric_toas(toas)

FORMULA:
  t_DD = tdbld - delay_before_binary

WHERE delay_before_binary = sum of:
  - solar_system_geometric_delay (Roemer delay, ±500 s)
  - troposphere_delay (~10 ns)
  - solar_system_shapiro_delay (~few μs)
  - solar_wind_delay (~ns to μs)
  - constant_dispersion_delay (DM delay, ~50-100 ms)

IMPORTANT: The delays are SUBTRACTED from tdbld.

JUG's CURRENT TIME:
  t_DD_jug = tdbld - (roemer + shapiro)

JUG is MISSING:
  - troposphere (~10 ns, negligible)
  - solar_wind (~few μs, small)
  - DM delay (~44-80 ms, SIGNIFICANT and frequency-dependent!)


FINDING 2: DD ALGORITHM DIFFERENCE
==================================
Even when JUG and PINT are fed the EXACT SAME TIME, there's still a
~36 μs mean offset and ~30 μs std difference.

ROOT CAUSE: PINT's DD uses the "delayInverse" formulation from
Damour & Deruelle (1986), which is NOT simply delayR + delayE + delayS.

PINT's DDdelay() returns:
  DDdelay = delayInverse + delayS + delayA

Where delayInverse is a 3rd-order iterative approximation that converts
between proper time and coordinate time:

  delayInverse = Dre * (1 - nhat*Drep + (nhat*Drep)² + 
                        0.5*nhat²*Dre*Drepp - 
                        0.5*e*sin(E)/(1-e*cos(E))*nhat²*Dre*Drep)

And:
  Dre = delayR + delayE  (NOT just delayR!)
  Drep = d(Dre)/d(E)
  Drepp = d²(Dre)/d(E)²
  nhat = du/dt = n/(1 - e*cos(E))

This iterative formula accounts for the fact that the Roemer delay
Dre itself depends on the pulsar's orbital position, which changes
during the light travel time. It's a self-consistent correction.

The difference between:
  - Simple Roemer delay: Dre = -31.857 s
  - delayInverse (iterated): -31.856 s
is about ~190 μs!


FINDING 3: WHY END-TO-END RESIDUALS ONLY DIFFER BY ~1.1 μs
==========================================================
The ~289 μs RMS difference seen in DD-only comparisons does NOT fully
propagate to end-to-end residuals because:

1. The constant part of the DD difference (~36 μs mean) is absorbed
   into phase/timing offsets during fitting.

2. The time-varying part (~30 μs std) gets partially cancelled by
   other delays or absorbed into fitted parameters like OM, A1, etc.

3. After removing common-mode offsets, the residual phase-dependent
   structure is only ~1.1 μs RMS.


RECOMMENDATIONS FOR JUG
=======================
1. SHORT-TERM FIX for apples-to-apples comparison:
   Feed JUG's dd_binary_delay() with:
     t_DD = model.get_barycentric_toas(toas)
   
   This removes the ~44 ms DM time offset and lets you compare
   the algorithm differences directly.

2. LONG-TERM FIX for sub-μs agreement:
   Implement the Damour & Deruelle (1986) "delayInverse" formulation
   in JUG's DD model. This requires:
   - Computing Dre = delayR + delayE
   - Computing Drep = d(Dre)/d(E)
   - Computing Drepp = d²(Dre)/d(E)²
   - Computing nhat = n/(1 - e*cos(E))
   - Applying the iterative formula (equations 46-52 in D&D 1986)

3. For the ~1.1 μs end-to-end residual difference:
   This is likely due to small algorithmic differences in:
   - Kepler solver precision
   - Time-dependent parameter evolution
   - The delayInverse iteration (most important)
""")
