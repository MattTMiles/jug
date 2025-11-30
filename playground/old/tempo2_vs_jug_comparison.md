# Tempo2 vs JUG: Step-by-Step Residual Calculation Comparison

## Tempo2's Calculation Flow (from documentation and outputs)

### Step-by-Step Process:

1. **Input**: Topocentric TOA (from .tim file)
   - Format: filename freq(MHz) MJD error(μs) observatory

2. **Clock Corrections**: Observatory → UTC → TAI → TT
   - Uses clock files in tempo2 format
   - Chain: obs_clock → UTC → GPS/TAI → TT (BIPM)

3. **Einstein Delay**: TT → TDB (or TCB if UNITS TCB)
   - Gravitational time dilation
   - ~93 μs mean, ~1200 μs std from temp_pre_components_next.out

4. **Geometric (Roemer) Delay**: Site position → SSB
   - Light travel time from observatory to Solar System Barycenter
   - Uses JPL ephemeris (DE440)
   - Accounts for proper motion, parallax
   - ~45.7 ms mean from temp_pre_components_next.out

5. **Shapiro Delay**: Solar system bodies
   - Gravitational time delay from Sun, planets
   - ~5.8 μs mean from temp_pre_components_next.out

6. **Barycentric Arrival Time (BAT)**:
   ```
   BAT = t_topo_TDB + geometric_delay + shapiro_delay
   ```
   - This is what's in temp_pre_components_next.out column 2

7. **Binary Delays**: Orbital motion of pulsar
   - Roemer delay (a1 * sin(orbital_phase))
   - Einstein delay (γ * sin(orbital_phase))
   - Shapiro delay from companion (-2*r*log(1-s*sin(phase)))
   - For J1909-3744: mean ~-7.7 ms, std ~1.38 s

8. **Emission Time**:
   ```
   t_emission = BAT - binary_delay
   ```

9. **DM Delay** (frequency-dependent):
   ```
   dm_delay = K_DM * DM(t) / freq²
   K_DM = 4.148808 ms (when freq in MHz, DM in pc/cm³)
   ```

10. **Infinite-Frequency Time**:
    ```
    t_inf = t_emission - dm_delay
    ```

11. **FD Delays** (profile evolution):
    ```
    fd_delay = Σ FDn * (log(f_GHz))^n
    ```

12. **Spin Phase**:
    ```
    dt = t_inf - PEPOCH
    phase = F0*dt + 0.5*F1*dt² + (1/6)*F2*dt³
    ```

13. **TZR Phase Reference**:
    - Compute phase at TZRMJD (after all delays)
    - Round to nearest integer pulse
    - This sets the "zero" of the residuals

14. **Phase Residual**:
    ```
    phase_resid = phase(t_inf) - phase(TZR) - TZRPHASE
    frac_phase = mod(phase_resid + 0.5, 1.0) - 0.5
    ```

15. **Time Residual**:
    ```
    residual = frac_phase / F0
    ```

## Analysis Needed

I need to check the following in JUG vs Tempo2:

### Check 1: Are we using BAT correctly?
- **Tempo2**: Provides BAT in temp_pre_components_next.out
- **JUG**: Should use this BAT, OR compute it from scratch

### Check 2: Binary delay calculation
- **Need to verify**: Is JUG computing binary delays the same way?
- **ELL1 model specifics**: a1, TASC, PB, EPS1, EPS2, GAMMA, PBDOT, XDOT

### Check 3: DM delay
- **Formula**: K_DM = 4148.808 ms or 4.148808e3 s?
- **Time evolution**: DM(t) = DM + DM1*(t-DMEPOCH) + DM2*(t-DMEPOCH)²

### Check 4: TZR calculation
- **Critical**: Must use emission time at TZR, not topocentric
- **Must include**: All delays (binary, DM, FD) at TZR epoch

### Check 5: Phase wrapping
- **Tempo2**: mod(phase + 0.5, 1.0) - 0.5
- **JUG**: Same?

### Check 6: Reference frame
- **UNITS TDB vs TCB**: Are F0, F1, F2 scaled correctly?
- **L_B = 1.550519768e-8**

## Next Steps

1. Extract one TOA from Tempo2 output
2. Trace through each step manually
3. Compare with JUG calculation at each step
4. Identify where they diverge
