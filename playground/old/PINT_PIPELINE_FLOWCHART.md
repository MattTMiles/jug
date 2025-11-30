# PINT Pipeline Flowchart: .tim File to Residuals

## Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT FILES                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ .par file (Pulsar Parameters)    │  .tim file (Time of Arrivals)    │   │
│  │ - F0 (spin frequency)            │  - MJD (UTC times)              │   │
│  │ - F1, F2 (derivatives)           │  - Frequencies                  │   │
│  │ - DM (dispersion measure)        │  - Observatory codes            │   │
│  │ - PEPOCH (reference epoch)       │  - Uncertainties                │   │
│  │ - TZRMJD, TZRFRQ (TZR)          │  - Flags                        │   │
│  │ - Binary parameters (if any)     │                                 │   │
│  │ - Astrometry (RA, DEC, etc)     │                                 │   │
│  │ - EPHEM, CLOCK settings          │                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PARSE .par FILE                                                     │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Extract all timing model parameters:                                        │
│   • Spin: F0, F1, F2, PEPOCH                                               │
│   • Dispersion: DM, DM1, DM2, ...                                          │
│   • Binary orbital: PB, A1, ECC, T0, OM, GAMMA, ... (ELL1 or BT model)    │
│   • Astrometry: RA, DEC, PMRA, PMDEC, PX                                   │
│   • Time zero reference: TZRMJD, TZRFRQ, TZRSITE                           │
│   • Ephemeris: EPHEM (e.g., DE440)                                         │
│   • Clock: CLOCK (e.g., BIPM2024)                                          │
│                                                                              │
│ Output: TimingModel object with all parameters                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PARSE .tim FILE                                                     │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Extract raw TOA data:                                                       │
│   For each line in .tim file:                                              │
│     • MJD (Topocentric UTC time in MJD format)                             │
│     • Frequency (MHz)                                                       │
│     • Observatory code (e.g., "meerkat")                                   │
│     • Uncertainty (measurement error)                                       │
│     • Any additional flags or parameters                                    │
│                                                                              │
│ Output: TOA table with N rows (N = number of observations)                 │
│         Columns: [mjd, freq, obs, error, flags, ...]                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: APPLY CLOCK CORRECTIONS                                            │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Convert topocentric UTC → Terrestrial Time (TT)                             │
│                                                                              │
│ For each TOA:                                                               │
│   1. Topocentric UTC (from .tim file)                                       │
│   2. Add observatory clock correction                                       │
│      • Uses tempo2-format clock files (*.clk)                              │
│      • Observatory → UTC chain                                              │
│   3. UTC → GPS (if needed)                                                  │
│   4. GPS/UTC → TAI (International Atomic Time)                              │
│   5. TAI → TT (Terrestrial Time) using BIPM file                           │
│                                                                              │
│ Formula: TT = UTC + clock_correction(observatory, UTC)                      │
│                                                                              │
│ Output: TT times added to TOA table as new column                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: COMPUTE OBSERVATORY POSITIONS (SSB FRAME)                          │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Use JPL ephemeris (DE440) to get positions in Solar System Barycenter      │
│                                                                              │
│ For each TOA at time TT:                                                    │
│   1. Query JPL ephemeris kernel (de440s.bsp)                               │
│   2. Get Earth position relative to SSB                                     │
│   3. Get observatory position on Earth surface (from observatories.dat)     │
│   4. Compute: obs_pos_SSB = Earth_pos_SSB + obs_pos_Earth                  │
│                                                                              │
│ Output: ssb_obs_pos, ssb_obs_vel added to TOA table                        │
│         Shape: (N, 3) for position; (N, 3) for velocity                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: COMPUTE TDB TIMES                                                   │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Convert TT → TDB (Barycentric Dynamical Time)                               │
│                                                                              │
│ For each TOA:                                                               │
│   TDB = TT - Einstein_delay(pos, vel) - Shapiro_delay(pos)                │
│                                                                              │
│ Where:                                                                       │
│   • Einstein_delay: Gravitational time dilation effect                      │
│   • Shapiro_delay: Light bending by Sun's gravity                          │
│   • Both depend on observatory position from Step 4                         │
│                                                                              │
│ Output: tdb times added to TOA table as new column                          │
│         These are what model.phase() will use!                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: EVALUATE TIMING MODEL AT EACH TDB TIME                            │
│ ─────────────────────────────────────────────────────────────────────────   │
│ THIS IS THE CRITICAL STEP: model.phase(toas)                               │
│                                                                              │
│ The timing model computes:                                                  │
│   phase(t) = F0*(t - PEPOCH) + 0.5*F1*(t - PEPOCH)^2 + ... (base)         │
│                                                                              │
│ Then applies all model components in order:                                 │
│   1. Spindown: F0, F1, F2 (rotational phase)                               │
│   2. Astrometry: RA, DEC, PMRA, PMDEC, PX (pulsar position)               │
│      → Roemer delay: light travel time from observer to SSB                │
│      → Annual parallax effects                                              │
│   3. Binary orbital: PB, A1, ECC, T0, etc. (if binary)                     │
│      → Roemer, Einstein, Shapiro delays from companion                     │
│   4. Dispersion (DM): DM, DM1, DM2, ...                                    │
│      → Frequency-dependent delay: DM_delay = K_DM * DM / f^2              │
│   5. Solar wind: SWM (if present)                                           │
│   6. Shapiro delay: from Sun                                                │
│   7. Troposphere: local atmospheric delay (if present)                     │
│   8. FD delays: profile evolution (if present)                              │
│                                                                              │
│ Key insight: All these effects are evaluated at TDB time!                   │
│                                                                              │
│ Output: Phase object with:                                                  │
│   • .int: integer cycles (large numbers, carries cycle count)              │
│   • .frac: fractional cycles (0 to 1, the actual residuals!)              │
│   • .value: int + frac (full phase value)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: EXTRACT RESIDUALS                                                   │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Convert phase residuals (cycles) to time residuals (seconds/microseconds)   │
│                                                                              │
│ For each TOA:                                                               │
│   phase_frac = phase.frac[i]  (in cycles, from Step 6)                     │
│   residual_seconds = phase_frac / F0                                        │
│   residual_microseconds = residual_seconds * 1e6                            │
│                                                                              │
│ Where:                                                                       │
│   • F0 = spin frequency (Hz) from .par file                                │
│   • phase.frac = fractional part of phase (Step 6)                         │
│                                                                              │
│ These residuals represent how far each TOA is from the model prediction.    │
│ They're the quantity that gets minimized in fitting!                        │
│                                                                              │
│ Output: Residual array of length N (microseconds)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT: RESIDUALS                                 │
│                                                                              │
│ Array of N values (one per TOA) showing timing residuals in microseconds   │
│ These can be:                                                               │
│   • Positive: TOA arrived later than model predicts                        │
│   • Negative: TOA arrived earlier than model predicts                      │
│   • RMS ~0.8 μs for well-timed pulsar                                      │
│   • Used to: (1) diagnose timing problems                                   │
│              (2) detect binary companions                                    │
│              (3) measure pulsar masses                                       │
│              (4) search for gravitational waves                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Breakdown: What model.phase() Does

### Component 1: Spindown (Rotational Phase)
```
Base phase = F0 * (t - PEPOCH) + 0.5 * F1 * (t - PEPOCH)^2
           + (1/6) * F2 * (t - PEPOCH)^3

Where:
  F0 = spin frequency (Hz)
  F1 = frequency derivative (Hz/s)
  F2 = second derivative (Hz/s²)
  t = TDB time
  PEPOCH = reference epoch
  
This gives the pulsar's rotational phase as a function of time.
```

### Component 2: Astrometry (Pulsar Position Effects)
```
Roemer delay = pulsar_distance * cos(angle_to_observer) / c

Where:
  pulsar_distance = inferred from parallax (PX)
  angle_to_observer = angle between pulsar and observer position
  
This accounts for:
  • Light travel time to/from pulsar
  • Annual parallax as Earth orbits Sun
  • Proper motion of pulsar across sky
  
Changes delay by ~500 seconds over year (Earth's orbit)
```

### Component 3: Binary Orbital Delays (if Binary)
```
Binary_delay = Roemer + Einstein + Shapiro
            = (light travel) + (time dilation) + (light bending)

For ELL1 model:
  Uses orbital parameters: PB, A1, TASC, EPS1, EPS2
  
For BT/DD/DDGR model:
  Uses: PB, A1, ECC, OM, T0, GAMMA, PBDOT
  Solves Kepler's equation to find orbit position
  
Changes delay by up to ~100 seconds (orbital period)
```

### Component 4: Dispersion Measure (DM)
```
DM_delay_seconds = K_DM * DM / f^2

Where:
  K_DM = 4.148808e3 (dispersion constant)
  DM = dispersion measure (pc/cm³)
  f = observation frequency (MHz)
  
Example:
  At 1400 MHz: DM_delay ≈ 0.003 seconds per pc/cm³
  At 400 MHz:  DM_delay ≈ 0.026 seconds per pc/cm³
  
This is FREQUENCY-DEPENDENT!
That's why we need multiple frequencies to measure DM.
```

### Component 5: Shapiro Delay (Solar)
```
Shapiro_delay ≈ (2*M_sun / c^3) * ln(1 + cos(angle_to_sun))

Where:
  M_sun = Sun's mass
  c = speed of light
  angle_to_sun = angle between pulsar and Sun
  
Maximum: ~100 microseconds when Sun is near pulsar
Minimum: ~0 when Sun is far from line of sight
```

## Key Parameters and Their Roles

| Parameter | Unit | Purpose | Example |
|-----------|------|---------|---------|
| F0 | Hz | Spin frequency | 339.3 |
| F1 | Hz/s | Spin derivative (spin-down) | -1.6e-15 |
| DM | pc/cm³ | Dispersion measure | 10.4 |
| PEPOCH | MJD | Reference epoch for phase | 59017.999 |
| RA | degrees | Pulsar right ascension | 290.3 |
| DEC | degrees | Pulsar declination | -27.4 |
| PMRA | mas/yr | Proper motion (RA) | -3.1 |
| PMDEC | mas/yr | Proper motion (Dec) | -55.2 |
| PX | mas | Parallax (distance) | 2.5 |
| TZRMJD | MJD | Time zero reference | 59679.248 |
| TZRFRQ | MHz | Reference frequency for TZR | 1029 |
| EPHEM | - | Ephemeris to use | DE440 |
| CLOCK | - | Clock correction system | BIPM2024 |

## Data Flow Summary

```
.par file ──┐
            ├──→ TimingModel (parameters)
.tim file ──┤
            ├──→ TOA table (raw data)
            │
            v
        [Clock corrections] → TT times
            │
            v
        [JPL ephemeris] → Observatory positions (SSB)
            │
            v
        [Compute TDB] → TDB times
            │
            v
        [model.phase(toas)] ← Evaluate all components
            │                  at TDB times
            v
        Phase object (int, frac)
            │
            v
        [Extract residuals] → residuals = frac / F0
            │
            v
        Residuals (microseconds)
```

## What JUG Needs to Implement

For JUG to replicate PINT's pipeline:

1. **Parser for .par files**
   - Extract F0, F1, F2, DM, DM1, etc.
   - Extract astrometric parameters (RA, DEC, PMRA, PMDEC, PX)
   - Extract binary parameters (if present)
   - Extract EPHEM and CLOCK specifications

2. **Parser for .tim files**
   - Extract MJD (topocentric UTC)
   - Extract frequency (MHz)
   - Extract observatory code
   - Handle any additional flags

3. **Clock correction system**
   - Load tempo2-format clock files (*.clk)
   - Implement UTC → TAI → TT conversion chain
   - Apply observatory-specific corrections

4. **JPL Ephemeris integration**
   - Load DE440 (or other) ephemeris kernel
   - Query positions at given times
   - Compute Earth and observatory positions

5. **TDB time computation**
   - Implement Einstein delay (gravitational time dilation)
   - Implement Shapiro delay (light bending by Sun)
   - Formula: TDB = TT - Einstein - Shapiro

6. **Astrometric calculations**
   - Compute Roemer delay from pulsar position
   - Implement proper motion
   - Implement parallax

7. **Binary orbital calculations**
   - Implement ELL1 model (or BT if needed)
   - Solve Kepler equation for BT
   - Compute binary delays

8. **DM dispersion delay**
   - Formula: K_DM * DM / f²
   - Handle DM variations (DM1, DM2, etc.)

9. **Phase calculation**
   - Evaluate: F0*(t-PEPOCH) + 0.5*F1*(t-PEPOCH)² + ...
   - Apply all delay components in order
   - Return (integer cycles, fractional cycles)

10. **Residual extraction**
    - residuals = phase_frac / F0
    - Convert cycles to seconds/microseconds

## Implementation Strategy for JUG

Start with the simplest case and add complexity:

### Minimal working version:
- [ ] Parse .par and .tim files
- [ ] Apply simple clock corrections
- [ ] Compute TDB (basic)
- [ ] Evaluate F0*(t-PEPOCH)
- [ ] Extract residuals
- [ ] Compare against PINT output

### Add complexity progressively:
- [ ] Add F1 (spin-down)
- [ ] Add astrometry (Roemer delay)
- [ ] Add DM dispersion
- [ ] Add Shapiro delay (Sun)
- [ ] Add binary (if needed for this pulsar)

### Validate at each step:
- Run on test pulsar
- Compare residuals against PINT/Tempo2
- Ensure RMS and distribution match

This modular approach lets you incrementally verify correctness!
