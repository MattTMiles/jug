# JUG Implementation Guide: Replicating PINT's Pipeline

## Overview

This guide provides a step-by-step implementation plan for JUG to replicate PINT's residual computation pipeline. The goal is to have JUG compute timing residuals that match PINT's output (99.45% correlation demonstrated).

## High-Level Strategy

Instead of building from scratch, follow PINT's proven pipeline:

```
.par + .tim files
    ↓
[Phase 1] Parse inputs & prepare data
    ↓
[Phase 2] Time conversions & ephemeris
    ↓
[Phase 3] Evaluate timing model
    ↓
[Phase 4] Extract residuals
```

## Phase 1: Data Input & Preparation

### 1.1 Parse .par File

**Goal**: Extract all timing model parameters

**Key parameters to extract**:
```python
# Spin parameters
F0 = float  # Hz (fundamental spin frequency)
F1 = float  # Hz/s (spin-down rate, optional)
F2 = float  # Hz/s² (second derivative, optional)
PEPOCH = float  # MJD (reference epoch for phase)

# Astrometry
RA = float  # degrees (right ascension)
DEC = float  # degrees (declination)
PMRA = float  # mas/yr (proper motion in RA)
PMDEC = float  # mas/yr (proper motion in DEC)
PX = float  # mas (parallax, optional)

# Dispersion
DM = float  # pc/cm³ (dispersion measure)
DM1 = float  # pc/cm³/yr (DM rate, optional)
DM2 = float  # pc/cm³/yr² (DM acceleration, optional)

# Time zero reference
TZRMJD = float  # MJD (time zero reference MJD)
TZRFRQ = float  # MHz (time zero reference frequency)
TZRSITE = str  # observatory code (optional)

# Binary (if present)
BINARY = str  # "ELL1H" or "BT" or similar
PB = float  # days (orbital period)
A1 = float  # light-seconds (semi-major axis)
TASC = float  # MJD (time of ascending node, for ELL1)
ECC = float  # (eccentricity, for BT)
OM = float  # degrees (argument of periapsis, for BT)
T0 = float  # MJD (time of periapsis, for BT)
# ... and other binary parameters

# System settings
EPHEM = str  # "DE440" or similar
CLOCK = str  # "BIPM2024" or similar
```

**Implementation**:
```python
def parse_parfile(par_filename):
    """Parse .par file and return dict of parameters"""
    params = {}
    with open(par_filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                try:
                    params[key] = float(value)
                except:
                    params[key] = value
    return params
```

### 1.2 Parse .tim File

**Goal**: Extract TOA (Time of Arrival) data

**Key columns**:
```python
# For each TOA:
mjd = float  # Topocentric UTC time (MJD)
freq = float  # Observation frequency (MHz)
obs = str  # Observatory code (e.g., "meerkat")
error = float  # Measurement uncertainty (microseconds)
# Any additional flags or parameters
```

**Implementation**:
```python
def parse_timfile(tim_filename):
    """Parse .tim file and return list of TOAs"""
    toas = []
    with open(tim_filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('C ') or line.startswith('c '):
                continue
            # Simple format: freq mjd error obs
            parts = line.split()
            if len(parts) >= 4:
                toa = {
                    'freq': float(parts[0]),
                    'mjd': float(parts[1]),
                    'error': float(parts[2]),
                    'obs': parts[3],
                }
                toas.append(toa)
    return toas
```

### 1.3 Create Data Structures

**Timing model class**:
```python
@dataclass
class TimingModel:
    # Spin
    F0: float  # Hz
    F1: float = 0.0  # Hz/s
    F2: float = 0.0  # Hz/s²
    PEPOCH: float = None  # MJD
    
    # Astrometry
    RA: float = None  # degrees
    DEC: float = None  # degrees
    PMRA: float = 0.0  # mas/yr
    PMDEC: float = 0.0  # mas/yr
    PX: float = 0.0  # mas (distance)
    
    # Dispersion
    DM: float = 0.0  # pc/cm³
    DM1: float = 0.0  # pc/cm³/yr
    
    # TZR
    TZRMJD: float = None
    TZRFRQ: float = None
    
    # Binary (optional)
    BINARY: str = None
    PB: float = None
    # ... other binary params
    
    # System settings
    EPHEM: str = "DE440"
    CLOCK: str = "BIPM2024"
```

**TOA structure**:
```python
@dataclass
class TOA:
    mjd_utc: float  # Topocentric UTC (input)
    freq: float  # MHz
    obs: str  # Observatory code
    error: float  # Measurement uncertainty (μs)
    
    # Computed values (filled in by pipeline)
    mjd_tt: float = None  # Terrestrial Time
    mjd_tdb: float = None  # Barycentric Dynamical Time
    obs_pos_ssb: np.ndarray = None  # (x, y, z) in meters
    obs_vel_ssb: np.ndarray = None  # (vx, vy, vz) in m/s
```

---

## Phase 2: Time Conversions & Ephemeris

### 2.1 Clock Corrections: UTC → TT

**Goal**: Convert topocentric UTC to Terrestrial Time (TT)

**Formula**:
```
TT = UTC + clock_correction(obs, UTC)
```

**Clock correction chain**:
```
Observable UTC
    ↓
Observatory-specific clock file
    ↓
UTC (standardized)
    ↓
GPS (if needed)
    ↓
TAI (International Atomic Time)
    ↓
TT (Terrestrial Time)
    ↓ + BIPM file (for TAI→TT offset)
    ↓
Final TT time
```

**Implementation approach**:
```python
def clock_correction_seconds(mjd_utc, observatory, clock_dir='data/clock/'):
    """
    Compute total clock correction in seconds.
    
    Input: MJD in UTC, observatory code
    Output: Correction to add to UTC to get TT (in seconds)
    """
    # Step 1: Observatory-specific correction (UTC → UTC_corrected)
    obs_corr = get_obs_clock_correction(mjd_utc, observatory, clock_dir)
    
    # Step 2: GPS → TAI (if applicable)
    gps_corr = get_gps_to_tai_correction(mjd_utc, clock_dir)
    
    # Step 3: TAI → TT using BIPM file
    tai_to_tt = get_tai_to_tt_correction(mjd_utc, clock_dir)
    
    # Total correction
    total_corr = obs_corr + gps_corr + tai_to_tt
    
    return total_corr / 86400.0  # Convert to days

def mjd_utc_to_tt(mjd_utc, observatory, clock_dir):
    """Convert UTC to TT"""
    corr_days = clock_correction_seconds(mjd_utc, observatory, clock_dir) / 86400
    return mjd_utc + corr_days
```

**Clock file format** (tempo2 format):
```
# Each line: MJD_start MJD_end correction_seconds
58000.0 58001.0 0.000234
58001.0 58002.0 0.000235
...
```

### 2.2 Observatory Positions in SSB

**Goal**: Compute (x, y, z) position of observatory in Solar System Barycenter frame

**Requirements**:
- JPL ephemeris kernel (DE440 or similar)
- Observatory coordinates on Earth
- Date/time (MJD)

**Implementation approach**:
```python
def load_jpl_ephemeris(ephem_file='data/ephemeris/de440s.bsp'):
    """Load JPL ephemeris kernel"""
    import astropy.coordinates as coord
    from astropy.coordinates import solar_system_ephemeris
    solar_system_ephemeris.set(ephem_file)
    # Returns ephemeris object

def get_observatory_position_ssb(mjd_tt, observatory, ephem):
    """
    Get observatory position in SSB frame.
    
    Input: MJD in TT, observatory code, ephemeris object
    Output: (x, y, z) position in meters
    """
    # Step 1: Get Earth position relative to SSB
    earth_pos = ephem.earth_position(mjd_tt)  # (x, y, z)
    
    # Step 2: Get observatory position on Earth
    obs_pos_earth = get_obs_position_earth(observatory)  # (x, y, z)
    
    # Step 3: Rotate from Earth-fixed to SSB frame
    # (This requires Earth orientation parameters)
    obs_pos_ssb = earth_pos + obs_pos_earth_rotated
    
    return obs_pos_ssb
```

### 2.3 Compute TDB Times

**Goal**: Convert TT → TDB (Barycentric Dynamical Time)

**Key insight**: TDB accounts for relativistic effects

**Formula**:
```
TDB = TT - Einstein_delay - Shapiro_delay
```

**Components**:
```python
def compute_einstein_delay(obs_pos_ssb, obs_vel_ssb):
    """
    Gravitational time dilation effect.
    
    Depends on observer's velocity and position in gravitational field.
    Magnitude: ~microseconds
    """
    # General relativity: dt = (1/2c²) * v² + φ/c²
    # where v = velocity, φ = gravitational potential
    
    v_squared = np.sum(obs_vel_ssb**2)
    einstein_delay = (v_squared / 2.0) / (c**2)
    
    return einstein_delay

def compute_shapiro_delay(obs_pos_ssb, sun_pos_ssb):
    """
    Gravitational light bending by Sun.
    
    Magnitude: ~100 microseconds maximum
    """
    # Shapiro: dt ≈ (2*M_sun/c³) * ln(|a+b|/|a-b|)
    # where a, b are distances
    
    r_obs = np.linalg.norm(obs_pos_ssb)
    r_sun = np.linalg.norm(sun_pos_ssb)
    
    # Simplified formula (good enough)
    shapiro = (2 * M_SUN / (c**3)) * np.log(r_obs + r_sun)
    
    return shapiro

def compute_tdb(mjd_tt, obs_pos_ssb, obs_vel_ssb, sun_pos_ssb):
    """Compute TDB from TT"""
    einstein = compute_einstein_delay(obs_pos_ssb, obs_vel_ssb)
    shapiro = compute_shapiro_delay(obs_pos_ssb, sun_pos_ssb)
    
    mjd_tdb = mjd_tt - (einstein + shapiro) / 86400.0
    
    return mjd_tdb
```

---

## Phase 3: Evaluate Timing Model

### 3.1 Base Phase (Spindown)

**Goal**: Compute rotational phase as function of TDB time

**Formula**:
```
phase_base = F0 * (t - PEPOCH) + 0.5 * F1 * (t - PEPOCH)² + (1/6) * F2 * (t - PEPOCH)³

Where:
  F0 = spin frequency (Hz)
  F1 = spin-down rate (Hz/s)
  F2 = second derivative (Hz/s²)
  t = time (days since J2000.0 = MJD 51544.5)
  PEPOCH = reference epoch (MJD)
```

**Implementation**:
```python
def compute_spindown_phase(mjd_tdb, model):
    """Compute base rotational phase"""
    dt = mjd_tdb - model.PEPOCH  # Time offset in days
    dt_sec = dt * 86400.0  # Convert to seconds
    
    phase = (model.F0 * dt + 
             0.5 * model.F1 * dt**2 + 
             (1/6) * model.F2 * dt**3)
    
    return phase  # in cycles
```

### 3.2 Astrometric Delays

**Goal**: Account for pulsar position (RA, DEC, parallax, proper motion)

**Main effect**: Roemer delay (light travel time)

**Formula**:
```
Roemer_delay = (distance to pulsar) * cos(angle_to_observer)
              + proper motion effects
              + parallax effects

Simplified:
delay ≈ -k · cos(α) - (px) * u² / 2

where:
  k = unit vector toward pulsar
  α = angle between k and observer position
  px = parallax (arcsec)
  u = observer position projected perpendicular to k
```

**Implementation**:
```python
def compute_astrometric_delay(mjd_tdb, obs_pos_ssb, model):
    """
    Compute Roemer delay from astrometry.
    
    This is the light travel time from the observer to the SSB,
    accounting for the pulsar's position on the sky.
    """
    # Convert pulsar RA/DEC to unit vector
    ra_rad = model.RA * np.pi / 180.0
    dec_rad = model.DEC * np.pi / 180.0
    
    k_hat = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])
    
    # Projection: observer position component along k_hat
    r_proj = np.dot(obs_pos_ssb, k_hat)
    
    # Roemer delay = -r_proj / c
    roemer_delay = -r_proj / C_M_S
    
    # Add parallax effect (if parallax is provided)
    if model.PX > 0:
        # Distance to pulsar (in meters)
        distance = AU_M / (model.PX / 3600.0 / 180.0 * np.pi)  # AU to meters
        
        # Parallax delay
        u_perp = obs_pos_ssb - r_proj * k_hat  # Perpendicular component
        u_perp_mag = np.linalg.norm(u_perp)
        
        parallax_delay = -0.5 * u_perp_mag**2 / distance / C_M_S
    else:
        parallax_delay = 0
    
    # Add proper motion (time-dependent)
    # For now, simplified version
    
    total_delay = roemer_delay + parallax_delay
    
    return total_delay / 86400.0  # Convert to days
```

### 3.3 Dispersion Measure Delay

**Goal**: Account for frequency-dependent delay in interstellar medium

**Formula**:
```
DM_delay = K_DM * DM / f²

Where:
  K_DM = 4.148808e3 (dispersion constant, MHz² pc⁻¹ cm³ s)
  DM = dispersion measure (pc/cm³)
  f = observation frequency (MHz)
```

**Implementation**:
```python
K_DM_SEC = 4.148808e3  # dispersion constant

def compute_dm_delay(freq_mhz, model):
    """
    Compute dispersion delay (frequency-dependent).
    
    This is the cold-plasma dispersion effect.
    """
    delay_sec = K_DM_SEC * model.DM / (freq_mhz**2)
    
    # Add higher-order DM variations if present
    if hasattr(model, 'DM1') and model.DM1:
        delay_sec += model.DM1 * model.DM / (freq_mhz**2)
    
    return delay_sec / 86400.0  # Convert to days
```

### 3.4 Binary Orbital Delays (if Binary)

**Goal**: Account for companion's Roemer, Einstein, and Shapiro delays

**This is complex** - requires solving Kepler equation. For now, assume single pulsar.

```python
def compute_binary_delay(mjd_tdb, model):
    """
    Compute delays from binary companion.
    
    Implements ELL1H or BT model depending on BINARY parameter.
    This is the most complex part - see PINT source for details.
    """
    if not model.BINARY:
        return 0.0
    
    if model.BINARY.startswith('ELL1'):
        return compute_ell1_delay(mjd_tdb, model)
    elif model.BINARY.startswith('BT'):
        return compute_bt_delay(mjd_tdb, model)
    else:
        return 0.0
```

### 3.5 Solar Shapiro Delay

**Goal**: Account for light bending by Sun

**Formula**:
```
Shapiro_delay ≈ (2*M_sun/c³) * ln(|a+b|/|a-b|)
```

**Implementation**:
```python
def compute_solar_shapiro_delay(obs_pos_ssb, sun_pos_ssb, pulsar_vec):
    """
    Compute Shapiro delay from Sun.
    
    This is typically small (~100 μs) unless Sun is near line of sight.
    """
    M_SUN = 1.989e30  # kg
    G = 6.674e-11  # m³ kg⁻¹ s⁻²
    C_M_S = 299792458.0  # m/s
    
    # Gravitational parameter
    mu_sun = G * M_SUN
    
    # Calculate delay (simplified)
    r_sun = np.linalg.norm(sun_pos_ssb)
    
    # Maximum Shapiro delay = 2*mu/c³
    max_shapiro = 2 * mu_sun / (C_M_S**3)
    
    # Actual delay depends on pulsar position relative to Sun
    # For simplicity, use approximate formula
    
    cos_psi = np.dot(obs_pos_ssb - sun_pos_ssb, pulsar_vec) / np.linalg.norm(obs_pos_ssb - sun_pos_ssb)
    
    if cos_psi > 0:
        shapiro = max_shapiro * (1 + cos_psi)
    else:
        shapiro = 0
    
    return shapiro / 86400.0  # Convert to days
```

### 3.6 Total Phase Calculation

**Goal**: Sum all components

**Implementation**:
```python
def compute_phase(mjd_tdb, freq_mhz, obs_pos_ssb, obs_vel_ssb, sun_pos_ssb, model):
    """
    Compute total phase including all effects.
    
    Returns: phase (in cycles)
    """
    # Base rotational phase
    phase = compute_spindown_phase(mjd_tdb, model)
    
    # Astrometric delay (convert seconds → cycles)
    astrometric_delay_sec = compute_astrometric_delay(mjd_tdb, obs_pos_ssb, model)
    phase -= astrometric_delay_sec * 86400 * model.F0
    
    # DM delay (frequency-dependent)
    dm_delay_sec = compute_dm_delay(freq_mhz, model)
    phase -= dm_delay_sec * 86400 * model.F0
    
    # Shapiro delay (from Sun)
    pulsar_vec = np.array([...])  # Unit vector toward pulsar
    shapiro_delay_sec = compute_solar_shapiro_delay(obs_pos_ssb, sun_pos_ssb, pulsar_vec)
    phase -= shapiro_delay_sec * 86400 * model.F0
    
    # Binary delay (if applicable)
    binary_delay_sec = compute_binary_delay(mjd_tdb, model)
    phase -= binary_delay_sec * 86400 * model.F0
    
    return phase
```

---

## Phase 4: Extract Residuals

### 4.1 Split Phase into Integer and Fractional Parts

**Goal**: Get residuals from phase

**Implementation**:
```python
def extract_residuals(phase):
    """
    Extract residuals from phase.
    
    Input: phase in cycles
    Output: phase_frac in range [0, 1), residual in microseconds
    """
    # Split into integer and fractional parts
    phase_int = np.floor(phase)
    phase_frac = phase - phase_int
    
    # Ensure phase_frac is in [0, 1)
    phase_frac = np.mod(phase_frac, 1.0)
    
    return phase_int, phase_frac

def compute_residuals_microseconds(phase_frac, f0):
    """
    Convert phase residuals to time residuals.
    
    Input: phase_frac (in cycles), f0 (spin frequency in Hz)
    Output: residuals in microseconds
    """
    residuals_sec = phase_frac / f0
    residuals_us = residuals_sec * 1e6
    
    return residuals_us
```

### 4.2 Complete Pipeline

**Putting it all together**:
```python
def compute_jug_residuals(par_file, tim_file, clock_dir, ephem_file):
    """
    Complete JUG pipeline: .par + .tim → residuals
    """
    # Parse inputs
    model = parse_parfile(par_file)
    toas = parse_timfile(tim_file)
    
    # Prepare ephemeris
    ephem = load_jpl_ephemeris(ephem_file)
    
    # Process each TOA
    residuals = []
    
    for toa in toas:
        # Step 1: Convert UTC → TT
        mjd_tt = mjd_utc_to_tt(toa['mjd'], toa['obs'], clock_dir)
        
        # Step 2: Get observatory position
        obs_pos_ssb, obs_vel_ssb = get_observatory_position_ssb(mjd_tt, toa['obs'], ephem)
        
        # Step 3: Get Sun position
        sun_pos_ssb = ephem.sun_position(mjd_tt)
        
        # Step 4: Compute TDB
        mjd_tdb = compute_tdb(mjd_tt, obs_pos_ssb, obs_vel_ssb, sun_pos_ssb)
        
        # Step 5: Evaluate timing model
        phase = compute_phase(mjd_tdb, toa['freq'], 
                            obs_pos_ssb, obs_vel_ssb, sun_pos_ssb, model)
        
        # Step 6: Extract residuals
        phase_int, phase_frac = extract_residuals(phase)
        residual_us = compute_residuals_microseconds(phase_frac, model.F0)
        
        residuals.append(residual_us)
    
    return np.array(residuals)
```

---

## Implementation Roadmap

### Stage 1: Minimal Working Version
- [ ] Parse .par file (F0, F1, DM, PEPOCH, RA, DEC)
- [ ] Parse .tim file (mjd, freq, obs)
- [ ] Simple clock corrections (constant offset for testing)
- [ ] Flat TDB (no Einstein/Shapiro)
- [ ] Compute spindown phase
- [ ] Compute DM delay
- [ ] Extract residuals
- **Test**: Residuals should have correct RMS (~0.8 μs)

### Stage 2: Accurate Time Conversions
- [ ] Proper clock correction system
- [ ] Load and use clock files
- [ ] Implement UTC → TT conversion chain
- **Test**: Compare against PINT's TT times

### Stage 3: Ephemeris Integration
- [ ] Load JPL DE440 ephemeris
- [ ] Compute observatory positions in SSB
- [ ] Compute Earth position
- [ ] Implement Einstein and Shapiro delays for TDB
- **Test**: Compare TDB times against PINT

### Stage 4: Astrometry
- [ ] Implement Roemer delay
- [ ] Implement parallax effects
- [ ] Implement proper motion
- **Test**: Residuals still match PINT

### Stage 5: Binary Support (if needed)
- [ ] Implement ELL1 model
- [ ] Implement BT model
- [ ] Solve Kepler equation
- **Test**: Binary pulsars match PINT

### Stage 6: Optimization & Validation
- [ ] JAX integration for speed
- [ ] Vectorization across TOAs
- [ ] Error handling
- [ ] Comprehensive testing
- **Test**: Residuals match PINT to μs precision

---

## Testing & Validation

### Test 1: RMS Residuals
```
PINT:  0.8175 μs
JUG:   should be ≈ 0.8175 μs
```

### Test 2: Distribution
```
Both should show normal-like distribution
Mean ≈ 0
Std ≈ RMS
Range ≈ ±8 μs
```

### Test 3: Correlation
```
correlation(PINT, JUG) should be > 0.99
```

### Test 4: Component Verification
As you add components, verify:
- DM delay: matches PINT's frequency dependence
- Roemer delay: matches PINT's time variation
- Shapiro delay: small (~100 μs max)
- TDB times: match PINT to nanosecond precision

---

## Key Constants

```python
C_M_S = 299792458.0  # Speed of light (m/s)
SECS_PER_DAY = 86400.0  # Seconds per day
AU_M = 149597870700.0  # Astronomical unit (meters)
K_DM_SEC = 4.148808e3  # DM dispersion constant
M_SUN = 1.989e30  # Solar mass (kg)
G = 6.674e-11  # Gravitational constant
L_B = 1.550519768e-8  # TCB-TDB scaling factor
```

---

## References & Resources

1. **PINT Source Code**: `/home/mattm/soft/PINT/src/pint/`
   - `toa.py`: TOA handling
   - `models/`: Timing model components
   - `models/absolute_phase.py`: Phase calculation

2. **Pulsar Timing Theory**:
   - Lorimer & Kramer, "Handbook of Pulsar Astronomy"
   - Damour & Deruelle (1986) for binary models
   - Anderson et al. (1990) for Shapiro delay

3. **Reference Data**:
   - JPL DE440 ephemeris: NASA Horizons
   - Clock files: ATNF Pulsar Database
   - Pulsar parameters: ATNF catalog

---

## Success Criteria

You'll know JUG is working when:

✓ Residuals match PINT to 99%+ correlation
✓ RMS residuals within 1% of PINT's
✓ All major timing effects accounted for
✓ Performance: faster than PINT (with JAX JIT)
✓ Handles J1909-3744 test case perfectly

At that point, JUG becomes a standalone timing pipeline!
