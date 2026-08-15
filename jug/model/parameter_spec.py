"""
ParameterSpec Registry
======================

Defines parameter metadata for all timing model parameters. This replaces
scattered param.startswith() checks with spec-driven routing.

Key concepts:
- ParameterSpec: Immutable dataclass defining parameter properties
- DerivativeGroup: Enum for routing to appropriate derivative functions
- PARAMETER_REGISTRY: Dict mapping canonical names to specs
- Aliases: Alternative names (e.g., NU -> F0)

Usage:
    from jug.model.parameter_spec import get_spec, canonicalize_param_name

    # Get spec for a parameter
    spec = get_spec('F0')
    print(spec.group)  # 'spin'
    print(spec.derivative_group)  # DerivativeGroup.SPIN

    # Resolve aliases
    canonical = canonicalize_param_name('NU')  # Returns 'F0'

    # List parameters by group
    spin_params = list_params_by_group('spin')
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
import re
from typing import Optional, Tuple, Dict, List
import re


class DerivativeGroup(Enum):
    """
    Groups parameters by their derivative computation pathway.

    Each group corresponds to a different derivative function module:
    - SPIN: derivatives_spin.py (F0, F1, F2, ...)
    - DM: derivatives_dm.py (DM, DM1, DM2, ...)
    - ASTROMETRY: derivatives_astrometry.py (RAJ, DECJ, PMRA, PMDEC, PX)
    - BINARY: derivatives_binary.py (PB, A1, ECC, OM, T0, ...)
    - EPOCH: Reference epochs (PEPOCH, DMEPOCH, POSEPOCH, T0) - not fitted directly
    - JUMP: Backend/receiver offsets
    - FD: Frequency-dependent delays
    """
    SPIN = auto()
    DM = auto()
    ASTROMETRY = auto()
    BINARY = auto()
    EPOCH = auto()
    JUMP = auto()
    FD = auto()
    SOLAR_WIND = auto()


@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for a timing model parameter.

    Attributes
    ----------
    name : str
        Canonical name (F0, RAJ, etc.)
    group : str
        Human-readable group (spin, dm, astrometry, binary, epoch)
    dtype : str
        Numeric type (float64 or longdouble)
    internal_unit : str
        Internal storage unit (Hz, rad, s, pc/cm^3)
    par_unit_str : str
        Unit label as it appears in .par files (ASCII, e.g. 's^-2', 'Msun')
    display_unit : str
        Human-readable unit for GUI display (Unicode OK, e.g. 'Hz/s', 'MSun').
        Falls back to par_unit_str when empty.
    fit_unit : str
        Unit string for design-matrix/fitting parameter increments. This is the
        API-facing convention used by ``compute_designmatrix`` and should follow
        PINT/Vela-style ``str(param.units)`` vocabulary.
    aliases : tuple
        Alternative names that resolve to this parameter
    derivative_group : DerivativeGroup
        Routing group for derivative computation
    default_fit : bool
        Whether this parameter is fitted by default
    gui_visible : bool
        Whether to show in GUI parameter list
    requires : tuple
        Prerequisites (e.g., DM1 requires DMEPOCH)
    par_codec_name : str
        Name of codec for I/O transformation

    Notes
    -----
    - All angles are stored internally as radians
    - Codecs handle conversion at I/O boundary only
    - This class is immutable (frozen=True)
    """
    name: str
    group: str
    derivative_group: DerivativeGroup
    dtype: str = "float64"
    internal_unit: str = ""
    par_unit_str: str = ""
    display_unit: str = ""  # Falls back to par_unit_str when empty
    fit_unit: str = ""
    aliases: Tuple[str, ...] = ()
    default_fit: bool = False
    gui_visible: bool = True
    requires: Tuple[str, ...] = ()
    par_codec_name: str = "float"
    # Display formatting for GUI fit reports
    display_format: str = ".6g"
    """Format specifier for displaying parameter values in fit reports.
    E.g. '.15f' for F0, '.6e' for small quantities, '.10f' for moderate precision."""
    # TCB->TDB conversion metadata (single source of truth)
    tcb_scaling_dim: Optional[int] = None
    """Effective time dimensionality for TCB->TDB scaling.
    None = no scaling needed. n means: x_tdb = x_tcb * IFTE_K^(-n).
    E.g. F0 (frequency) = -1, A1 (time) = 1, PBDOT (dimensionless) = 0."""
    is_epoch: bool = False
    """True for MJD epoch parameters that need TCB->TDB epoch conversion."""
    high_precision: bool = False
    """True for parameters requiring np.longdouble precision in par reader."""


# =============================================================================
# Parameter Registry
# =============================================================================

# Spin parameters
_SPIN_PARAMS = [
    ParameterSpec(
        name="F0",
        group="spin",
        derivative_group=DerivativeGroup.SPIN,
        dtype="float64",
        internal_unit="Hz",
        par_unit_str="Hz",
        aliases=("NU", "F"),
        default_fit=True,
        requires=("PEPOCH",),
        display_format=".15f",
        tcb_scaling_dim=-1,
        high_precision=True,
    ),
    ParameterSpec(
        name="F1",
        group="spin",
        derivative_group=DerivativeGroup.SPIN,
        dtype="float64",
        internal_unit="Hz/s",
        par_unit_str="s^-2",
        display_unit="Hz/s",
        aliases=("NUDOT", "FDOT"),
        default_fit=True,
        requires=("PEPOCH",),
        display_format=".6e",
        tcb_scaling_dim=-2,
        high_precision=True,
    ),
    ParameterSpec(
        name="PEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        gui_visible=False,
        par_codec_name="epoch_mjd",
        is_epoch=True,
        high_precision=True,
    ),
]

# Add F2-F20 (higher-order spin derivatives, matching FB pattern)
for i in range(2, 21):
    _SPIN_PARAMS.append(
        ParameterSpec(
            name=f"F{i}",
            group="spin",
            derivative_group=DerivativeGroup.SPIN,
            dtype="longdouble",
            internal_unit=f"Hz/s^{i}",
            par_unit_str=f"s^-{i + 1}",
            display_unit=f"Hz/s^{i}",
            requires=("PEPOCH",),
            display_format=".6e",
            tcb_scaling_dim=-(i + 1),
            high_precision=True,
        )
    )

# DM parameters
_DM_PARAMS = [
    ParameterSpec(
        name="DM",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3",
        par_unit_str="pc cm^-3",
        display_unit="pc/cm^3",
        aliases=("DM0",),
        default_fit=True,
        display_format=".10f",
        tcb_scaling_dim=-1,
    ),
    ParameterSpec(
        name="DM1",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3/yr",
        par_unit_str="pc cm^-3 yr^-1",
        display_unit="pc/cm^3/yr",
        requires=("DMEPOCH",),
        display_format=".10f",
        tcb_scaling_dim=-2,
    ),
    ParameterSpec(
        name="DM2",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3/yr^2",
        par_unit_str="pc cm^-3 yr^-2",
        display_unit="pc/cm^3/yr^2",
        requires=("DMEPOCH",),
        display_format=".10f",
        tcb_scaling_dim=-3,
    ),
    ParameterSpec(
        name="DMEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        gui_visible=False,
        par_codec_name="epoch_mjd",
        is_epoch=True,
        high_precision=True,
    ),
]

# Add DM3-DM20 (higher-order DM derivatives, matching FB pattern)
for i in range(3, 21):
    _DM_PARAMS.append(
        ParameterSpec(
            name=f"DM{i}",
            group="dm",
            derivative_group=DerivativeGroup.DM,
            dtype="float64",
            internal_unit=f"pc/cm^3/yr^{i}",
            par_unit_str=f"pc cm^-3 yr^-{i}",
            display_unit=f"pc/cm^3/yr^{i}",
            requires=("DMEPOCH",),
            display_format=".10f",
            tcb_scaling_dim=-(i + 1),
        )
    )

# Astrometry parameters
_ASTROMETRY_PARAMS = [
    ParameterSpec(
        name="RAJ",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad",  # CRITICAL: radians internally
        par_unit_str="HH:MM:SS.sss",
        requires=("POSEPOCH",),
        par_codec_name="raj",
    ),
    ParameterSpec(
        name="DECJ",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad",  # CRITICAL: radians internally
        par_unit_str="DD:MM:SS.sss",
        requires=("POSEPOCH",),
        par_codec_name="decj",
    ),
    ParameterSpec(
        name="PMRA",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad/yr",
        par_unit_str="mas/yr",
        # PMLAMBDA/PMBETA are ecliptic (aliases of PMELONG/PMELAT), not equatorial.
        aliases=("PMRAC",),
        requires=("POSEPOCH",),
        display_format=".6f",
    ),
    ParameterSpec(
        name="PMDEC",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad/yr",
        par_unit_str="mas/yr",
        requires=("POSEPOCH",),
        display_format=".6f",
    ),
    ParameterSpec(
        name="PX",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad",
        par_unit_str="mas",
        aliases=("PARALLAX",),
        display_format=".6f",
    ),
    ParameterSpec(
        name="ELONG",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="degrees",
        requires=("POSEPOCH",),
        # Tempo2 LAMBDA is ecliptic longitude in degrees (same as ELONG), not RAJ.
        aliases=("LAMBDA",),
    ),
    ParameterSpec(
        name="ELAT",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="degrees",
        requires=("POSEPOCH",),
        # Tempo2 BETA is ecliptic latitude in degrees (same as ELAT), not DECJ.
        aliases=("BETA",),
    ),
    ParameterSpec(
        name="PMELONG",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="mas/yr",
        par_unit_str="mas/yr",
        # Tempo2 PMLAMBDA is ecliptic longitude proper motion (same as PMELONG).
        aliases=("PMLAMBDA",),
        requires=("POSEPOCH",),
        display_format=".6f",
    ),
    ParameterSpec(
        name="PMELAT",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="mas/yr",
        par_unit_str="mas/yr",
        # Tempo2 PMBETA is ecliptic latitude proper motion (same as PMELAT).
        aliases=("PMBETA",),
        requires=("POSEPOCH",),
        display_format=".6f",
    ),
    ParameterSpec(
        name="POSEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        gui_visible=False,
        par_codec_name="epoch_mjd",
        is_epoch=True,
        high_precision=True,
    ),
]

# Binary Keplerian parameters
_BINARY_PARAMS = [
    ParameterSpec(
        name="PB",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="day",
        par_unit_str="d",
        display_unit="days",
        aliases=("PORB",),
        display_format=".15f",
        tcb_scaling_dim=1,
        high_precision=True,
    ),
    ParameterSpec(
        name="A1",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="lt-s",
        par_unit_str="lt-s",
        aliases=("ASINI",),
        display_format=".10f",
        tcb_scaling_dim=1,
    ),
    ParameterSpec(
        name="ECC",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("E",),
        display_format=".12e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="OM",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="rad",
        par_unit_str="deg",
        aliases=("OMEGA",),
        display_format=".10f",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="T0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        par_codec_name="epoch_mjd",
        display_format=".12f",
        is_epoch=True,
        high_precision=True,
    ),
    # ELL1 parameters
    ParameterSpec(
        name="TASC",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        par_codec_name="epoch_mjd",
        display_format=".12f",
        is_epoch=True,
        high_precision=True,
    ),
    ParameterSpec(
        name="EPS1",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        display_format=".12e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="EPS2",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        display_format=".12e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="EPS1DOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="1/s",
        par_unit_str="1/s",
        display_format=".6e",
        tcb_scaling_dim=-1,
    ),
    ParameterSpec(
        name="EPS2DOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="1/s",
        par_unit_str="1/s",
        display_format=".6e",
        tcb_scaling_dim=-1,
    ),
    # Post-Keplerian / Shapiro parameters
    ParameterSpec(
        name="SINI",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        display_format=".12f",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="M2",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="Msun",
        par_unit_str="Msun",
        display_unit="MSun",
        display_format=".12f",
        tcb_scaling_dim=0,
    ),
    # DDS alternate Shapiro inclination: SHAPMAX = -log(1 - sin i).
    ParameterSpec(
        name="SHAPMAX",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        display_format=".12f",
        tcb_scaling_dim=0,
    ),
    # DDGR total system mass (derives SINI/GAMMA/PBDOT/OMDOT/DR/DTH via GR).
    ParameterSpec(
        name="MTOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="Msun",
        par_unit_str="Msun",
        display_unit="MSun",
        display_format=".12f",
        tcb_scaling_dim=0,
    ),
    # DDGR excess periastron advance / orbital decay beyond GR.
    ParameterSpec(
        name="XOMDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg/yr",
        par_unit_str="deg/yr",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="XPBDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s/s",
        par_unit_str="s/s",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    # Time derivatives
    ParameterSpec(
        name="PBDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s/s",
        par_unit_str="s/s",
        display_unit="s/s",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="XDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="lt-s/s",
        par_unit_str="lt-s/s",
        aliases=("A1DOT",),
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    # Periastron advance (DD model)
    ParameterSpec(
        name="OMDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg/yr",
        par_unit_str="deg/yr",
        display_format=".6e",
        tcb_scaling_dim=-1,
    ),
    # Time dilation + gravitational redshift (DD model)
    ParameterSpec(
        name="GAMMA",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        display_format=".6e",
        tcb_scaling_dim=1,
    ),
    # Eccentricity derivative (T2 model)
    ParameterSpec(
        name="EDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="1/s",
        par_unit_str="",
        display_format=".6e",
        tcb_scaling_dim=-1,
    ),
    # Orthometric Shapiro parameters (ELL1H model)
    ParameterSpec(
        name="H3",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="H4",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="STIG",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("STIGMA",),
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    # DD model relativistic deformation parameters
    ParameterSpec(
        name="DR",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="DTH",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("DTHETA",),
        display_format=".6e",
        tcb_scaling_dim=0,
    ),
    # Aberration parameters (DD model)
    ParameterSpec(
        name="A0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        display_format=".6e",
        tcb_scaling_dim=1,
    ),
    ParameterSpec(
        name="B0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        display_format=".6e",
        tcb_scaling_dim=1,
    ),
    # Kopeikin annual orbital parallax parameters (DDK model)
    ParameterSpec(
        name="KIN",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="deg",
        display_format=".10f",
        tcb_scaling_dim=0,
    ),
    ParameterSpec(
        name="KOM",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="deg",
        display_format=".10f",
        tcb_scaling_dim=0,
    ),
]

# Add FB parameters (FB0 to FB20)
for i in range(21):
    _BINARY_PARAMS.append(
        ParameterSpec(
            name=f"FB{i}",
            group="binary",
            derivative_group=DerivativeGroup.BINARY,
            dtype="float64",
            internal_unit=f"Hz/s^{i}" if i > 0 else "Hz",
            par_unit_str="",
            display_format=".6e",
            tcb_scaling_dim=-(i + 1),
            high_precision=(i == 0),
        )
    )

# FD (frequency-dependent) parameters — FD1 to FD20
_FD_PARAMS = []
for i in range(1, 21):
    _FD_PARAMS.append(
        ParameterSpec(
            name=f"FD{i}",
            group="fd",
            derivative_group=DerivativeGroup.FD,
            dtype="float64",
            internal_unit="s",
            par_unit_str="s",
            display_format=".6e",
        )
    )

# Solar wind parameters
_SW_PARAMS = [
    ParameterSpec(
        name="NE_SW",
        group="solar_wind",
        derivative_group=DerivativeGroup.SOLAR_WIND,
        dtype="float64",
        internal_unit="cm^-3",
        par_unit_str="cm^-3",
        display_unit="cm^-^3",
        aliases=("NE1AU",),
    ),
]


def _resolve_fit_unit(spec: ParameterSpec) -> str:
    """Return PINT/Vela-compatible fit unit string for a registry spec."""
    if spec.fit_unit:
        return spec.fit_unit

    name = spec.name.upper()

    explicit = {
        "RAJ": "hourangle",
        "DECJ": "deg",
        "PMRA": "mas/yr",
        "PMDEC": "mas/yr",
        "PX": "mas",
        "ELONG": "deg",
        "ELAT": "deg",
        "PMELONG": "mas/yr",
        "PMELAT": "mas/yr",
        "PB": "d",
        "A1": "ls",
        "OM": "deg",
        "T0": "MJD",
        "TASC": "MJD",
        "SINI": "1",
        "M2": "solMass",
        "PBDOT": "1",
        "XDOT": "ls/s",
        "OMDOT": "deg/year",
        "GAMMA": "s",
        "EDOT": "1/s",
        "H3": "s",
        "H4": "s",
        "STIG": "1",
        "DR": "1",
        "DTH": "1",
        "A0": "s",
        "B0": "s",
        "KIN": "deg",
        "KOM": "deg",
        "PEPOCH": "MJD",
        "POSEPOCH": "MJD",
        "DMEPOCH": "MJD",
        "NE_SW": "cm^-3",
        "EPS1": "1",
        "EPS2": "1",
        "EPS1DOT": "1/s",
        "EPS2DOT": "1/s",
    }
    if name in explicit:
        return explicit[name]

    if re.match(r"^F\d+$", name):
        order = int(name[1:])
        return "Hz" if order == 0 else f"Hz/s^{order}"

    if re.match(r"^DM\d+$", name):
        order = int(name[2:])
        if order == 0:
            return "pc cm^-3"
        return f"pc cm^-3 yr^-{order}"

    if re.match(r"^FB\d+$", name):
        order = int(name[2:])
        return f"1/s^{order + 1}"

    if re.match(r"^FD\d+$", name):
        return "s"

    aliases = {
        "": "1",
        "degrees": "deg",
        "day": "d",
        "lt-s": "ls",
        "lt-s/s": "ls/s",
        "Msun": "solMass",
        "s/s": "1",
    }
    if spec.par_unit_str:
        return aliases.get(spec.par_unit_str, spec.par_unit_str)
    if spec.internal_unit:
        return aliases.get(spec.internal_unit, spec.internal_unit)
    return "1"


def _populate_fit_units(specs: List[ParameterSpec]) -> List[ParameterSpec]:
    """Populate fit_unit for static registry specs."""
    return [replace(spec, fit_unit=_resolve_fit_unit(spec)) for spec in specs]


_SPIN_PARAMS = _populate_fit_units(_SPIN_PARAMS)
_DM_PARAMS = _populate_fit_units(_DM_PARAMS)
_ASTROMETRY_PARAMS = _populate_fit_units(_ASTROMETRY_PARAMS)
_BINARY_PARAMS = _populate_fit_units(_BINARY_PARAMS)
_FD_PARAMS = _populate_fit_units(_FD_PARAMS)
_SW_PARAMS = _populate_fit_units(_SW_PARAMS)

# Build the registry
PARAMETER_REGISTRY: Dict[str, ParameterSpec] = {}
_ALIAS_MAP: Dict[str, str] = {}  # alias -> canonical name

for spec in _SPIN_PARAMS + _DM_PARAMS + _ASTROMETRY_PARAMS + _BINARY_PARAMS + _FD_PARAMS + _SW_PARAMS:
    PARAMETER_REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        _ALIAS_MAP[alias] = spec.name


# =============================================================================
# Derived TCB/TDB conversion lists (generated from registry)
# =============================================================================

def get_scaled_parameters() -> list:
    """Return (param_name, effective_dim) pairs from the registry.

    This replaces the hardcoded SCALED_PARAMETERS list in timescales.py
    for all parameters that are in the registry.
    """
    result = []
    for spec in PARAMETER_REGISTRY.values():
        if spec.tcb_scaling_dim is not None:
            result.append((spec.name, spec.tcb_scaling_dim))
    return result


def get_epoch_parameters() -> set:
    """Return the set of epoch parameter names from the registry."""
    return {spec.name for spec in PARAMETER_REGISTRY.values() if spec.is_epoch}


def get_high_precision_params() -> set:
    """Return the set of parameter names requiring longdouble precision."""
    return {spec.name for spec in PARAMETER_REGISTRY.values() if spec.high_precision}


# =============================================================================
# Helper Functions
# =============================================================================

# FDJUMP has two par-file spellings for one physical parameter (Tempo2 FDJUMPp
# vs PINT FDpJUMP). They cannot live in _ALIAS_MAP: the prefix index and the
# mask index collide, which is why PINT rewrites at parse time instead of
# registering an ordinary alias. Canonical internal id is FDJUMP{p}_{q}
# (mask q defaults to 1) and FDJUMPDM_{k}.
_FDJUMP_TEMPO2_INSTANCE_RE = re.compile(r"^FDJUMP(\d+)_(\d+)$", re.I)
_FDJUMP_TEMPO2_BARE_RE = re.compile(r"^FDJUMP(\d+)$", re.I)
_FDJUMP_PINT_INSTANCE_RE = re.compile(r"^FD(\d+)JUMP(\d+)$", re.I)
_FDJUMP_PINT_BARE_RE = re.compile(r"^FD(\d+)JUMP$", re.I)
_FDJUMPDM_RE = re.compile(r"^FDJUMPDM(?:_(\d+)|(\d+))?$", re.I)
_FDJUMP_CONTROL_KEYS = frozenset({"FDJUMP_SCALE", "FDJUMPLOG"})


def canonicalize_fdjump_name(name: str) -> Optional[str]:
    """Return the canonical FDJUMP id, or None if ``name`` is not an FDJUMP.

    Canonical form is ``FDJUMP{p}_{q}`` (Tempo2 prefix, JUG instance index;
    omitted mask defaults to 1) or ``FDJUMPDM_{k}``.

    Accepted spellings
    ------------------
    ``FDJUMP1`` / ``FD1JUMP``
        Par keyword, no mask → ``FDJUMP1_1``.
    ``FDJUMP1_1`` / ``FD1JUMP1``
        JUG internal / PINT mask instance → ``FDJUMP1_1``.
    ``FDJUMPDM`` / ``FDJUMPDM1`` / ``FDJUMPDM_1``
        DM-like sibling → ``FDJUMPDM_1``.
    """
    key = name.strip()
    if key.upper() in _FDJUMP_CONTROL_KEYS:
        return None
    dm = _FDJUMPDM_RE.fullmatch(key)
    if dm:
        idx = dm.group(1) or dm.group(2) or "1"
        return f"FDJUMPDM_{int(idx)}"
    m = _FDJUMP_TEMPO2_INSTANCE_RE.fullmatch(key)
    if m:
        return f"FDJUMP{int(m.group(1))}_{int(m.group(2))}"
    m = _FDJUMP_PINT_INSTANCE_RE.fullmatch(key)
    if m:
        return f"FDJUMP{int(m.group(1))}_{int(m.group(2))}"
    m = _FDJUMP_TEMPO2_BARE_RE.fullmatch(key)
    if m:
        return f"FDJUMP{int(m.group(1))}_1"
    m = _FDJUMP_PINT_BARE_RE.fullmatch(key)
    if m:
        return f"FDJUMP{int(m.group(1))}_1"
    return None


def fdjump_aliases(name: str) -> Tuple[str, ...]:
    """Return unambiguous spellings of one FDJUMP, or () if not an FDJUMP."""
    canonical = canonicalize_fdjump_name(name)
    if canonical is None:
        return ()
    if canonical.startswith("FDJUMPDM_"):
        k = int(canonical.rsplit("_", 1)[1])
        aliases = (f"FDJUMPDM_{k}", f"FDJUMPDM{k}")
        return aliases + (("FDJUMPDM",) if k == 1 else ())
    m = _FDJUMP_TEMPO2_INSTANCE_RE.fullmatch(canonical)
    if m is None:
        return (canonical,)
    p, q = int(m.group(1)), int(m.group(2))
    aliases = (
        f"FDJUMP{p}_{q}",
        f"FD{p}JUMP{q}",
    )
    if q == 1:
        aliases += (f"FDJUMP{p}", f"FD{p}JUMP")
    return aliases


def is_fdjump_param(name: str) -> bool:
    """True for any FDJUMP / FDJUMPDM spelling, including PINT ``FDpJUMPq``."""
    return canonicalize_fdjump_name(name) is not None


def canonicalize_param_name(name: str) -> str:
    """
    Resolve parameter aliases to canonical names.

    Parameters
    ----------
    name : str
        Parameter name (possibly an alias)

    Returns
    -------
    str
        Canonical parameter name

    Examples
    --------
    >>> canonicalize_param_name('NU')
    'F0'
    >>> canonicalize_param_name('F0')
    'F0'
    >>> canonicalize_param_name('FD1JUMP1')
    'FDJUMP1_1'
    >>> canonicalize_param_name('UNKNOWN')
    'UNKNOWN'
    """
    fdjump = canonicalize_fdjump_name(name)
    if fdjump is not None:
        return fdjump
    return _ALIAS_MAP.get(name, name)


def get_spec(name: str) -> Optional[ParameterSpec]:
    """
    Get the ParameterSpec for a parameter.

    Parameters
    ----------
    name : str
        Parameter name (aliases are resolved)

    Returns
    -------
    ParameterSpec or None
        The spec if found, None otherwise

    Examples
    --------
    >>> spec = get_spec('F0')
    >>> spec.group
    'spin'
    >>> spec = get_spec('NU')  # Alias
    >>> spec.name
    'F0'
    """
    canonical = canonicalize_param_name(name)
    spec = PARAMETER_REGISTRY.get(canonical)
    if spec is not None:
        return spec

    fb_match = re.fullmatch(r'FB(\d+)', canonical)
    if fb_match:
        index = int(fb_match.group(1))
        return ParameterSpec(
            name=canonical,
            group="binary",
            derivative_group=DerivativeGroup.BINARY,
            dtype="float64",
            internal_unit=f"Hz/s^{index}" if index > 0 else "Hz",
            par_unit_str="",
            display_format=".6e",
            tcb_scaling_dim=-(index + 1),
            high_precision=(index == 0),
        )
    return None


def get_display_unit(name: str) -> str:
    """
    Get the human-readable display unit for a parameter.

    Returns ``display_unit`` if set, otherwise falls back to ``par_unit_str``.
    Returns ``""`` for unknown parameters.

    Parameters
    ----------
    name : str
        Parameter name (aliases are resolved)

    Returns
    -------
    str
        Display-friendly unit string

    Examples
    --------
    >>> get_display_unit('F0')
    'Hz'
    >>> get_display_unit('F1')
    'Hz/s'
    >>> get_display_unit('M2')
    'MSun'
    """
    spec = get_spec(name)
    if spec is None:
        return ""
    return spec.display_unit if spec.display_unit else spec.par_unit_str


def get_fit_unit(name: str) -> str:
    """Return design-matrix fit unit string for a parameter."""
    spec = get_spec(name)
    if spec is None:
        if is_jump_param(name):
            return "s"
        return ""
    return spec.fit_unit


def get_derivative_group(name: str) -> Optional[DerivativeGroup]:
    """
    Get the derivative group for a parameter.

    Parameters
    ----------
    name : str
        Parameter name (aliases are resolved)

    Returns
    -------
    DerivativeGroup or None
        The derivative group if found, None otherwise

    Examples
    --------
    >>> get_derivative_group('F0')
    DerivativeGroup.SPIN
    >>> get_derivative_group('DM')
    DerivativeGroup.DM
    """
    spec = get_spec(name)
    return spec.derivative_group if spec else None


def list_params_by_group(group: str) -> List[str]:
    """
    List all parameters in a group.

    Parameters
    ----------
    group : str
        Group name (spin, dm, astrometry, binary, epoch)

    Returns
    -------
    list of str
        Parameter names in the group

    Examples
    --------
    >>> list_params_by_group('spin')
    ['F0', 'F1', 'F2', 'F3', 'PEPOCH']
    """
    return [
        spec.name for spec in PARAMETER_REGISTRY.values()
        if spec.group == group
    ]


def list_params_by_derivative_group(derivative_group: DerivativeGroup) -> List[str]:
    """
    List all parameters in a derivative group.

    Parameters
    ----------
    derivative_group : DerivativeGroup
        The derivative group

    Returns
    -------
    list of str
        Parameter names in the derivative group
    """
    return [
        spec.name for spec in PARAMETER_REGISTRY.values()
        if spec.derivative_group == derivative_group
    ]


def list_fittable_params() -> List[str]:
    """
    List all parameters that can be fitted.

    Returns parameters where derivative_group is not EPOCH
    (epochs are reference points, not fitted directly).

    Returns
    -------
    list of str
        Fittable parameter names

    Examples
    --------
    >>> 'F0' in list_fittable_params()
    True
    >>> 'PEPOCH' in list_fittable_params()
    False
    """
    return [
        spec.name for spec in PARAMETER_REGISTRY.values()
        if spec.derivative_group != DerivativeGroup.EPOCH
    ]


def is_spin_param(name: str) -> bool:
    """
    Check if a parameter is a spin parameter.

    This replaces param.startswith('F') checks.

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if spin parameter, False otherwise
    """
    spec = get_spec(name)
    return spec is not None and spec.derivative_group == DerivativeGroup.SPIN


def is_dm_param(name: str) -> bool:
    """
    Check if a parameter is a DM parameter.

    This replaces param.startswith('DM') checks.

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if DM parameter, False otherwise
    """
    spec = get_spec(name)
    return spec is not None and spec.derivative_group == DerivativeGroup.DM


def is_astrometry_param(name: str) -> bool:
    """
    Check if a parameter is an astrometry parameter.

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if astrometry parameter, False otherwise
    """
    spec = get_spec(name)
    return spec is not None and spec.derivative_group == DerivativeGroup.ASTROMETRY


def is_binary_param(name: str) -> bool:
    """
    Check if a parameter is a binary parameter.

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if binary parameter, False otherwise
    """
    spec = get_spec(name)
    return spec is not None and spec.derivative_group == DerivativeGroup.BINARY


def is_fd_param(name: str) -> bool:
    """
    Check if a parameter is an FD (frequency-dependent) parameter.

    FD parameters are dynamically named (FD1, FD2, ..., FD15, etc.)
    so we use pattern matching rather than static registry lookup.

    Patterns recognized:
    - FD followed by a number: FD1, FD2, FD10, FD15

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if FD parameter, False otherwise
    """
    import re
    # Match FD followed by one or more digits
    return bool(re.match(r'^FD\d+$', name))


def is_jump_param(name: str) -> bool:
    """
    Check if a parameter is a JUMP parameter.

    JUMP parameters are dynamically named (JUMP1, JUMP2, JUMP_-sys_..., etc.)
    so we use pattern matching rather than static registry lookup.

    Patterns recognized:
    - JUMP followed by a number: JUMP1, JUMP2, JUMP10
    - JUMP followed by underscore and identifier: JUMP_MJD, JUMP_-sys_...
    - Plain JUMP (legacy format)

    Parameters
    ----------
    name : str
        Parameter name

    Returns
    -------
    bool
        True if JUMP parameter, False otherwise

    Examples
    --------
    >>> is_jump_param('JUMP1')
    True
    >>> is_jump_param('JUMP_MJD_58000_59000')
    True
    >>> is_jump_param('JUMP')
    True
    >>> is_jump_param('F0')
    False
    """
    if not name.startswith('JUMP'):
        return False
    # Accept: JUMP, JUMP1, JUMP_foo, etc.
    suffix = name[4:]  # Everything after 'JUMP'
    if suffix == '':
        return True  # Plain 'JUMP'
    if suffix[0].isdigit():
        return True  # JUMP1, JUMP2, JUMP10, etc.
    if suffix[0] == '_':
        return True  # JUMP_MJD, JUMP_-sys_..., etc.
    return False


def create_jump_spec(name: str) -> ParameterSpec:
    """
    Create a ParameterSpec for a dynamically-named JUMP parameter.

    Use this when you encounter a JUMP parameter not in the registry.

    Parameters
    ----------
    name : str
        JUMP parameter name (e.g., 'JUMP1', 'JUMP_MJD_58000_59000')

    Returns
    -------
    ParameterSpec
        A spec for this JUMP parameter

    Raises
    ------
    ValueError
        If name is not a valid JUMP parameter
    """
    if not is_jump_param(name):
        raise ValueError(f"'{name}' is not a valid JUMP parameter")
    
    return ParameterSpec(
        name=name,
        group="jump",
        derivative_group=DerivativeGroup.JUMP,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        fit_unit="s",
    )


def get_spin_params_from_list(params: List[str]) -> List[str]:
    """
    Filter a list to only spin parameters.

    Replacement for: [p for p in params if p.startswith('F') and p[1:].isdigit()]

    Parameters
    ----------
    params : list of str
        Parameter names to filter

    Returns
    -------
    list of str
        Only the spin parameters
    """
    return [p for p in params if is_spin_param(p)]


def get_dm_params_from_list(params: List[str]) -> List[str]:
    """
    Filter a list to only DM parameters.

    Replacement for: [p for p in params if p.startswith('DM')]

    Parameters
    ----------
    params : list of str
        Parameter names to filter

    Returns
    -------
    list of str
        Only the DM parameters
    """
    return [p for p in params if is_dm_param(p)]


def get_binary_params_from_list(params: List[str]) -> List[str]:
    """
    Filter a list to only binary parameters.

    Parameters
    ----------
    params : list of str
        Parameter names to filter

    Returns
    -------
    list of str
        Only the binary parameters (ELL1: PB, A1, TASC, EPS1, EPS2, PBDOT, SINI, M2, etc.)
    """
    return [p for p in params if is_binary_param(p)]


def get_astrometry_params_from_list(params: List[str]) -> List[str]:
    """
    Filter a list to only astrometry parameters.

    Parameters
    ----------
    params : list of str
        Parameter names to filter

    Returns
    -------
    list of str
        Only the astrometry parameters (RAJ, DECJ, PMRA, PMDEC, PX)
    """
    return [p for p in params if is_astrometry_param(p)]


def get_fd_params_from_list(params: List[str]) -> List[str]:
    """
    Filter a list to only FD (frequency-dependent) parameters.

    Parameters
    ----------
    params : list of str
        Parameter names to filter

    Returns
    -------
    list of str
        Only the FD parameters (FD1, FD2, FD3, ...)
    """
    return [p for p in params if is_fd_param(p)]


def is_sw_param(name: str) -> bool:
    """Check if a parameter is a solar wind parameter (NE_SW / NE1AU)."""
    spec = get_spec(name)
    return spec is not None and spec.derivative_group == DerivativeGroup.SOLAR_WIND


def get_sw_params_from_list(params: List[str]) -> List[str]:
    """Filter a list to only solar wind parameters."""
    return [p for p in params if is_sw_param(p)]


def validate_fit_param(name: str) -> bool:
    """Validate that a parameter name is registered and can be fitted.

    Checks the PARAMETER_REGISTRY (after alias resolution) and known
    pattern families (FD, JUMP). Raises clear errors for unregistered
    or out-of-range parameters.

    Parameters
    ----------
    name : str
        Parameter name (aliases are resolved first)

    Returns
    -------
    bool
        True if the parameter is valid and fittable

    Raises
    ------
    ValueError
        If the parameter is not registered or is out of range
    """
    import re

    canonical = canonicalize_param_name(name)

    # Check direct registry lookup
    if canonical in PARAMETER_REGISTRY:
        return True

    # Check pattern families: JUMP (always valid via pattern match)
    if is_jump_param(canonical):
        return True

    # Check FDJUMP / FDJUMPDM patterns (Tempo2 FDJUMPp, PINT FDpJUMP, FDJUMPDM)
    if is_fdjump_param(canonical):
        return True

    # Check FD pattern - registered FD1..FD20, higher indices not yet implemented
    fd_match = re.match(r'^FD(\d+)$', canonical)
    if fd_match:
        fd_idx = int(fd_match.group(1))
        if 1 <= fd_idx <= 20:
            return True  # FD1-FD20 are registered
        raise ValueError(
            f"Parameter '{name}' (FD{fd_idx}) is out of range. "
            f"FD1-FD20 are registered; parametric families (FDn>20) are not yet implemented as first-class families."
        )

    # FB is a true parametric family; the forward model and derivatives consume
    # every contiguous FB0..FBN term present in the parameter dictionary.
    fb_match = re.match(r'^FB(\d+)$', canonical)
    if fb_match:
        return True

    # Unknown parameter
    raise ValueError(
        f"Parameter '{name}' is not registered. "
        f"Parametric families (JUMP1..N, DMX_*, FDn>20) are not yet implemented as first-class families."
    )
