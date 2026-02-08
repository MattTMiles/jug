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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List


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
        Human-readable unit for GUI display (Unicode OK, e.g. 'Hz/s', 'M☉').
        Falls back to par_unit_str when empty.
    aliases : tuple
        Alternative names that resolve to this parameter
    component_name : str
        Name of the component that provides this parameter
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
    aliases: Tuple[str, ...] = ()
    component_name: str = ""
    default_fit: bool = False
    gui_visible: bool = True
    requires: Tuple[str, ...] = ()
    par_codec_name: str = "float"


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
        component_name="SpinComponent",
        default_fit=True,
        requires=("PEPOCH",),
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
        component_name="SpinComponent",
        default_fit=True,
        requires=("PEPOCH",),
    ),
    ParameterSpec(
        name="F2",
        group="spin",
        derivative_group=DerivativeGroup.SPIN,
        dtype="longdouble",  # High-order terms need precision
        internal_unit="Hz/s^2",
        par_unit_str="s^-3",
        display_unit="Hz/s²",
        component_name="SpinComponent",
        requires=("PEPOCH",),
    ),
    ParameterSpec(
        name="F3",
        group="spin",
        derivative_group=DerivativeGroup.SPIN,
        dtype="longdouble",
        internal_unit="Hz/s^3",
        par_unit_str="s^-4",
        display_unit="Hz/s³",
        component_name="SpinComponent",
        requires=("PEPOCH",),
    ),
    ParameterSpec(
        name="PEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        component_name="SpinComponent",
        gui_visible=False,
        par_codec_name="epoch_mjd",
    ),
]

# DM parameters
_DM_PARAMS = [
    ParameterSpec(
        name="DM",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3",
        par_unit_str="pc cm^-3",
        display_unit="pc/cm³",
        aliases=("DM0",),
        component_name="DispersionComponent",
        default_fit=True,
    ),
    ParameterSpec(
        name="DM1",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3/yr",
        par_unit_str="pc cm^-3 yr^-1",
        display_unit="pc/cm³/yr",
        component_name="DispersionComponent",
        requires=("DMEPOCH",),
    ),
    ParameterSpec(
        name="DM2",
        group="dm",
        derivative_group=DerivativeGroup.DM,
        dtype="float64",
        internal_unit="pc/cm^3/yr^2",
        par_unit_str="pc cm^-3 yr^-2",
        display_unit="pc/cm³/yr²",
        component_name="DispersionComponent",
        requires=("DMEPOCH",),
    ),
    ParameterSpec(
        name="DMEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        component_name="DispersionComponent",
        gui_visible=False,
        par_codec_name="epoch_mjd",
    ),
]

# Astrometry parameters
_ASTROMETRY_PARAMS = [
    ParameterSpec(
        name="RAJ",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad",  # CRITICAL: radians internally
        par_unit_str="HH:MM:SS.sss",
        component_name="AstrometryComponent",
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
        component_name="AstrometryComponent",
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
        aliases=("PMRAC",),  # PM in RA*cos(DEC)
        component_name="AstrometryComponent",
        requires=("POSEPOCH",),
    ),
    ParameterSpec(
        name="PMDEC",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad/yr",
        par_unit_str="mas/yr",
        component_name="AstrometryComponent",
        requires=("POSEPOCH",),
    ),
    ParameterSpec(
        name="PX",
        group="astrometry",
        derivative_group=DerivativeGroup.ASTROMETRY,
        dtype="float64",
        internal_unit="rad",  # arcsec -> rad
        par_unit_str="mas",
        aliases=("PARALLAX",),
        component_name="AstrometryComponent",
    ),
    ParameterSpec(
        name="POSEPOCH",
        group="epoch",
        derivative_group=DerivativeGroup.EPOCH,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        component_name="AstrometryComponent",
        gui_visible=False,
        par_codec_name="epoch_mjd",
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
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="A1",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="lt-s",
        par_unit_str="lt-s",
        aliases=("ASINI",),
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="ECC",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("E",),
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="OM",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="rad",
        par_unit_str="deg",
        aliases=("OMEGA",),
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="T0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        component_name="BinaryComponent",
        par_codec_name="epoch_mjd",
    ),
    # ELL1 parameters
    ParameterSpec(
        name="TASC",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="MJD",
        par_unit_str="MJD",
        component_name="BinaryComponent",
        par_codec_name="epoch_mjd",
    ),
    ParameterSpec(
        name="EPS1",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="EPS2",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        component_name="BinaryComponent",
    ),
    # Post-Keplerian / Shapiro parameters
    ParameterSpec(
        name="SINI",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="M2",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="Msun",
        par_unit_str="Msun",
        display_unit="M☉",
        component_name="BinaryComponent",
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
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="XDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="lt-s/s",
        par_unit_str="lt-s/s",
        aliases=("A1DOT",),
        component_name="BinaryComponent",
    ),
    # Periastron advance (DD model)
    ParameterSpec(
        name="OMDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg/yr",
        par_unit_str="deg/yr",
        component_name="BinaryComponent",
    ),
    # Time dilation + gravitational redshift (DD model)
    ParameterSpec(
        name="GAMMA",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="BinaryComponent",
    ),
    # Eccentricity derivative (T2 model)
    ParameterSpec(
        name="EDOT",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="1/s",
        par_unit_str="",
        component_name="BinaryComponent",
    ),
    # Orthometric Shapiro parameters (ELL1H model)
    ParameterSpec(
        name="H3",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="H4",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="STIG",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("STIGMA",),
        component_name="BinaryComponent",
    ),
    # DD model relativistic deformation parameters
    ParameterSpec(
        name="DR",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="DTH",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="",
        par_unit_str="",
        aliases=("DTHETA",),
        component_name="BinaryComponent",
    ),
    # Aberration parameters (DD model)
    ParameterSpec(
        name="A0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="B0",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="BinaryComponent",
    ),
    # Kopeikin annual orbital parallax parameters (DDK model)
    ParameterSpec(
        name="KIN",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="deg",
        component_name="BinaryComponent",
    ),
    ParameterSpec(
        name="KOM",
        group="binary",
        derivative_group=DerivativeGroup.BINARY,
        dtype="float64",
        internal_unit="deg",
        par_unit_str="deg",
        component_name="BinaryComponent",
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
            component_name="BinaryComponent",
        )
    )

# FD (frequency-dependent) parameters
_FD_PARAMS = [
    ParameterSpec(
        name="FD1",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD2",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD3",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD4",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD5",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD6",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD7",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD8",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
    ParameterSpec(
        name="FD9",
        group="fd",
        derivative_group=DerivativeGroup.FD,
        dtype="float64",
        internal_unit="s",
        par_unit_str="s",
        component_name="FDComponent",
    ),
]

# Solar wind parameters
_SW_PARAMS = [
    ParameterSpec(
        name="NE_SW",
        group="solar_wind",
        derivative_group=DerivativeGroup.SOLAR_WIND,
        dtype="float64",
        internal_unit="cm^-3",
        par_unit_str="cm^-3",
        display_unit="cm⁻³",
        aliases=("NE1AU",),
        component_name="SolarWindComponent",
    ),
]

# Build the registry
PARAMETER_REGISTRY: Dict[str, ParameterSpec] = {}
_ALIAS_MAP: Dict[str, str] = {}  # alias -> canonical name

for spec in _SPIN_PARAMS + _DM_PARAMS + _ASTROMETRY_PARAMS + _BINARY_PARAMS + _FD_PARAMS + _SW_PARAMS:
    PARAMETER_REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        _ALIAS_MAP[alias] = spec.name


# =============================================================================
# Helper Functions
# =============================================================================

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
    >>> canonicalize_param_name('UNKNOWN')
    'UNKNOWN'
    """
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
    return PARAMETER_REGISTRY.get(canonical)


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
    'M☉'
    """
    spec = get_spec(name)
    if spec is None:
        return ""
    return spec.display_unit if spec.display_unit else spec.par_unit_str


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
        component_name="JumpComponent",
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

    # Check FD pattern - registered FD1..FD9, higher indices not yet implemented
    fd_match = re.match(r'^FD(\d+)$', canonical)
    if fd_match:
        fd_idx = int(fd_match.group(1))
        if 1 <= fd_idx <= 9:
            return True  # FD1-FD9 are registered
        raise ValueError(
            f"Parameter '{name}' (FD{fd_idx}) is out of range. "
            f"FD1-FD9 are registered; parametric families (FDn>9) are not yet implemented as first-class families."
        )

    # Check FB pattern - registered FB0..FB20, higher indices not yet implemented
    fb_match = re.match(r'^FB(\d+)$', canonical)
    if fb_match:
        fb_idx = int(fb_match.group(1))
        if 0 <= fb_idx <= 20:
            return True  # FB0-FB20 are registered
        raise ValueError(
            f"Parameter '{name}' (FB{fb_idx}) is out of range. "
            f"FB0-FB20 are registered; parametric families (FBn>20) are not yet implemented as first-class families."
        )

    # Unknown parameter
    raise ValueError(
        f"Parameter '{name}' is not registered. "
        f"Parametric families (JUMP1..N, DMX_*, FDn>9, FBn>20) are not yet implemented as first-class families."
    )
