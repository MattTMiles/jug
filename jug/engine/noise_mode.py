"""Per-process noise configuration with auto-detection from .par file.

Each noise process (EFAC, EQUAD, ECORR, RedNoise, DMNoise, ...) has its own
on/off toggle.  The default state is determined by scanning the parsed par
parameters: if the par file contains the relevant keywords, that process
starts enabled.

The ``NOISE_REGISTRY`` is the single source of truth for noise process
metadata — canonical names, display labels, tooltips, ordering, and detection
logic.  The GUI derives all display information from here.

Usage
-----
>>> from jug.engine.noise_mode import NoiseConfig
>>> nc = NoiseConfig.from_par(params)
>>> nc.enabled["EFAC"]        # True if par had EFAC/T2EFAC lines
>>> nc.toggle("ECORR")        # flip one process
>>> nc.active_processes()      # list of currently-enabled names

Adding a new noise type requires only adding a ``NoiseProcessSpec`` entry to
``NOISE_REGISTRY`` (plus implementing the noise model and fitter logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from jug.noise.red_noise import RedNoiseProcess, DMNoiseProcess, ChromaticNoiseProcess


# ---------------------------------------------------------------------------
# Canonical process name constants
# ---------------------------------------------------------------------------

EFAC = "EFAC"
EQUAD = "EQUAD"
ECORR = "ECORR"
RED_NOISE = "RedNoise"
DM_NOISE = "DMNoise"
DMX = "DMX"
CHROMATIC_NOISE = "ChromaticNoise"
CW = "CW"
BWM = "BWM"
CHROMATIC_EVENT = "ChromaticEvent"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _has_efac(params: dict) -> bool:
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key in ("EFAC", "T2EFAC"):
            return True
    return False


def _has_equad(params: dict) -> bool:
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key in ("EQUAD", "T2EQUAD"):
            return True
    return False


def _has_ecorr(params: dict) -> bool:
    for line in params.get("_noise_lines", []):
        key = line.split()[0].upper()
        if key in ("ECORR", "TNECORR"):
            return True
    return False


def _has_red_noise(params: dict) -> bool:
    if "TNRedAmp" in params and "TNRedGam" in params:
        return True
    if "TNREDAMP" in params and "TNREDGAM" in params:
        return True
    if "RN_log10_A" in params and "RN_gamma" in params:
        return True
    if "RNAMP" in params and "RNIDX" in params:
        return True
    return False


def _has_dm_noise(params: dict) -> bool:
    if "TNDMAmp" in params and "TNDMGam" in params:
        return True
    if "TNDMAMP" in params and "TNDMGAM" in params:
        return True
    if "DM_log10_A" in params and "DM_gamma" in params:
        return True
    return False

def _has_chromatic_noise(params: dict) -> bool:
    if "TNChromAmp" in params and "TNChromGam" in params:
        return True
    if "TNCHROMAMP" in params and "TNCHROMGAM" in params:
        return True
    if "CHROM_log10_A" in params and "CHROM_gamma" in params:
        return True
    return False


def _has_dmx(params: dict) -> bool:
    return any(k.startswith("DMXR1_") for k in params)


# Deterministic signal detectors
def _has_cw(params: dict) -> bool:
    return "CW_LOG10_H" in params and "CW_LOG10_FGW" in params


def _has_bwm(params: dict) -> bool:
    return "BWM_LOG10_H" in params and "BWM_T0" in params


def _has_chromatic_event(params: dict) -> bool:
    return all(k in params for k in ("CHROMEV_EPOCH", "CHROMEV_AMP", "CHROMEV_TAU"))


# ---------------------------------------------------------------------------
# NoiseProcessSpec — single source of truth for each noise process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseProcessSpec:
    """Metadata for a noise process.

    All display information (label, tooltip, ordering) lives here so that the
    GUI never needs its own hardcoded copies.

    If ``impl_class`` is set, the GUI can introspect its dataclass fields to
    auto-generate editable parameter rows — no hardcoded param lists needed.
    """
    name: str
    label: str
    tooltip: str
    display_order: int
    detector: Callable[[dict], bool]
    is_power_law: bool = False
    impl_class: type = None  # type: ignore[assignment]


NOISE_REGISTRY: Dict[str, NoiseProcessSpec] = {}

_NOISE_SPECS = [
    NoiseProcessSpec(
        name=EFAC, label="EFAC",
        tooltip="Error scale factor per backend",
        display_order=0, detector=_has_efac,
    ),
    NoiseProcessSpec(
        name=EQUAD, label="EQUAD",
        tooltip="Added variance per backend (μs)",
        display_order=1, detector=_has_equad,
    ),
    NoiseProcessSpec(
        name=ECORR, label="ECORR",
        tooltip="Correlated noise within epochs (μs)",
        display_order=2, detector=_has_ecorr,
    ),
    NoiseProcessSpec(
        name=RED_NOISE, label="Achr. Red Noise",
        tooltip="Achromatic red noise (power-law)",
        display_order=3, detector=_has_red_noise, is_power_law=True,
        impl_class=RedNoiseProcess,
    ),
    NoiseProcessSpec(
        name=DM_NOISE, label="DM Noise",
        tooltip="Chromatic DM noise (power-law)",
        display_order=4, detector=_has_dm_noise, is_power_law=True,
        impl_class=DMNoiseProcess,
    ),
    NoiseProcessSpec(
        name=DMX, label="DMX",
        tooltip="Per-epoch dispersion measure",
        display_order=5, detector=_has_dmx,
    ),
    NoiseProcessSpec(
        name=CHROMATIC_NOISE, label="Chromatic Noise",
        tooltip="Chromatic noise (power-law)",
        display_order=6, detector=_has_chromatic_noise, is_power_law=True,
        impl_class=ChromaticNoiseProcess,
    ),
    # Deterministic signals (subtract-only, no realisation)
    NoiseProcessSpec(
        name=CW, label="CW Signal",
        tooltip="Continuous gravitational wave (Earth term)",
        display_order=10, detector=_has_cw,
    ),
    NoiseProcessSpec(
        name=BWM, label="Burst w/ Memory",
        tooltip="Gravitational wave burst with memory",
        display_order=11, detector=_has_bwm,
    ),
    NoiseProcessSpec(
        name=CHROMATIC_EVENT, label="Chromatic Event",
        tooltip="Frequency-dependent transient (DM/scattering event)",
        display_order=12, detector=_has_chromatic_event,
    ),
]

for _spec in _NOISE_SPECS:
    NOISE_REGISTRY[_spec.name] = _spec


# ---------------------------------------------------------------------------
# Helper functions for GUI / external consumers
# ---------------------------------------------------------------------------

def get_noise_label(name: str) -> str:
    """Return the display label for a noise process, or *name* if unknown."""
    spec = NOISE_REGISTRY.get(name)
    return spec.label if spec else name


def get_noise_tooltip(name: str) -> str:
    """Return the tooltip for a noise process, or empty string if unknown."""
    spec = NOISE_REGISTRY.get(name)
    return spec.tooltip if spec else ""


def get_noise_display_order() -> List[str]:
    """Return noise process names sorted by display_order."""
    return [s.name for s in sorted(NOISE_REGISTRY.values(),
                                    key=lambda s: s.display_order)]


def get_all_noise_names() -> set:
    """Return the set of all registered noise process names."""
    return set(NOISE_REGISTRY.keys())


# LaTeX-style markup → Unicode rendering
import re as _re

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def render_label(text: str) -> str:
    r"""Convert LaTeX-style markup to Unicode for GUI display.

    Supports ``\greek``, ``_{sub}``, and ``^{sup}`` notation so that
    ``_gui_labels`` can be written in plain ASCII::

        >>> render_label(r"\gamma (spectral)")
        'γ (spectral)'
        >>> render_label("log_{10}(A)")
        'log₁₀(A)'
        >>> render_label(r"Chrom. index \beta")
        'Chrom. index β'
    """
    # Greek letters: \alpha → α
    text = _re.sub(
        r"\\([A-Za-z]+)",
        lambda m: _GREEK.get(m.group(1), m.group(0)),
        text,
    )
    # Subscripts: _{10} → ₁₀
    text = _re.sub(
        r"_\{([0-9]+)\}",
        lambda m: m.group(1).translate(_SUBSCRIPT_DIGITS),
        text,
    )
    # Superscripts: ^{2} → ²
    text = _re.sub(
        r"\^\{([0-9]+)\}",
        lambda m: m.group(1).translate(_SUPERSCRIPT_DIGITS),
        text,
    )
    return text


def get_impl_param_defs(proc_name: str, params: dict = None) -> List[dict]:
    """Return GUI-displayable parameter definitions from the impl class.

    Introspects the dataclass fields of the implementation class and uses
    its ``_gui_labels`` dict for friendly display names and ``_par_keys``
    dict to resolve par-file values.
    Returns a list of dicts suitable for ``NoiseProcessRow``.
    """
    import dataclasses as _dc

    spec = NOISE_REGISTRY.get(proc_name)
    if spec is None or spec.impl_class is None:
        return []

    cls = spec.impl_class
    if not _dc.is_dataclass(cls):
        return []

    gui_labels = getattr(cls, '_gui_labels', {})
    par_keys = getattr(cls, '_par_keys', {})
    gui_defaults = getattr(cls, '_gui_defaults', {})
    result = []
    for f in _dc.fields(cls):
        label = render_label(gui_labels.get(f.name, f.name))
        default = f.default if f.default is not _dc.MISSING else gui_defaults.get(f.name, "")

        # Try to get value from par params via _par_keys mapping
        value = None
        if params and f.name in par_keys:
            for pk in par_keys[f.name]:
                if pk in params:
                    value = params[pk]
                    break
        if value is None:
            value = default

        # Format for display
        if isinstance(value, float):
            display = f"{value:.4f}"
        elif isinstance(value, int) or (isinstance(value, float) and value == int(value)):
            display = str(int(value))
        else:
            display = str(value)

        result.append({
            "key": f.name,
            "label": label,
            "value": display,
            "editable": True,
        })
    return result


def get_par_key_for_field(proc_name: str, field_name: str) -> Optional[str]:
    """Return the canonical (first) par-file key for a dataclass field.

    Used internally by :func:`write_noise_params`.
    """
    spec = NOISE_REGISTRY.get(proc_name)
    if spec is None or spec.impl_class is None:
        return None
    par_keys = getattr(spec.impl_class, '_par_keys', {})
    aliases = par_keys.get(field_name, [])
    return aliases[0] if aliases else None


def write_noise_params(proc_name: str, field_values: dict, params: dict) -> None:
    """Write noise field values into a par-params dict using canonical par keys.

    This is the inverse of ``from_par()`` — given GUI field names and values,
    it writes them into the session params dict so ``from_par()`` can read them.

    Parameters
    ----------
    proc_name : str
        Registry name (e.g. ``"ChromaticNoise"``).
    field_values : dict
        ``{field_name: value}`` — field names as returned by
        ``get_impl_param_defs()``, values as strings or numbers.
    params : dict
        The mutable session params dict to update in-place.
    """
    for field_name, value in field_values.items():
        par_key = get_par_key_for_field(proc_name, field_name)
        if par_key is None:
            continue
        try:
            params[par_key] = float(value)
        except (ValueError, TypeError):
            params[par_key] = value


def compute_noise_realization(
    proc_name: str,
    params: dict,
    toas_mjd,
    residuals_sec,
    errors_sec,
    freq_mhz=None,
):
    """Compute MAP noise realization for any registered power-law process.

    Returns realization in seconds, or None if the process can't be parsed.
    """
    spec = NOISE_REGISTRY.get(proc_name)
    if spec is None or spec.impl_class is None:
        return None

    cls = spec.impl_class
    from_par = getattr(cls, 'from_par', None)
    if from_par is None:
        return None

    proc = from_par(params)
    if proc is None:
        return None

    from jug.noise.red_noise import realize_noise_generic
    return realize_noise_generic(proc, toas_mjd, residuals_sec, errors_sec, freq_mhz)


# ---------------------------------------------------------------------------
# NoiseConfig
# ---------------------------------------------------------------------------

class NoiseConfig:
    """Per-process noise toggle configuration.

    Parameters
    ----------
    enabled : dict[str, bool], optional
        Initial enabled state.  Missing processes default to False.
    """

    def __init__(self, enabled: Optional[Dict[str, bool]] = None):
        self.enabled: Dict[str, bool] = dict(enabled) if enabled else {}

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_par(cls, params: dict) -> "NoiseConfig":
        """Build config by auto-detecting noise processes in parsed par params.

        Parameters
        ----------
        params : dict
            Parsed parameter dictionary (from ``read_par``).  Must include
            ``_noise_lines`` for white-noise detection and regular param keys
            for red/DM noise.

        Returns
        -------
        NoiseConfig
            Config with each detected process enabled.
        """
        enabled = {}
        for name, spec in NOISE_REGISTRY.items():
            enabled[name] = spec.detector(params)
        return cls(enabled)

    # -- Queries -----------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """Return True if *name* is a known and enabled process."""
        return self.enabled.get(name, False)

    def active_processes(self) -> List[str]:
        """Return list of currently-enabled process names."""
        return [k for k, v in self.enabled.items() if v]

    def has_any_noise(self) -> bool:
        """Return True if at least one noise process is enabled."""
        return any(self.enabled.values())

    # -- Mutations ---------------------------------------------------------

    def toggle(self, name: str) -> bool:
        """Flip the toggle for *name* and return the new state.

        If the process was not previously tracked it becomes True (enabled).
        """
        new_state = not self.enabled.get(name, False)
        self.enabled[name] = new_state
        return new_state

    def enable(self, name: str) -> None:
        """Enable a single process."""
        self.enabled[name] = True

    def disable(self, name: str) -> None:
        """Disable a single process."""
        self.enabled[name] = False

    def enable_all(self) -> None:
        """Enable every known process."""
        for k in self.enabled:
            self.enabled[k] = True

    def disable_all(self) -> None:
        """Disable every known process."""
        for k in self.enabled:
            self.enabled[k] = False

    # -- Serialization (for session save/load) -----------------------------

    def to_dict(self) -> Dict[str, bool]:
        """Return a plain dict suitable for JSON serialization."""
        return dict(self.enabled)

    @classmethod
    def from_dict(cls, d: Dict[str, bool]) -> "NoiseConfig":
        """Restore from a serialized dict."""
        return cls(enabled=d)

    # -- Display -----------------------------------------------------------

    def __repr__(self) -> str:
        active = ", ".join(self.active_processes()) or "(none)"
        return f"NoiseConfig(active=[{active}])"

    def summary(self) -> str:
        """Human-readable summary of noise toggle states."""
        lines = []
        for name in sorted(self.enabled):
            state = "ON" if self.enabled[name] else "OFF"
            lines.append(f"  {name}: {state}")
        return "Noise processes:\n" + "\n".join(lines) if lines else "Noise processes: (none detected)"
