"""Runtime convention profile for PINT-family compatibility mode.

``EngineConventionProfile`` owns physics-relevant choices (timescales, implicit
defaults, Shapiro/tropo flags, phase-mean policy).  Test-only comparison knobs
remain in ``diagnostic_conventions.DiagnosticConventions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

PhaseMeanMode = Literal["weighted", "unweighted"]
TimeEph = Literal["IF99", "FB90"]
T2CMethod = Literal["IAU2000B", "TEMPO"]


def normalize_compatibility_mode(compatibility: str) -> str:
    """Map API compatibility string to ``pint``.

    Raises
    ------
    ValueError
        If *compatibility* is not ``pint`` (tempo2 mode is not available in this
        portable build).
    """
    mode = str(compatibility).lower()
    if mode == "pint":
        return "pint"
    if mode == "tempo2":
        raise ValueError(
            f"compatibility={compatibility!r} is not supported in this build; "
            "only 'pint' is available."
        )
    raise ValueError(
        f"Unknown compatibility={compatibility!r}; expected 'pint'"
    )


def _normalize_compatibility(compatibility: str) -> str:
    return normalize_compatibility_mode(compatibility)


def validate_engine_profile_matches_compatibility(
    compatibility: str,
    engine_profile: EngineConventionProfile,
) -> None:
    """Reject mixed-mode runs where profile and API compatibility disagree."""
    mode = normalize_compatibility_mode(compatibility)
    if engine_profile.compatibility != mode:
        raise ValueError(
            f"engine_conventions.compatibility={engine_profile.compatibility!r} "
            f"does not match compatibility={mode!r}. "
            "Pass a profile built for the same mode (or omit engine_conventions)."
        )


def resolve_engine_profile(
    params: dict[str, Any],
    compatibility: str,
    *,
    engine_conventions: EngineConventionProfile | None = None,
    implicit_tempo2_defaults: bool | None = None,
) -> EngineConventionProfile:
    """Return runtime profile for *compatibility*, validating against an explicit profile."""
    del implicit_tempo2_defaults
    if engine_conventions is not None:
        validate_engine_profile_matches_compatibility(compatibility, engine_conventions)
        return engine_conventions
    return EngineConventionProfile.from_params(params, compatibility)


def _flag_from_par(params: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in params:
        return default
    flag = str(params[key]).upper().strip()
    return flag in ("Y", "YES", "TRUE", "1", "T")


def _keyword(params: dict[str, Any], key: str, default: str) -> str:
    if key not in params:
        return default
    return str(params[key]).upper().strip()


@dataclass(frozen=True)
class EngineConventionProfile:
    """Resolved runtime conventions for delay and residual computation."""

    compatibility: str = "pint"
    units: str = "TDB"
    timeeph: TimeEph = "IF99"
    t2cmethod: T2CMethod = "IAU2000B"
    dilatefreq: bool = False
    planet_shapiro: bool = False
    solar_shapiro: bool = True
    correct_troposphere: bool = False
    phase_mean_mode: PhaseMeanMode = "weighted"
    implicit_tempo2_defaults: bool = False
    tempo1_emulation: bool = False
    ephem: str = "de440"
    _sources: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_tempo2(self) -> bool:
        return False

    @property
    def is_tcb(self) -> bool:
        return self.units.upper() == "TCB"

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility,
            "units": self.units,
            "timeeph": self.timeeph,
            "t2cmethod": self.t2cmethod,
            "dilatefreq": self.dilatefreq,
            "planet_shapiro": self.planet_shapiro,
            "solar_shapiro": self.solar_shapiro,
            "correct_troposphere": self.correct_troposphere,
            "phase_mean_mode": self.phase_mean_mode,
            "implicit_tempo2_defaults": self.implicit_tempo2_defaults,
            "tempo1_emulation": self.tempo1_emulation,
            "ephem": self.ephem,
            "sources": dict(self._sources),
        }

    @classmethod
    def from_params(
        cls,
        params: dict[str, Any],
        compatibility: str = "pint",
        *,
        implicit_tempo2_defaults: bool | None = None,
        phase_mean_mode: PhaseMeanMode | None = None,
    ) -> EngineConventionProfile:
        """Build a profile from parsed par parameters."""
        del implicit_tempo2_defaults
        mode = _normalize_compatibility(compatibility)
        units = _keyword(
            params,
            "UNITS",
            str(params.get("_par_timescale", "TDB")).upper(),
        )

        sources: dict[str, str] = {}
        timeeph = _keyword(params, "TIMEEPH", "IF99")  # type: ignore[assignment]
        t2cmethod = _keyword(params, "T2CMETHOD", "IAU2000B")  # type: ignore[assignment]
        dilatefreq = _flag_from_par(params, "DILATEFREQ", default=False)
        planet_shapiro = _flag_from_par(params, "PLANET_SHAPIRO", default=False)
        correct_tropo = _flag_from_par(params, "CORRECT_TROPOSPHERE", default=False)
        for key in ("TIMEEPH", "T2CMETHOD", "DILATEFREQ", "PLANET_SHAPIRO", "CORRECT_TROPOSPHERE"):
            if key in params:
                sources[key] = "par"

        solar_shapiro = not _flag_from_par(params, "NO_SS_SHAPIRO", default=False)
        if "NO_SS_SHAPIRO" in params:
            sources["NO_SS_SHAPIRO"] = "par"

        if phase_mean_mode is None:
            phase_mean: PhaseMeanMode = "weighted"
        else:
            phase_mean = phase_mean_mode

        ephem = str(params.get("EPHEM", "DE440")).lower()

        return cls(
            compatibility=mode,
            units=units,
            timeeph=timeeph,
            t2cmethod=t2cmethod,
            dilatefreq=dilatefreq,
            planet_shapiro=planet_shapiro,
            solar_shapiro=solar_shapiro,
            correct_troposphere=correct_tropo,
            phase_mean_mode=phase_mean,
            implicit_tempo2_defaults=False,
            tempo1_emulation=False,
            ephem=ephem,
            _sources=sources,
        )

    def with_overrides(self, **kwargs: Any) -> EngineConventionProfile:
        """Return a copy with selected fields replaced."""
        allowed = {
            "compatibility",
            "units",
            "timeeph",
            "t2cmethod",
            "dilatefreq",
            "planet_shapiro",
            "solar_shapiro",
            "correct_troposphere",
            "phase_mean_mode",
            "implicit_tempo2_defaults",
            "tempo1_emulation",
            "ephem",
        }
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        return replace(self, **clean)


def default_engine_profile(compatibility: str = "pint") -> EngineConventionProfile:
    """Default runtime profile for pint mode (no par file)."""
    mode = _normalize_compatibility(compatibility)
    return EngineConventionProfile(
        compatibility=mode,
        phase_mean_mode="weighted",
        implicit_tempo2_defaults=False,
    )