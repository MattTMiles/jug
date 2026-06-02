"""Runtime convention profile for PINT-family and tempo2 compatibility modes.

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
    """Map API compatibility string to ``pint`` or ``tempo2``."""
    mode = str(compatibility).lower()
    if mode in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
        return "tempo2"
    return "pint"


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
    if engine_conventions is not None:
        validate_engine_profile_matches_compatibility(compatibility, engine_conventions)
        return engine_conventions
    return EngineConventionProfile.from_params(
        params,
        compatibility,
        implicit_tempo2_defaults=implicit_tempo2_defaults,
    )


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
    """Resolved runtime conventions for delay and residual computation.

    Parameters
    ----------
    compatibility
        ``pint`` or ``tempo2``.
    units
        ``TDB`` or ``TCB`` (from par ``UNITS``).
    timeeph
        Einstein delay implementation for TT↔TDB mapping context.
    t2cmethod
        TT(TAI) correction method label (metadata / future tropo hooks).
    dilatefreq
        Whether SS time dilation is applied to barycentric frequencies.
    planet_shapiro
        Include Jupiter–Neptune Shapiro delays.
    solar_shapiro
        Include solar Shapiro delay.
    correct_troposphere
        Apply neutral-atmosphere delay when supported.
    phase_mean_mode
        Weighted vs unweighted phase offset removal.
    implicit_tempo2_defaults
        When True and ``compatibility='tempo2'``, omitted TDB keywords follow
        tempo2 ``initialise.C`` / libstempo behaviour.
    ephem
        Normalised ephemeris name (e.g. ``de405``).
    """

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
    ephem: str = "de440"
    _sources: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_tempo2(self) -> bool:
        return self.compatibility == "tempo2"

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
        mode = _normalize_compatibility(compatibility)
        units = _keyword(
            params,
            "UNITS",
            str(params.get("_par_timescale", "TDB")).upper(),
        )
        tempo2_mode = mode == "tempo2"
        use_implicit = (
            tempo2_mode if implicit_tempo2_defaults is None else implicit_tempo2_defaults
        )

        sources: dict[str, str] = {}

        if use_implicit and tempo2_mode and units == "TDB":
            timeeph: TimeEph = "IF99"
            t2cmethod: T2CMethod = "IAU2000B"
            dilatefreq = True
            planet_shapiro = True
            correct_tropo = True
            for key, val in (
                ("TIMEEPH", timeeph),
                ("T2CMETHOD", t2cmethod),
                ("DILATEFREQ", "Y" if dilatefreq else "N"),
                ("PLANET_SHAPIRO", "Y" if planet_shapiro else "N"),
                ("CORRECT_TROPOSPHERE", "Y" if correct_tropo else "N"),
            ):
                sources[key] = "implicit_tempo2"
            if "TIMEEPH" in params:
                timeeph = _keyword(params, "TIMEEPH", "IF99")  # type: ignore[assignment]
                sources["TIMEEPH"] = "par"
            if "T2CMETHOD" in params:
                t2cmethod = _keyword(params, "T2CMETHOD", "IAU2000B")  # type: ignore[assignment]
                sources["T2CMETHOD"] = "par"
            if "DILATEFREQ" in params:
                dilatefreq = _flag_from_par(params, "DILATEFREQ")
                sources["DILATEFREQ"] = "par"
            if "PLANET_SHAPIRO" in params:
                planet_shapiro = _flag_from_par(params, "PLANET_SHAPIRO")
                sources["PLANET_SHAPIRO"] = "par"
            if "CORRECT_TROPOSPHERE" in params:
                correct_tropo = _flag_from_par(params, "CORRECT_TROPOSPHERE")
                sources["CORRECT_TROPOSPHERE"] = "par"
        else:
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
            phase_mean: PhaseMeanMode = "unweighted" if tempo2_mode else "weighted"
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
            implicit_tempo2_defaults=use_implicit and tempo2_mode,
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
            "ephem",
        }
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        return replace(self, **clean)


def default_engine_profile(compatibility: str = "pint") -> EngineConventionProfile:
    """Default runtime profile for a compatibility mode (no par file)."""
    mode = _normalize_compatibility(compatibility)
    return EngineConventionProfile(
        compatibility=mode,
        phase_mean_mode="unweighted" if mode == "tempo2" else "weighted",
        implicit_tempo2_defaults=mode == "tempo2",
    )
