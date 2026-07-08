"""User-selectable conventions for tempo2 compatibility diagnostics.

These settings control how Phase A comparisons are run.  Product decisions
for ``compatibility="tempo2"`` acceptance remain locked (raw residuals,
unweighted phase mean); the knobs here are for reproducible diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResidualMetric = Literal["raw", "weighted_centered"]
Tempo2TdbDefaults = Literal["implicit_tempo2", "explicit_par"]
OracleTerms = Literal["libstempo_properties", "tempo2_general2_plugin"]
TermSet = Literal["core", "extended"]
PhaseMeanMode = Literal["weighted", "unweighted"]


@dataclass(frozen=True)
class DiagnosticConventions:
    """Conventions for residual and delay-term diagnostics.

    Parameters
    ----------
    residual_metric
        ``raw`` — compare residuals as returned (required for tempo2 acceptance).
        ``weighted_centered`` — subtract weighted mean before comparison; only
        for PINT-family cross-checks, never tempo2 acceptance.
    tempo2_tdb_defaults
        ``implicit_tempo2`` — apply tempo2 implicit defaults for omitted TDB
        keywords (IF99, DILATEFREQ on, etc.) when documenting parity context.
        ``explicit_par`` — use only what is present in the par file.
    oracle_terms
        ``libstempo_properties`` — use libstempo Python properties first.
        ``tempo2_general2_plugin`` — fall back to tempo2 plugin term dumps when
        a property is unavailable (test oracle only).
    term_set
        ``core`` — Roemer, Shapiro, DM/SW, residuals, barycentric frequency.
        ``extended`` — also TZR, binary proxy, FD/tropo where available.
    phase_mean_mode
        Override phase mean subtraction.  ``None`` derives from compatibility
        (unweighted for tempo2, weighted for pint).
    """

    residual_metric: ResidualMetric = "raw"
    tempo2_tdb_defaults: Tempo2TdbDefaults = "implicit_tempo2"
    oracle_terms: OracleTerms = "libstempo_properties"
    term_set: TermSet = "core"
    phase_mean_mode: PhaseMeanMode | None = None

    def resolved_phase_mean_mode(self, compatibility: str) -> PhaseMeanMode:
        if self.phase_mean_mode is not None:
            return self.phase_mean_mode
        mode = str(compatibility).lower()
        if mode in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
            return "unweighted"
        return "weighted"

    def tempo2_implicit_defaults_snapshot(self) -> dict[str, Any]:
        """Document tempo2 defaults for TDB pars with omitted keywords."""
        if self.tempo2_tdb_defaults != "implicit_tempo2":
            return {}
        return {
            "TIMEEPH": "IF99",
            "DILATEFREQ": "Y",
            "T2CMETHOD": "IAU2000B",
            "PLANET_SHAPIRO": "Y",
            "CORRECT_TROPOSPHERE": "Y",
            "NE_SW": 4.0,
            "EPHVER": 5,
        }

    def validate_for_tempo2_acceptance(self) -> None:
        if self.residual_metric != "raw":
            raise ValueError(
                "tempo2 acceptance requires residual_metric='raw'; "
                f"got {self.residual_metric!r}"
            )

    def apply_tempo2_implicit_defaults(self, compatibility: str) -> bool:
        """Whether tempo2 implicit TDB defaults may affect runtime physics.

        Diagnostic metadata may still record implicit defaults for any mode,
        but only tempo2 compatibility may change delay-term behavior.
        """
        mode = str(compatibility).lower()
        if mode not in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
            return False
        return self.tempo2_tdb_defaults == "implicit_tempo2"


from jug.residuals.engine_conventions import EngineConventionProfile, default_engine_profile


def resolve_planet_shapiro_enabled(
    params: dict[str, Any],
    profile: EngineConventionProfile,
) -> bool:
    """Resolve PLANET_SHAPIRO for the active engine profile."""
    if "PLANET_SHAPIRO" in params:
        return str(params["PLANET_SHAPIRO"]).upper() in ("1", "Y", "YES", "TRUE", "T")
    if profile.is_tempo2:
        return True
    return profile.planet_shapiro


def resolve_ne_sw_cm3(
    params: dict[str, Any],
    profile: EngineConventionProfile,
) -> float:
    """Resolve solar-wind electron density (cm^-3) for dispersion delays.

    Tempo2 ``initialise.C`` sets ``NE_SW_DEFAULT = 4`` even when the par file
    omits ``NE_SW``.  That default enters ``tdis2`` in ``formBats`` and must be
    included in pre-binary delay for tempo2 spin ``bbat`` parity.

    In tempo1-emulation mode (``EPHVER < 5``), ``preProcessSimple.C``
    unconditionally overrides ``ne_sw`` to 9.961 cm^-3 — even when the par
    file explicitly sets ``NE_SW`` (as the IPTA DR2 TDB pars do).
    """
    if getattr(profile, "tempo1_emulation", False):
        return 9.961
    if "NE_SW" in params:
        return float(params["NE_SW"])
    if profile.is_tempo2 and profile.implicit_tempo2_defaults:
        return 4.0
    return 0.0


def default_conventions(compatibility: str = "tempo2") -> DiagnosticConventions:
    """Return default diagnostic conventions for a compatibility mode."""
    mode = str(compatibility).lower()
    tempo2_mode = mode in ("tempo2", "tempo2-compatible", "tempo2_compatible")
    return DiagnosticConventions(
        phase_mean_mode=None,
        residual_metric="raw",
        tempo2_tdb_defaults="implicit_tempo2" if tempo2_mode else "explicit_par",
    )


@dataclass
class TermDiagnosticMetadata:
    """Provenance metadata attached to per-term diagnostic arrays."""

    compatibility: str
    provider: str
    geometry_backend: str
    term_sources: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility,
            "provider": self.provider,
            "geometry_backend": self.geometry_backend,
            "term_sources": dict(self.term_sources),
        }
