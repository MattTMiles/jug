"""
Integration tests for noise integration in the fitting pipeline.

Tests prove that:
1. Red noise / DM noise Fourier basis is built and augments the design matrix
2. EFAC/EQUAD white noise changes fit results vs raw errors
3. NoiseConfig gating disables noise effects (returns to baseline)
4. DMX design matrix is built when DMX ranges are present
"""
import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths to test data
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pulsars"
PAR_NOISE = DATA_DIR / "pulsars_w_noise" / "J1909-3744_tdb.par"
PAR_NO_NOISE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"

# Skip all tests if data files are missing
pytestmark = pytest.mark.skipif(
    not PAR_NOISE.exists() or not TIM_FILE.exists(),
    reason="Test data files not found",
)

FIT_PARAMS = ["F0", "F1"]
MAX_ITER = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_setup_from_files(par, tim, noise_config=None):
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_files
    return _build_general_fit_setup_from_files(
        par, tim, FIT_PARAMS, None, verbose=False, noise_config=noise_config
    )


def _fit_via_session(par, tim, noise_config=None):
    from jug.engine.session import TimingSession
    session = TimingSession(par, tim)
    return session.fit_parameters(
        FIT_PARAMS, max_iter=MAX_ITER, noise_config=noise_config
    )


# ===========================================================================
# Structural tests – verify the setup dataclass has the right fields
# ===========================================================================
class TestNoiseSetupStructure:
    """Verify that setup objects contain expected noise structures."""

    def test_red_noise_basis_built(self):
        """Par file with TNREDAMP/TNREDGAM/TNREDC → Fourier basis is built."""
        setup = _build_setup_from_files(PAR_NOISE, TIM_FILE)
        assert setup.red_noise_basis is not None, "Red noise basis should be built"
        assert setup.red_noise_prior is not None, "Red noise prior should be built"
        n_toas = setup.toas_mjd.shape[0]
        n_harmonics = setup.red_noise_basis.shape[1]
        assert setup.red_noise_basis.shape[0] == n_toas
        assert n_harmonics == 330  # 165 harmonics * 2 (sin+cos)
        assert setup.red_noise_prior.shape == (n_harmonics,)

    def test_dm_noise_basis_built(self):
        """Par file with TNDMAMP/TNDMGAM/TNDMC → DM noise Fourier basis is built."""
        setup = _build_setup_from_files(PAR_NOISE, TIM_FILE)
        assert setup.dm_noise_basis is not None, "DM noise basis should be built"
        assert setup.dm_noise_prior is not None, "DM noise prior should be built"
        n_toas = setup.toas_mjd.shape[0]
        n_harmonics = setup.dm_noise_basis.shape[1]
        assert setup.dm_noise_basis.shape[0] == n_toas
        assert n_harmonics == 330  # 165 harmonics * 2
        assert setup.dm_noise_prior.shape == (n_harmonics,)

    def test_no_noise_par_has_no_basis(self):
        """Par file without TN* keys → no Fourier basis."""
        setup = _build_setup_from_files(PAR_NO_NOISE, TIM_FILE)
        assert setup.red_noise_basis is None
        assert setup.red_noise_prior is None
        assert setup.dm_noise_basis is None
        assert setup.dm_noise_prior is None

    def test_noise_config_auto_detected(self):
        """NoiseConfig is auto-populated from par file."""
        setup = _build_setup_from_files(PAR_NOISE, TIM_FILE)
        nc = setup.noise_config
        assert nc is not None
        assert nc.is_enabled("EFAC")
        assert nc.is_enabled("EQUAD")
        assert nc.is_enabled("RedNoise")
        assert nc.is_enabled("DMNoise")

    def test_noise_config_override_disables_red_noise(self):
        """Passing a NoiseConfig with RedNoise=False suppresses Fourier basis."""
        from jug.engine.noise_mode import NoiseConfig
        nc = NoiseConfig.from_par({})
        # Enable white noise only
        nc.enabled["EFAC"] = True
        nc.enabled["EQUAD"] = True
        nc.enabled["RedNoise"] = False
        nc.enabled["DMNoise"] = False

        setup = _build_setup_from_files(PAR_NOISE, TIM_FILE, noise_config=nc)
        assert setup.red_noise_basis is None, "Red noise should be suppressed"
        assert setup.dm_noise_basis is None, "DM noise should be suppressed"


# ===========================================================================
# Numerical tests – verify that noise changes fit results
# ===========================================================================
class TestNoiseNumericalEffects:
    """Verify that enabling/disabling noise measurably changes fit results."""

    def test_white_noise_changes_rms(self):
        """EFAC/EQUAD changes error bars and therefore RMS."""
        from jug.engine.noise_mode import NoiseConfig

        # All noise enabled (auto-detect)
        result_with = _fit_via_session(PAR_NOISE, TIM_FILE)

        # All noise disabled
        nc_off = NoiseConfig()
        for k in list(nc_off.enabled.keys()):
            nc_off.enabled[k] = False
        result_without = _fit_via_session(PAR_NOISE, TIM_FILE, noise_config=nc_off)

        # RMS should differ (EFAC/EQUAD inflate error bars → higher weighted RMS)
        assert result_with["final_rms"] != result_without["final_rms"], (
            "Enabling EFAC/EQUAD should change weighted RMS"
        )

    def test_disabling_all_noise_matches_no_noise_par(self):
        """Disabling all noise on noise par ≈ fitting with no-noise par."""
        from jug.engine.noise_mode import NoiseConfig

        nc_off = NoiseConfig()
        for k in list(nc_off.enabled.keys()):
            nc_off.enabled[k] = False

        result_disabled = _fit_via_session(PAR_NOISE, TIM_FILE, noise_config=nc_off)
        result_no_noise_par = _fit_via_session(PAR_NO_NOISE, TIM_FILE)

        # F0 should match closely (same raw errors, same data)
        f0_diff = abs(
            result_disabled["final_params"]["F0"]
            - result_no_noise_par["final_params"]["F0"]
        )
        assert f0_diff < 1e-10, f"F0 should match when noise is disabled: diff={f0_diff}"

        # RMS should match exactly (same weighting)
        rms_diff = abs(result_disabled["final_rms"] - result_no_noise_par["final_rms"])
        assert rms_diff < 1e-6, f"RMS should match: diff={rms_diff}"

    def test_fourier_basis_augments_design_matrix(self):
        """With red/DM noise, the setup has Fourier basis columns that augment solving."""
        setup_with = _build_setup_from_files(PAR_NOISE, TIM_FILE)
        setup_without = _build_setup_from_files(PAR_NO_NOISE, TIM_FILE)

        # The setup with noise should have Fourier basis columns
        n_extra = 0
        if setup_with.red_noise_basis is not None:
            n_extra += setup_with.red_noise_basis.shape[1]
        if setup_with.dm_noise_basis is not None:
            n_extra += setup_with.dm_noise_basis.shape[1]

        assert n_extra > 0, "Noise par should produce extra Fourier columns"
        assert n_extra == 660, f"Expected 660 Fourier columns (330 red + 330 DM), got {n_extra}"

        # Without noise, no extra columns
        n_extra_no = 0
        if setup_without.red_noise_basis is not None:
            n_extra_no += setup_without.red_noise_basis.shape[1]
        if setup_without.dm_noise_basis is not None:
            n_extra_no += setup_without.dm_noise_basis.shape[1]
        assert n_extra_no == 0

    def test_red_dm_noise_prior_positive(self):
        """Prior variances must be positive and finite."""
        setup = _build_setup_from_files(PAR_NOISE, TIM_FILE)
        assert np.all(setup.red_noise_prior > 0), "Red noise prior should be positive"
        assert np.all(np.isfinite(setup.red_noise_prior)), "Red noise prior should be finite"
        assert np.all(setup.dm_noise_prior > 0), "DM noise prior should be positive"
        assert np.all(np.isfinite(setup.dm_noise_prior)), "DM noise prior should be finite"


# ===========================================================================
# NoiseConfig gating tests
# ===========================================================================
class TestNoiseConfigGating:
    """Verify that NoiseConfig correctly gates noise effects."""

    def test_efac_equad_gating(self):
        """Disabling EFAC/EQUAD should change error bars."""
        from jug.engine.noise_mode import NoiseConfig

        # EFAC/EQUAD ON
        nc_on = NoiseConfig.from_par({})
        nc_on.enabled["EFAC"] = True
        nc_on.enabled["EQUAD"] = True
        nc_on.enabled["RedNoise"] = False
        nc_on.enabled["DMNoise"] = False
        setup_on = _build_setup_from_files(PAR_NOISE, TIM_FILE, noise_config=nc_on)

        # EFAC/EQUAD OFF
        nc_off = NoiseConfig()
        for k in list(nc_off.enabled.keys()):
            nc_off.enabled[k] = False
        setup_off = _build_setup_from_files(PAR_NOISE, TIM_FILE, noise_config=nc_off)

        # Error bars should differ
        assert not np.allclose(setup_on.errors_us, setup_off.errors_us), (
            "EFAC/EQUAD should modify error bars"
        )

    def test_noise_config_roundtrip_via_session(self):
        """NoiseConfig override persists through session.fit_parameters()."""
        from jug.engine.noise_mode import NoiseConfig

        nc_off = NoiseConfig()
        for k in list(nc_off.enabled.keys()):
            nc_off.enabled[k] = False

        result = _fit_via_session(PAR_NOISE, TIM_FILE, noise_config=nc_off)
        # Should match no-noise baseline
        assert abs(result["final_rms"] - 0.405502) < 0.001, (
            f"With all noise off, RMS should be ~0.405502, got {result['final_rms']}"
        )

    def test_session_default_auto_detects_noise(self):
        """Without noise_config override, session auto-detects from par file."""
        result = _fit_via_session(PAR_NOISE, TIM_FILE)
        # With noise model active, fit should converge to a reasonable RMS
        assert result["final_rms"] > 0.1, (
            f"With noise, RMS should be >0.1, got {result['final_rms']}"
        )


# ===========================================================================
# DMX integration tests (B1855+09 from PINT test data)
# ===========================================================================
PINT_DATADIR = Path("/home/mattm/soft/PINT/tests/datafile")
PAR_DMX = PINT_DATADIR / "B1855+09_NANOGrav_9yv1.gls.par"
TIM_DMX = PINT_DATADIR / "B1855+09_NANOGrav_9yv1.tim"

_skip_dmx = pytest.mark.skipif(
    not PAR_DMX.exists() or not TIM_DMX.exists(),
    reason="B1855+09 PINT test data not found",
)


@_skip_dmx
class TestDMXIntegration:
    """Verify DMX design matrix is built and affects fit results."""

    def test_dmx_design_matrix_built(self):
        """Par file with DMX ranges → DMX design matrix is built."""
        setup = _build_setup_from_files(PAR_DMX, TIM_DMX)
        assert setup.dmx_design_matrix is not None, "DMX design matrix should be built"
        assert setup.dmx_labels is not None
        n_toas = setup.toas_mjd.shape[0]
        assert setup.dmx_design_matrix.shape == (n_toas, 72), (
            f"Expected (4005, 72), got {setup.dmx_design_matrix.shape}"
        )
        assert len(setup.dmx_labels) == 72

    def test_dmx_red_noise_coexist(self):
        """B1855+09 has both red noise and DMX — both should be built."""
        setup = _build_setup_from_files(PAR_DMX, TIM_DMX)
        assert setup.red_noise_basis is not None, "Red noise basis should be built"
        assert setup.red_noise_basis.shape[1] == 90  # 45 harmonics * 2
        assert setup.dmx_design_matrix is not None, "DMX design matrix should be built"
        assert setup.dmx_design_matrix.shape[1] == 72

    def test_dmx_ecorr_coexist(self):
        """B1855+09 has ECORR — GLS basis should be built alongside DMX."""
        setup = _build_setup_from_files(PAR_DMX, TIM_DMX)
        # ECORR is now handled via the GLS basis (not whitener)
        assert setup.ecorr_basis is not None, "ECORR basis should be built"
        assert setup.ecorr_prior is not None, "ECORR prior should be built"
        assert setup.ecorr_whitener is None, "Whitener should be disabled when ECORR is in GLS basis"
        assert setup.dmx_design_matrix is not None

    def test_dmx_noise_config(self):
        """NoiseConfig should detect EFAC/EQUAD/ECORR/RedNoise for B1855+09."""
        setup = _build_setup_from_files(PAR_DMX, TIM_DMX)
        nc = setup.noise_config
        assert nc.is_enabled("EFAC")
        assert nc.is_enabled("EQUAD")
        assert nc.is_enabled("ECORR")
        assert nc.is_enabled("RedNoise")
        # DMNoise not present in B1855+09
        assert not nc.is_enabled("DMNoise")
