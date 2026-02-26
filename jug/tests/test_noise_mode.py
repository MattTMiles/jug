"""Tests for jug.engine.noise_mode -- per-process noise toggles."""

import pytest

from jug.engine.noise_mode import (
    NoiseConfig,
    EFAC, EQUAD, ECORR, RED_NOISE, DM_NOISE,
)


# ---------------------------------------------------------------------------
# Auto-detection from par params
# ---------------------------------------------------------------------------

class TestFromPar:
    """Test NoiseConfig.from_par auto-detection."""

    def test_empty_params(self):
        nc = NoiseConfig.from_par({})
        assert nc.active_processes() == []
        assert not nc.has_any_noise()

    def test_efac_only(self):
        params = {"_noise_lines": ["T2EFAC -f L-wide 1.1"]}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(EFAC)
        assert not nc.is_enabled(EQUAD)
        assert not nc.is_enabled(ECORR)

    def test_equad_only(self):
        params = {"_noise_lines": ["EQUAD -f 430_ASP 0.5"]}
        nc = NoiseConfig.from_par(params)
        assert not nc.is_enabled(EFAC)
        assert nc.is_enabled(EQUAD)

    def test_ecorr_only(self):
        params = {"_noise_lines": ["ECORR -f L-wide 0.01"]}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(ECORR)
        assert not nc.is_enabled(EFAC)

    def test_mixed_white_noise(self):
        params = {"_noise_lines": [
            "T2EFAC -f L-wide 1.1",
            "T2EQUAD -f L-wide 0.1",
            "ECORR -f L-wide 0.01",
        ]}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(EFAC)
        assert nc.is_enabled(EQUAD)
        assert nc.is_enabled(ECORR)

    def test_red_noise_temponest(self):
        params = {"TNRedAmp": -13.5, "TNRedGam": 3.0}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(RED_NOISE)
        assert not nc.is_enabled(DM_NOISE)

    def test_red_noise_enterprise(self):
        params = {"RN_log10_A": -13.5, "RN_gamma": 3.0}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(RED_NOISE)

    def test_dm_noise_temponest(self):
        params = {"TNDMAmp": -14.0, "TNDMGam": 2.5}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(DM_NOISE)
        assert not nc.is_enabled(RED_NOISE)

    def test_dm_noise_enterprise(self):
        params = {"DM_log10_A": -14.0, "DM_gamma": 2.5}
        nc = NoiseConfig.from_par(params)
        assert nc.is_enabled(DM_NOISE)

    def test_full_noise_model(self):
        """Par with all noise types detected."""
        params = {
            "_noise_lines": [
                "T2EFAC -f L-wide 1.1",
                "T2EQUAD -f L-wide 0.1",
                "ECORR -f L-wide 0.01",
            ],
            "TNRedAmp": -13.5,
            "TNRedGam": 3.0,
            "TNDMAmp": -14.0,
            "TNDMGam": 2.5,
        }
        nc = NoiseConfig.from_par(params)
        assert nc.has_any_noise()
        assert len(nc.active_processes()) == 5

    def test_red_noise_partial_not_detected(self):
        """Only one of the red noise pair -- should NOT detect."""
        params = {"TNRedAmp": -13.5}  # missing TNRedGam
        nc = NoiseConfig.from_par(params)
        assert not nc.is_enabled(RED_NOISE)


# ---------------------------------------------------------------------------
# Toggle API
# ---------------------------------------------------------------------------

class TestToggleAPI:

    def test_toggle_on_off(self):
        nc = NoiseConfig(enabled={EFAC: True, EQUAD: False})
        assert nc.toggle(EFAC) is False
        assert nc.toggle(EQUAD) is True
        assert not nc.is_enabled(EFAC)
        assert nc.is_enabled(EQUAD)

    def test_toggle_unknown_process(self):
        """Toggling an untracked process starts it as True."""
        nc = NoiseConfig()
        assert nc.toggle("BandNoise") is True
        assert nc.is_enabled("BandNoise")

    def test_enable_disable(self):
        nc = NoiseConfig(enabled={EFAC: False})
        nc.enable(EFAC)
        assert nc.is_enabled(EFAC)
        nc.disable(EFAC)
        assert not nc.is_enabled(EFAC)

    def test_enable_all(self):
        nc = NoiseConfig(enabled={EFAC: False, EQUAD: False, RED_NOISE: True})
        nc.enable_all()
        assert all(nc.enabled.values())

    def test_disable_all(self):
        nc = NoiseConfig(enabled={EFAC: True, EQUAD: True, RED_NOISE: True})
        nc.disable_all()
        assert not any(nc.enabled.values())

    def test_active_processes(self):
        nc = NoiseConfig(enabled={EFAC: True, EQUAD: False, RED_NOISE: True})
        active = nc.active_processes()
        assert EFAC in active
        assert RED_NOISE in active
        assert EQUAD not in active


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_roundtrip(self):
        nc = NoiseConfig(enabled={EFAC: True, EQUAD: False, RED_NOISE: True})
        d = nc.to_dict()
        nc2 = NoiseConfig.from_dict(d)
        assert nc2.enabled == nc.enabled

    def test_is_plain_dict(self):
        nc = NoiseConfig(enabled={EFAC: True})
        d = nc.to_dict()
        assert isinstance(d, dict)
        assert d[EFAC] is True


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_repr(self):
        nc = NoiseConfig(enabled={EFAC: True, RED_NOISE: True, EQUAD: False})
        r = repr(nc)
        assert "EFAC" in r
        assert "RedNoise" in r

    def test_summary(self):
        nc = NoiseConfig(enabled={EFAC: True, EQUAD: False})
        s = nc.summary()
        assert "EFAC: ON" in s
        assert "EQUAD: OFF" in s

    def test_summary_empty(self):
        nc = NoiseConfig()
        s = nc.summary()
        assert "none" in s.lower()
