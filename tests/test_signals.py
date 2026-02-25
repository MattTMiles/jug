"""Tests for the deterministic signal framework (jug/signals/).

Tests:
  - Signal registry and detection
  - CW waveform known-answer test
  - BWM waveform known-answer test
  - Chromatic event waveform known-answer test
  - Par-file round-trip (parse → detect → compute)
  - Noise registry entries for signals
"""

import tempfile
import os

import numpy as np
import numpy.testing as npt
import pytest


# ---------------------------------------------------------------------------
# Registry and detection
# ---------------------------------------------------------------------------

class TestSignalRegistry:
    def test_registry_populated(self):
        from jug.signals import SIGNAL_REGISTRY
        assert "CW" in SIGNAL_REGISTRY
        assert "BWM" in SIGNAL_REGISTRY
        assert "ChromaticEvent" in SIGNAL_REGISTRY

    def test_detect_cw(self):
        from jug.signals import detect_signals
        params = {
            "CW_LOG10_H": -14.5,
            "CW_COS_GWTHETA": 0.3,
            "CW_GWPHI": 1.2,
            "CW_LOG10_FGW": -8.2,
            "CW_PHASE0": 0.5,
            "CW_COS_INC": 0.7,
            "CW_PSI": 0.8,
            "RAJ": 1.0,
            "DECJ": -0.5,
        }
        signals = detect_signals(params)
        names = [s.signal_name for s in signals]
        assert "CW" in names

    def test_detect_bwm(self):
        from jug.signals import detect_signals
        params = {
            "BWM_LOG10_H": -14.0,
            "BWM_COS_GWTHETA": 0.5,
            "BWM_GWPHI": 2.0,
            "BWM_T0": 55000.0,
            "BWM_POL": 0.3,
            "RAJ": 0.5,
            "DECJ": 0.2,
        }
        signals = detect_signals(params)
        names = [s.signal_name for s in signals]
        assert "BWM" in names

    def test_detect_chromatic(self):
        from jug.signals import detect_signals
        params = {
            "CHROMEV_EPOCH": 55100.0,
            "CHROMEV_AMP": 1e-6,
            "CHROMEV_TAU": 30.0,
        }
        signals = detect_signals(params)
        names = [s.signal_name for s in signals]
        assert "ChromaticEvent" in names

    def test_no_detect_missing_keys(self):
        from jug.signals import detect_signals
        params = {"CW_LOG10_H": -14.0}  # missing CW_LOG10_FGW
        signals = detect_signals(params)
        names = [s.signal_name for s in signals]
        assert "CW" not in names


# ---------------------------------------------------------------------------
# CW waveform
# ---------------------------------------------------------------------------

class TestContinuousWave:
    def test_output_shape(self):
        from jug.signals.continuous_wave import ContinuousWaveSignal
        sig = ContinuousWaveSignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=-8.0, phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        toas = np.linspace(50000.0, 55000.0, 1000)
        waveform = sig.compute_waveform(toas)
        assert waveform.shape == (1000,)

    def test_amplitude_scaling(self):
        """Doubling log10_h by +1 (10× amplitude) should scale waveform by 10×."""
        from jug.signals.continuous_wave import ContinuousWaveSignal
        toas = np.linspace(50000.0, 55000.0, 500)
        sig1 = ContinuousWaveSignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=-8.0, phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        sig2 = ContinuousWaveSignal(
            log10_h=-13.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=-8.0, phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        w1 = sig1.compute_waveform(toas)
        w2 = sig2.compute_waveform(toas)
        # w2 should be 10× w1 (float32 limits precision)
        npt.assert_allclose(w2, 10.0 * w1, rtol=1e-4)

    def test_frequency_scaling(self):
        """Frequency doubled → waveform oscillates 2× faster, amplitude halved."""
        from jug.signals.continuous_wave import ContinuousWaveSignal
        toas = np.linspace(50000.0, 55000.0, 10000)

        sig1 = ContinuousWaveSignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=-8.0, phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        sig2 = ContinuousWaveSignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=np.log10(2e-8), phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        w1 = sig1.compute_waveform(toas)
        w2 = sig2.compute_waveform(toas)
        # At double frequency, amplitude envelope ~ h/(2*omega) is halved
        assert np.max(np.abs(w2)) < np.max(np.abs(w1))
        npt.assert_allclose(np.max(np.abs(w2)) / np.max(np.abs(w1)), 0.5, rtol=0.05)

    def test_zero_amplitude(self):
        """With h≈0, waveform should be negligible."""
        from jug.signals.continuous_wave import ContinuousWaveSignal
        sig = ContinuousWaveSignal(
            log10_h=-30.0, cos_gwtheta=0.5, gwphi=1.0,
            log10_fgw=-8.0, phase0=0.0, cos_inc=0.5, psi=0.3,
            raj=1.0, decj=-0.5,
        )
        toas = np.linspace(50000.0, 55000.0, 100)
        waveform = sig.compute_waveform(toas)
        assert np.max(np.abs(waveform)) < 1e-20

    def test_from_par(self):
        from jug.signals.continuous_wave import ContinuousWaveSignal
        params = {
            "CW_LOG10_H": -14.5,
            "CW_COS_GWTHETA": 0.3,
            "CW_GWPHI": 1.2,
            "CW_LOG10_FGW": -8.2,
            "CW_PHASE0": 0.5,
            "CW_COS_INC": 0.7,
            "CW_PSI": 0.8,
            "RAJ": 1.0,
            "DECJ": -0.5,
        }
        sig = ContinuousWaveSignal.from_par(params)
        assert sig.log10_h == -14.5
        assert sig.log10_fgw == -8.2
        assert sig.raj == 1.0


# ---------------------------------------------------------------------------
# BWM waveform
# ---------------------------------------------------------------------------

class TestBurstWithMemory:
    def test_output_shape(self):
        from jug.signals.burst_memory import BurstWithMemorySignal
        sig = BurstWithMemorySignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            t0_mjd=52500.0, pol=0.3, raj=1.0, decj=-0.5,
        )
        toas = np.linspace(50000.0, 55000.0, 1000)
        waveform = sig.compute_waveform(toas)
        assert waveform.shape == (1000,)

    def test_step_function(self):
        """Before burst epoch, waveform should be zero. After, it grows linearly."""
        from jug.signals.burst_memory import BurstWithMemorySignal
        t0 = 52500.0
        sig = BurstWithMemorySignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            t0_mjd=t0, pol=0.3, raj=1.0, decj=-0.5,
        )
        toas_before = np.linspace(50000.0, t0 - 1.0, 100)
        toas_after = np.linspace(t0 + 1.0, 55000.0, 100)
        toas = np.concatenate([toas_before, toas_after])
        waveform = sig.compute_waveform(toas)

        # Before burst: all zero
        npt.assert_allclose(waveform[:100], 0.0, atol=1e-30)
        # After burst: nonzero and increasing magnitude
        after_wf = np.abs(waveform[100:])
        assert np.all(after_wf > 0)
        # Linear growth: later values should be larger
        assert after_wf[-1] > after_wf[0]

    def test_amplitude_scaling(self):
        from jug.signals.burst_memory import BurstWithMemorySignal
        toas = np.linspace(50000.0, 55000.0, 500)
        sig1 = BurstWithMemorySignal(
            log10_h=-14.0, cos_gwtheta=0.5, gwphi=1.0,
            t0_mjd=52000.0, pol=0.3, raj=1.0, decj=-0.5,
        )
        sig2 = BurstWithMemorySignal(
            log10_h=-13.0, cos_gwtheta=0.5, gwphi=1.0,
            t0_mjd=52000.0, pol=0.3, raj=1.0, decj=-0.5,
        )
        w1 = sig1.compute_waveform(toas)
        w2 = sig2.compute_waveform(toas)
        npt.assert_allclose(w2, 10.0 * w1, rtol=1e-5)

    def test_from_par(self):
        from jug.signals.burst_memory import BurstWithMemorySignal
        params = {
            "BWM_LOG10_H": -15.0,
            "BWM_COS_GWTHETA": 0.2,
            "BWM_GWPHI": 2.5,
            "BWM_T0": 54000.0,
            "BWM_POL": 0.1,
            "RAJ": 0.7,
            "DECJ": 0.3,
        }
        sig = BurstWithMemorySignal.from_par(params)
        assert sig.log10_h == -15.0
        assert sig.t0_mjd == 54000.0


# ---------------------------------------------------------------------------
# Chromatic event waveform
# ---------------------------------------------------------------------------

class TestChromaticEvent:
    def test_output_shape(self):
        from jug.signals.chromatic_event import ChromaticEventSignal
        sig = ChromaticEventSignal(
            epoch_mjd=52500.0, amp_sec=1e-6, tau_day=30.0,
            chrom_idx=2.0, sign=-1.0,
        )
        toas = np.linspace(50000.0, 55000.0, 1000)
        freqs = np.full(1000, 1400.0)
        waveform = sig.compute_waveform(toas, freqs)
        assert waveform.shape == (1000,)

    def test_requires_frequencies(self):
        from jug.signals.chromatic_event import ChromaticEventSignal
        sig = ChromaticEventSignal()
        toas = np.linspace(50000.0, 55000.0, 100)
        with pytest.raises(ValueError, match="frequencies"):
            sig.compute_waveform(toas)

    def test_exponential_decay(self):
        """After epoch, waveform should decay exponentially."""
        from jug.signals.chromatic_event import ChromaticEventSignal
        epoch = 52500.0
        tau = 30.0
        sig = ChromaticEventSignal(
            epoch_mjd=epoch, amp_sec=1e-6, tau_day=tau,
            chrom_idx=2.0, sign=-1.0,
        )
        # TOAs after the event at reference frequency
        toas = np.array([epoch + 1.0, epoch + tau, epoch + 2 * tau, epoch + 3 * tau])
        freqs = np.full_like(toas, 1400.0)
        waveform = sig.compute_waveform(toas, freqs)

        # All should be positive (amp > 0)
        assert np.all(waveform > 0)
        # Should decrease monotonically
        assert np.all(np.diff(waveform) < 0)
        # At t = epoch + tau, should be ~ amp * exp(-1) ≈ 0.368 * amp
        npt.assert_allclose(waveform[1] / waveform[0], np.exp(-(tau - 1) / tau), rtol=0.01)

    def test_before_event_zero(self):
        """Before the event epoch, waveform should be zero (sign=-1)."""
        from jug.signals.chromatic_event import ChromaticEventSignal
        epoch = 52500.0
        sig = ChromaticEventSignal(
            epoch_mjd=epoch, amp_sec=1e-6, tau_day=30.0,
            chrom_idx=2.0, sign=-1.0,
        )
        toas = np.linspace(50000.0, epoch - 1.0, 100)
        freqs = np.full_like(toas, 1400.0)
        waveform = sig.compute_waveform(toas, freqs)
        npt.assert_allclose(waveform, 0.0, atol=1e-20)

    def test_chromatic_scaling(self):
        """Lower frequency → larger waveform (for positive chrom_idx)."""
        from jug.signals.chromatic_event import ChromaticEventSignal
        epoch = 52500.0
        sig = ChromaticEventSignal(
            epoch_mjd=epoch, amp_sec=1e-6, tau_day=30.0,
            chrom_idx=2.0, sign=-1.0,
        )
        # Use two TOAs so the reference MJD is before the test point
        toas = np.array([epoch, epoch + 10.0])
        # At 700 MHz vs 1400 MHz
        w_low = sig.compute_waveform(toas, np.array([700.0, 700.0]))
        w_high = sig.compute_waveform(toas, np.array([1400.0, 1400.0]))
        # For idx=2: (700/1400)^-2 = 4.0 scaling (check second TOA)
        npt.assert_allclose(w_low[1] / w_high[1], 4.0, rtol=1e-5)

    def test_from_par(self):
        from jug.signals.chromatic_event import ChromaticEventSignal
        params = {
            "CHROMEV_EPOCH": 53000.0,
            "CHROMEV_AMP": 2e-6,
            "CHROMEV_TAU": 50.0,
            "CHROMEV_IDX": 4.0,
            "CHROMEV_SIGN": 1.0,
        }
        sig = ChromaticEventSignal.from_par(params)
        assert sig.epoch_mjd == 53000.0
        assert sig.chrom_idx == 4.0
        assert sig.sign == 1.0


# ---------------------------------------------------------------------------
# Par-file round-trip
# ---------------------------------------------------------------------------

class TestParFileRoundTrip:
    def test_cw_par_roundtrip(self):
        """Write a par file with CW params, read it, detect signal, compute waveform."""
        from jug.io.par_reader import parse_par_file
        from jug.signals import detect_signals

        par_content = """PSR J0437-4715
RAJ 1.1456
DECJ -0.8237
F0 173.6879458
F1 -1.72e-15
PEPOCH 55000.0
DM 2.64
EPHEM DE440
CLK TT(BIPM2021)
UNITS TDB
CW_log10_h -14.5
CW_cos_gwtheta 0.3
CW_gwphi 1.2
CW_log10_fgw -8.2
CW_phase0 0.5
CW_cos_inc 0.7
CW_psi 0.8
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
            f.write(par_content)
            tmpfile = f.name

        try:
            params = parse_par_file(tmpfile)
            signals = detect_signals(params)
            assert len(signals) >= 1
            cw = [s for s in signals if s.signal_name == "CW"][0]
            toas = np.linspace(50000.0, 55000.0, 100)
            waveform = cw.compute_waveform(toas)
            assert waveform.shape == (100,)
            assert np.max(np.abs(waveform)) > 0
        finally:
            os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Noise registry integration
# ---------------------------------------------------------------------------

class TestNoiseRegistryIntegration:
    def test_signal_specs_in_registry(self):
        from jug.engine.noise_mode import NOISE_REGISTRY
        assert "CW" in NOISE_REGISTRY
        assert "BWM" in NOISE_REGISTRY
        assert "ChromaticEvent" in NOISE_REGISTRY

    def test_signal_detection(self):
        from jug.engine.noise_mode import NOISE_REGISTRY
        params = {"CW_LOG10_H": -14.0, "CW_LOG10_FGW": -8.0}
        cw_spec = NOISE_REGISTRY["CW"]
        assert cw_spec.detector(params) is True

    def test_signal_no_detect_missing_key(self):
        from jug.engine.noise_mode import NOISE_REGISTRY
        params = {"CW_LOG10_H": -14.0}  # missing CW_LOG10_FGW
        cw_spec = NOISE_REGISTRY["CW"]
        assert cw_spec.detector(params) is False

    def test_signal_labels(self):
        from jug.engine.noise_mode import NOISE_REGISTRY
        assert NOISE_REGISTRY["CW"].label == "CW Signal"
        assert NOISE_REGISTRY["BWM"].label == "Burst w/ Memory"
        assert NOISE_REGISTRY["ChromaticEvent"].label == "Chromatic Event"
