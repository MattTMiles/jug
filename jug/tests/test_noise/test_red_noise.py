"""Tests for jug.noise.red_noise -- red noise and DM noise processes."""

import numpy as np
import pytest

from jug.noise.red_noise import (
    build_fourier_design_matrix,
    powerlaw_spectrum,
    turnover_spectrum,
    RedNoiseProcess,
    DMNoiseProcess,
    parse_red_noise_params,
    parse_dm_noise_params,
)


# ---------------------------------------------------------------------------
# build_fourier_design_matrix
# ---------------------------------------------------------------------------

class TestFourierDesignMatrix:
    def test_basic_shape(self):
        toas = np.linspace(58000, 59000, 50)
        n_harm = 10
        F, freqs = build_fourier_design_matrix(toas, n_harm)
        assert F.shape == (50, 20)  # 2 * n_harm columns
        assert freqs.shape == (10,)

    def test_frequencies_are_harmonics_of_Tspan(self):
        toas = np.linspace(58000, 59000, 50)
        Tspan_days = 1000.0
        F, freqs = build_fourier_design_matrix(toas, 5, Tspan_days)
        Tspan_sec = Tspan_days * 86400.0
        expected = np.arange(1, 6) / Tspan_sec
        np.testing.assert_allclose(freqs, expected, rtol=1e-12)

    def test_sin_cos_interleaved(self):
        """Columns should be [sin(f1), cos(f1), sin(f2), cos(f2), ...]"""
        toas = np.array([58000, 58100, 58200, 58300, 58400.0])
        F, freqs = build_fourier_design_matrix(toas, 2)

        t_sec = (toas - toas.min()) * 86400.0
        phase1 = 2 * np.pi * freqs[0] * t_sec

        # Column 0 should be sin(2pi f_1 t)
        np.testing.assert_allclose(F[:, 0], np.sin(phase1), atol=1e-12)
        # Column 1 should be cos(2pi f_1 t)
        np.testing.assert_allclose(F[:, 1], np.cos(phase1), atol=1e-12)

    def test_negative_Tspan_defaults(self):
        """Negative Tspan defaults to 365.25 days."""
        toas = np.array([58000.0, 58000.0])
        F, freqs = build_fourier_design_matrix(toas, 5, Tspan_days=-1.0)
        assert F.shape == (2, 10)  # should use default

    def test_single_toa(self):
        F, freqs = build_fourier_design_matrix(np.array([58000.0]), 3)
        assert F.shape == (1, 6)
        # Default Tspan is 365.25 days for single-TOA

    def test_auto_Tspan(self):
        toas = np.linspace(57000, 59000, 100)
        F1, f1 = build_fourier_design_matrix(toas, 5)
        F2, f2 = build_fourier_design_matrix(toas, 5, Tspan_days=2000.0)
        np.testing.assert_allclose(f1, f2, rtol=1e-10)


# ---------------------------------------------------------------------------
# powerlaw_spectrum
# ---------------------------------------------------------------------------

class TestPowerlawSpectrum:
    def test_shape(self):
        freqs = np.logspace(-9, -7, 20)
        P = powerlaw_spectrum(freqs, log10_A=-14.0, gamma=13.0/3.0)
        assert P.shape == (20,)

    def test_positive(self):
        freqs = np.logspace(-9, -7, 20)
        P = powerlaw_spectrum(freqs, -14.0, 4.33)
        assert np.all(P > 0)

    def test_red_spectrum(self):
        """Lower frequencies should have more power."""
        freqs = np.logspace(-9, -7, 20)
        P = powerlaw_spectrum(freqs, -14.0, 4.33)
        assert P[0] > P[-1]

    def test_amplitude_scaling(self):
        """Doubling log10_A by 1 -> 100* amplitude -> 10000* PSD."""
        freqs = np.array([1e-8])
        P1 = powerlaw_spectrum(freqs, -14.0, 4.33)[0]
        P2 = powerlaw_spectrum(freqs, -13.0, 4.33)[0]
        np.testing.assert_allclose(P2 / P1, 100.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# turnover_spectrum
# ---------------------------------------------------------------------------

class TestTurnoverSpectrum:
    def test_matches_powerlaw_at_high_freq(self):
        """Well above f_bend, turnover spectrum ~= power law."""
        freqs = np.logspace(-5, -4, 20)  # well above f_bend=1e-8
        P_pl = powerlaw_spectrum(freqs, -14.0, 4.33)
        P_to = turnover_spectrum(freqs, -14.0, 4.33, f_bend_hz=1e-8)
        np.testing.assert_allclose(P_to, P_pl, rtol=1e-3)

    def test_suppression_at_low_freq(self):
        """Below f_bend, turnover spectrum should be suppressed vs pure PL."""
        freqs = np.array([1e-10])
        P_pl = powerlaw_spectrum(freqs, -14.0, 4.33)[0]
        P_to = turnover_spectrum(freqs, -14.0, 4.33, f_bend_hz=1e-8)[0]
        assert P_to < P_pl


# ---------------------------------------------------------------------------
# RedNoiseProcess
# ---------------------------------------------------------------------------

class TestRedNoiseProcess:
    def test_spectrum(self):
        rn = RedNoiseProcess(log10_A=-14.0, gamma=13.0/3.0)
        freqs = np.logspace(-9, -7, 10)
        P = rn.spectrum(freqs)
        assert P.shape == (10,)

    def test_build_basis_and_prior(self):
        rn = RedNoiseProcess(log10_A=-14.0, gamma=13.0/3.0, n_harmonics=5)
        toas = np.linspace(58000, 59000, 50)
        F, phi = rn.build_basis_and_prior(toas)
        assert F.shape == (50, 10)
        assert phi.shape == (10,)
        # phi should be positive
        assert np.all(phi > 0)
        # Each harmonic's sin/cos pair should have the same prior
        np.testing.assert_allclose(phi[0], phi[1])


# ---------------------------------------------------------------------------
# DMNoiseProcess
# ---------------------------------------------------------------------------

class TestDMNoiseProcess:
    def test_chromatic_scaling(self):
        """Higher observing freq -> smaller chromatic weight -> smaller basis."""
        dm = DMNoiseProcess(log10_A=-14.0, gamma=13.0/3.0, n_harmonics=3)
        toas = np.linspace(58000, 59000, 10)
        freq_low = np.full(10, 400.0)   # 400 MHz
        freq_high = np.full(10, 1400.0)  # 1400 MHz

        F_low, _ = dm.build_basis_and_prior(toas, freq_low)
        F_high, _ = dm.build_basis_and_prior(toas, freq_high)

        # At 1400 MHz, chromatic weight is 1.0; at 400 MHz it's (1400/400)^2 = 12.25
        assert np.abs(F_low).max() > np.abs(F_high).max()

    def test_shape(self):
        dm = DMNoiseProcess(log10_A=-14.0, gamma=2.0, n_harmonics=4)
        toas = np.linspace(58000, 59000, 20)
        freq = np.full(20, 1284.0)
        F, phi = dm.build_basis_and_prior(toas, freq)
        assert F.shape == (20, 8)
        assert phi.shape == (8,)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_red_noise_temponest(self):
        params = {"TNRedAmp": "-14.0", "TNRedGam": "4.33", "TNRedC": "30"}
        rn = parse_red_noise_params(params)
        assert rn is not None
        assert rn.log10_A == -14.0
        assert rn.gamma == 4.33
        assert rn.n_harmonics == 30

    def test_parse_red_noise_alternative(self):
        params = {"RN_log10_A": "-13.5", "RN_gamma": "3.0"}
        rn = parse_red_noise_params(params)
        assert rn is not None
        assert rn.log10_A == -13.5

    def test_parse_red_noise_absent(self):
        assert parse_red_noise_params({}) is None

    def test_parse_dm_noise_temponest(self):
        params = {"TNDMAmp": "-14.0", "TNDMGam": "2.0", "TNDMC": "15"}
        dm = parse_dm_noise_params(params)
        assert dm is not None
        assert dm.log10_A == -14.0
        assert dm.n_harmonics == 15

    def test_parse_dm_noise_absent(self):
        assert parse_dm_noise_params({}) is None
