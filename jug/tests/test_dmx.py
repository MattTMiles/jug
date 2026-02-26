"""Tests for jug.model.dmx -- DMX support."""

import numpy as np
import pytest

from jug.model.dmx import (
    DMXRange,
    parse_dmx_ranges,
    assign_toas_to_dmx,
    build_dmx_design_matrix,
    get_dmx_delays,
)

from jug.utils.constants import K_DM_SEC


# ---------------------------------------------------------------------------
# DMXRange
# ---------------------------------------------------------------------------

class TestDMXRange:
    def test_creation(self):
        r = DMXRange(index=1, r1_mjd=58000.0, r2_mjd=58100.0, value=0.5, label="DMX_0001")
        assert r.r1_mjd == 58000.0
        assert r.r2_mjd == 58100.0
        assert r.value == 0.5

    def test_defaults(self):
        r = DMXRange(index=1, r1_mjd=58000.0, r2_mjd=58100.0)
        assert r.value == 0.0
        assert r.freq_lo_mhz == 0.0
        assert r.freq_hi_mhz == np.inf


# ---------------------------------------------------------------------------
# parse_dmx_ranges
# ---------------------------------------------------------------------------

class TestParseDMXRanges:
    def test_basic_parsing(self):
        params = {
            "DMXR1_0001": "58000.0",
            "DMXR2_0001": "58100.0",
            "DMX_0001": "0.5",
            "DMXR1_0002": "58100.0",
            "DMXR2_0002": "58200.0",
            "DMX_0002": "0.3",
        }
        ranges = parse_dmx_ranges(params)
        assert len(ranges) == 2
        assert ranges[0].index == 1
        assert ranges[0].r1_mjd == 58000.0
        assert ranges[0].value == 0.5
        assert ranges[1].index == 2

    def test_missing_r2_skipped(self):
        """Incomplete range (missing R2) should be skipped."""
        params = {"DMXR1_0001": "58000.0"}
        ranges = parse_dmx_ranges(params)
        assert len(ranges) == 0

    def test_unpadded_keys(self):
        """Support unpadded keys like DMXR1_1."""
        params = {
            "DMXR1_1": "58000.0",
            "DMXR2_1": "58100.0",
        }
        ranges = parse_dmx_ranges(params)
        assert len(ranges) == 1

    def test_frequency_bounds(self):
        params = {
            "DMXR1_0001": "58000.0",
            "DMXR2_0001": "58100.0",
            "DMXF1_0001": "400.0",
            "DMXF2_0001": "2000.0",
        }
        ranges = parse_dmx_ranges(params)
        assert ranges[0].freq_lo_mhz == 400.0
        assert ranges[0].freq_hi_mhz == 2000.0

    def test_empty_params(self):
        assert parse_dmx_ranges({}) == []


# ---------------------------------------------------------------------------
# assign_toas_to_dmx
# ---------------------------------------------------------------------------

class TestAssignToasToDMX:
    def test_simple_assignment(self):
        ranges = [
            DMXRange(1, 58000.0, 58100.0),
            DMXRange(2, 58100.0, 58200.0),
        ]
        toas = np.array([58050, 58150, 58250.0])
        freq = np.full(3, 1284.0)

        assignment, mask = assign_toas_to_dmx(toas, freq, ranges)
        assert assignment[0] == 0   # in range 1
        assert assignment[1] == 1   # in range 2
        assert assignment[2] == -1  # not in any range

    def test_frequency_filtering(self):
        """DMX matching is now MJD-only (matching PINT/Tempo2).
        DMXF1/DMXF2 are informational; they don't filter TOA assignment."""
        ranges = [
            DMXRange(1, 58000.0, 58200.0, freq_lo_mhz=1000.0, freq_hi_mhz=2000.0),
        ]
        toas = np.array([58050, 58050.0])
        freq = np.array([1284.0, 400.0])

        assignment, _ = assign_toas_to_dmx(toas, freq, ranges)
        assert assignment[0] == 0   # 1284 MHz in time range
        assert assignment[1] == 0   # 400 MHz also in time range (freq not filtered)

    def test_overlap_first_wins(self):
        ranges = [
            DMXRange(1, 58000.0, 58150.0),
            DMXRange(2, 58100.0, 58200.0),
        ]
        toas = np.array([58125.0])  # in overlap region
        freq = np.full(1, 1284.0)

        assignment, mask = assign_toas_to_dmx(toas, freq, ranges)
        assert assignment[0] == 0  # first range wins

    def test_boundary_inclusive(self):
        ranges = [DMXRange(1, 58000.0, 58100.0)]
        toas = np.array([58000.0, 58100.0])  # exact boundaries
        freq = np.full(2, 1284.0)

        assignment, _ = assign_toas_to_dmx(toas, freq, ranges)
        assert assignment[0] == 0
        assert assignment[1] == 0


# ---------------------------------------------------------------------------
# build_dmx_design_matrix
# ---------------------------------------------------------------------------

class TestBuildDMXDesignMatrix:
    def test_shape(self):
        ranges = [
            DMXRange(1, 58000.0, 58100.0, label="DMX_0001"),
            DMXRange(2, 58100.0, 58200.0, label="DMX_0002"),
        ]
        toas = np.array([58050, 58150, 58250.0])
        freq = np.full(3, 1284.0)

        M, labels = build_dmx_design_matrix(toas, freq, ranges)
        assert M.shape == (3, 2)
        assert labels == ["DMX_0001", "DMX_0002"]

    def test_values(self):
        ranges = [DMXRange(1, 58000.0, 58200.0)]
        toas = np.array([58050, 58250.0])
        freq = np.full(2, 1284.0)

        M, _ = build_dmx_design_matrix(toas, freq, ranges)

        expected_deriv = K_DM_SEC / (1284.0 ** 2)
        np.testing.assert_allclose(M[0, 0], expected_deriv, rtol=1e-10)
        assert M[1, 0] == 0.0  # outside range

    def test_zero_outside_range(self):
        ranges = [DMXRange(1, 58000.0, 58100.0)]
        toas = np.array([58200.0])
        freq = np.full(1, 1284.0)

        M, _ = build_dmx_design_matrix(toas, freq, ranges)
        assert M[0, 0] == 0.0


# ---------------------------------------------------------------------------
# get_dmx_delays
# ---------------------------------------------------------------------------

class TestGetDMXDelays:
    def test_basic_delay(self):
        ranges = [DMXRange(1, 58000.0, 58200.0, value=1.0)]
        toas = np.array([58050.0])
        freq = np.array([1284.0])

        delay = get_dmx_delays(toas, freq, ranges)
        expected = 1.0 * K_DM_SEC / (1284.0 ** 2)
        np.testing.assert_allclose(delay[0], expected, rtol=1e-10)

    def test_zero_delay_outside(self):
        ranges = [DMXRange(1, 58000.0, 58100.0, value=1.0)]
        toas = np.array([58200.0])
        freq = np.array([1284.0])

        delay = get_dmx_delays(toas, freq, ranges)
        assert delay[0] == 0.0

    def test_zero_dmx_value(self):
        ranges = [DMXRange(1, 58000.0, 58200.0, value=0.0)]
        toas = np.array([58050.0])
        freq = np.array([1284.0])

        delay = get_dmx_delays(toas, freq, ranges)
        assert delay[0] == 0.0
