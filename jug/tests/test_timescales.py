"""Tests for TCB <-> TDB timescale conversion utilities.

This module tests the timescale conversion functions in jug/utils/timescales.py,
verifying that TCB to TDB conversion follows the Irwin & Fukushima (1999) convention
and matches PINT/Tempo2 behavior.
"""

import pytest
import numpy as np
from jug.utils.timescales import (
    IFTE_MJD0, IFTE_K, IFTE_KM1,
    parse_timescale,
    convert_tcb_epoch_to_tdb,
    convert_tdb_epoch_to_tcb,
    scale_parameter_tcb_to_tdb,
    scale_parameter_tdb_to_tcb,
    convert_par_params_to_tdb,
    convert_par_params_to_tcb,
)


class TestConstants:
    """Test that the IFTE constants are correct."""
    
    def test_ifte_k_value(self):
        """Verify IFTE_K = 1 + 1.55051979176e-8."""
        assert abs(float(IFTE_K) - 1.000000015505198) < 1e-16
    
    def test_ifte_km1_value(self):
        """Verify IFTE_KM1 (L_B) value."""
        assert abs(float(IFTE_KM1) - 1.55051979176e-8) < 1e-20
    
    def test_ifte_mjd0_value(self):
        """Verify reference epoch MJD."""
        assert abs(float(IFTE_MJD0) - 43144.0003725) < 1e-10


class TestParseTimescale:
    """Test timescale parsing from parameter dicts."""
    
    def test_parse_tdb(self):
        params = {'_par_timescale': 'TDB'}
        assert parse_timescale(params) == 'TDB'
    
    def test_parse_tcb(self):
        params = {'_par_timescale': 'TCB'}
        assert parse_timescale(params) == 'TCB'
    
    def test_parse_default(self):
        params = {}
        assert parse_timescale(params) == 'TDB'


class TestEpochConversion:
    """Test MJD epoch conversion between TCB and TDB."""
    
    def test_tcb_to_tdb_at_reference_epoch(self):
        """At IFTE_MJD0, TCB = TDB (by definition)."""
        mjd_tcb = IFTE_MJD0
        mjd_tdb = convert_tcb_epoch_to_tdb(mjd_tcb)
        assert abs(mjd_tdb - IFTE_MJD0) < 1e-12
    
    def test_tdb_to_tcb_at_reference_epoch(self):
        """At IFTE_MJD0, TDB = TCB (by definition)."""
        mjd_tdb = IFTE_MJD0
        mjd_tcb = convert_tdb_epoch_to_tcb(mjd_tdb)
        assert abs(mjd_tcb - IFTE_MJD0) < 1e-12
    
    def test_tcb_to_tdb_after_reference(self):
        """TCB runs faster than TDB, so TCB > TDB after reference epoch."""
        mjd_tcb = IFTE_MJD0 + 10000.0  # ~27 years later
        mjd_tdb = convert_tcb_epoch_to_tdb(mjd_tcb)
        # TDB should be slightly less than TCB (slower clock)
        assert mjd_tdb < mjd_tcb
        # Difference should be ~0.155 days per 10000 days
        assert abs((mjd_tcb - mjd_tdb) - 10000.0 * float(IFTE_KM1)) < 1e-6
    
    def test_tdb_to_tcb_after_reference(self):
        """Inverse conversion."""
        mjd_tdb = IFTE_MJD0 + 10000.0
        mjd_tcb = convert_tdb_epoch_to_tcb(mjd_tdb)
        assert mjd_tcb > mjd_tdb
    
    def test_round_trip_tcb_tdb_tcb(self):
        """Converting TCB -> TDB -> TCB should recover original."""
        mjd_tcb_orig = np.longdouble("55000.123456789")
        mjd_tdb = convert_tcb_epoch_to_tdb(mjd_tcb_orig)
        mjd_tcb_final = convert_tdb_epoch_to_tcb(mjd_tdb)
        assert abs(mjd_tcb_final - mjd_tcb_orig) < 1e-12
    
    def test_round_trip_tdb_tcb_tdb(self):
        """Converting TDB -> TCB -> TDB should recover original."""
        mjd_tdb_orig = np.longdouble("55000.987654321")
        mjd_tcb = convert_tdb_epoch_to_tcb(mjd_tdb_orig)
        mjd_tdb_final = convert_tcb_epoch_to_tdb(mjd_tcb)
        assert abs(mjd_tdb_final - mjd_tdb_orig) < 1e-12
    
    def test_known_value_pepoch(self):
        """Test with a known PEPOCH value from PINT documentation."""
        # From PINT docs: PEPOCH 55000 (TCB) ~= 54999.99914475 (TDB)
        mjd_tcb = np.longdouble("55000.0")
        mjd_tdb = convert_tcb_epoch_to_tdb(mjd_tcb)
        expected_tdb = (55000.0 - float(IFTE_MJD0)) / float(IFTE_K) + float(IFTE_MJD0)
        assert abs(mjd_tdb - expected_tdb) < 1e-10


class TestParameterScaling:
    """Test parameter scaling based on effective dimensionality."""
    
    def test_scale_dimensionless(self):
        """Parameters with eff_dim=0 should not scale."""
        val_tcb = 0.5
        val_tdb = scale_parameter_tcb_to_tdb(val_tcb, 0)
        assert val_tdb == val_tcb
    
    def test_scale_f0(self):
        """F0 (frequency) has eff_dim=-1, scales by IFTE_K."""
        f0_tcb = 100.0  # Hz
        f0_tdb = scale_parameter_tcb_to_tdb(f0_tcb, -1)
        # F0_tdb = F0_tcb * IFTE_K^1
        expected = f0_tcb * float(IFTE_K)
        assert abs(f0_tdb - expected) < 1e-12
    
    def test_scale_f1(self):
        """F1 has eff_dim=-2, scales by IFTE_K^2."""
        f1_tcb = -1e-15  # Hz/s
        f1_tdb = scale_parameter_tcb_to_tdb(f1_tcb, -2)
        expected = f1_tcb * float(IFTE_K)**2
        assert abs(f1_tdb - expected) / abs(expected) < 1e-12
    
    def test_scale_a1(self):
        """A1 (semi-major axis) has eff_dim=1, scales by IFTE_K^(-1)."""
        a1_tcb = 2.0  # lt-s
        a1_tdb = scale_parameter_tcb_to_tdb(a1_tcb, 1)
        expected = a1_tcb / float(IFTE_K)
        assert abs(a1_tdb - expected) < 1e-12
    
    def test_scale_pb(self):
        """PB (orbital period) has eff_dim=1, scales by IFTE_K^(-1)."""
        pb_tcb = 1.0  # days
        pb_tdb = scale_parameter_tcb_to_tdb(pb_tcb, 1)
        expected = pb_tcb / float(IFTE_K)
        assert abs(pb_tdb - expected) < 1e-12
    
    def test_scale_inverse_tcb_to_tdb_to_tcb(self):
        """Scaling TCB -> TDB -> TCB should recover original."""
        val_tcb = 123.456
        eff_dim = -2
        val_tdb = scale_parameter_tcb_to_tdb(val_tcb, eff_dim)
        val_tcb_final = scale_parameter_tdb_to_tcb(val_tdb, eff_dim)
        assert abs(val_tcb_final - val_tcb) / abs(val_tcb) < 1e-14


class TestConvertParParamsToTDB:
    """Test the main conversion function for parameter dictionaries."""
    
    def test_already_tdb(self):
        """If already TDB, no conversion should happen."""
        params = {
            '_par_timescale': 'TDB',
            'PEPOCH': 55000.0,
            'F0': 100.0,
        }
        params_orig = params.copy()
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        assert result is params  # Same object
        assert params['PEPOCH'] == params_orig['PEPOCH']
        assert params['F0'] == params_orig['F0']
        assert "already in TDB" in log[0]
    
    def test_convert_epochs(self):
        """Test that epoch parameters are converted correctly."""
        params = {
            '_par_timescale': 'TCB',
            'PEPOCH': np.longdouble("55000.0"),
            'T0': np.longdouble("55100.0"),
            'TASC': np.longdouble("55200.0"),
        }
        
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # Check that epochs were converted
        assert params['PEPOCH'] < 55000.0  # TDB runs slower
        assert params['T0'] < 55100.0
        assert params['TASC'] < 55200.0
        
        # Check manually
        expected_pepoch = convert_tcb_epoch_to_tdb(np.longdouble("55000.0"))
        assert abs(params['PEPOCH'] - float(expected_pepoch)) < 1e-10
    
    def test_convert_frequency_params(self):
        """Test that F0, F1, F2 are scaled correctly."""
        params = {
            '_par_timescale': 'TCB',
            'F0': 100.0,
            'F1': -1e-15,
            'F2': 1e-25,
        }
        
        f0_orig, f1_orig, f2_orig = params['F0'], params['F1'], params['F2']
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # F0_tdb = F0_tcb * IFTE_K
        assert abs(params['F0'] - f0_orig * float(IFTE_K)) < 1e-12
        # F1_tdb = F1_tcb * IFTE_K^2
        assert abs(params['F1'] - f1_orig * float(IFTE_K)**2) < 1e-25
        # F2_tdb = F2_tcb * IFTE_K^3
        assert abs(params['F2'] - f2_orig * float(IFTE_K)**3) / abs(f2_orig) < 1e-12
    
    def test_convert_dm_params(self):
        """Test that DM, DM1, DM2 are scaled correctly."""
        params = {
            '_par_timescale': 'TCB',
            'DM': 50.0,
            'DM1': 0.001,
        }
        
        dm_orig, dm1_orig = params['DM'], params['DM1']
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # DM has eff_dim=-1
        assert abs(params['DM'] - dm_orig * float(IFTE_K)) < 1e-12
        # DM1 has eff_dim=-2
        assert abs(params['DM1'] - dm1_orig * float(IFTE_K)**2) < 1e-15
    
    def test_convert_binary_params(self):
        """Test that A1, PB, FB0 are scaled correctly."""
        params = {
            '_par_timescale': 'TCB',
            'A1': 2.0,
            'PB': 1.5,
            'FB0': 0.666,
            'ECC': 0.1,  # Dimensionless, should not change
            'OM': 45.0,  # Angle, should not change
        }
        
        a1_orig, pb_orig, fb0_orig = params['A1'], params['PB'], params['FB0']
        ecc_orig, om_orig = params['ECC'], params['OM']
        
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # A1 and PB have eff_dim=1, scale by IFTE_K^(-1)
        assert abs(params['A1'] - a1_orig / float(IFTE_K)) < 1e-12
        assert abs(params['PB'] - pb_orig / float(IFTE_K)) < 1e-12
        # FB0 has eff_dim=-1, scales by IFTE_K
        assert abs(params['FB0'] - fb0_orig * float(IFTE_K)) < 1e-12
        # Dimensionless params unchanged
        assert params['ECC'] == ecc_orig
        assert params['OM'] == om_orig
    
    def test_convert_dmx_values(self):
        """Test that DMX_* parameters are scaled correctly."""
        params = {
            '_par_timescale': 'TCB',
            'DMX_0001': 0.05,
            'DMX_0002': -0.03,
            'DMXR1_0001': np.longdouble("55000.0"),
            'DMXR2_0001': np.longdouble("55100.0"),
        }
        
        dmx1_orig = params['DMX_0001']
        dmxr1_orig = params['DMXR1_0001']
        
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # DMX values have eff_dim=-1 (same as DM)
        assert abs(params['DMX_0001'] - dmx1_orig * float(IFTE_K)) < 1e-15
        # DMXR1/DMXR2 are epochs, should be converted
        expected_dmxr1 = convert_tcb_epoch_to_tdb(dmxr1_orig)
        assert abs(params['DMXR1_0001'] - float(expected_dmxr1)) < 1e-10
    
    def test_no_conversion_for_noise_params(self):
        """Test that EFAC, EQUAD, ECORR are not converted."""
        params = {
            '_par_timescale': 'TCB',
            'EFAC': 1.5,
            'EQUAD': 2.0,
            'ECORR': 3.0,
            'DMEFAC': 1.2,
        }
        
        orig_vals = params.copy()
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # Noise params should be unchanged
        assert params['EFAC'] == orig_vals['EFAC']
        assert params['EQUAD'] == orig_vals['EQUAD']
        assert params['ECORR'] == orig_vals['ECORR']
        assert params['DMEFAC'] == orig_vals['DMEFAC']
    
    def test_no_conversion_for_tzr(self):
        """Test that TZRMJD and TZRFRQ are not converted (per PINT)."""
        params = {
            '_par_timescale': 'TCB',
            'TZRMJD': 55000.0,
            'TZRFRQ': 1400.0,
        }
        
        orig_vals = params.copy()
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        # TZR params should be unchanged
        assert params['TZRMJD'] == orig_vals['TZRMJD']
        assert params['TZRFRQ'] == orig_vals['TZRFRQ']
    
    def test_metadata_set(self):
        """Test that conversion metadata is set correctly."""
        params = {
            '_par_timescale': 'TCB',
            'PEPOCH': 55000.0,
        }
        
        result, log = convert_par_params_to_tdb(params, verbose=False)
        
        assert params['_timescale_in'] == 'TCB'
        assert params['_tcb_converted'] is True
        assert params['UNITS'] == 'TDB'
        assert params['_par_timescale'] == 'TDB'
    
    def test_round_trip_conversion(self):
        """Test that TCB -> TDB -> TCB recovers original values."""
        params_tcb_orig = {
            '_par_timescale': 'TCB',
            'PEPOCH': 55000.0,
            'F0': 100.0,
            'F1': -1e-15,
            'DM': 50.0,
            'A1': 2.0,
            'PB': 1.5,
            'ECC': 0.1,
        }
        
        # Save original values
        pepoch_orig = params_tcb_orig['PEPOCH']
        f0_orig = params_tcb_orig['F0']
        a1_orig = params_tcb_orig['A1']
        
        # Convert to TDB
        params_tdb = params_tcb_orig.copy()
        convert_par_params_to_tdb(params_tdb, verbose=False)
        
        # Convert back to TCB
        params_tcb_final = params_tdb.copy()
        convert_par_params_to_tcb(params_tcb_final, verbose=False)
        
        # Check recovery (within numerical precision)
        assert abs(params_tcb_final['PEPOCH'] - pepoch_orig) < 1e-10
        assert abs(params_tcb_final['F0'] - f0_orig) / f0_orig < 1e-14
        assert abs(params_tcb_final['A1'] - a1_orig) / a1_orig < 1e-14


class TestIntegration:
    """Integration tests with validate_par_timescale."""
    
    def test_validate_converts_tcb(self):
        """Test that validate_par_timescale converts TCB to TDB."""
        from jug.io.par_reader import validate_par_timescale
        
        params = {
            '_par_timescale': 'TCB',
            'PEPOCH': 55000.0,
            'F0': 100.0,
        }
        
        f0_orig = params['F0']
        result = validate_par_timescale(params, context="Test", verbose=False)
        
        # Should return TDB
        assert result == 'TDB'
        # F0 should be scaled
        assert params['F0'] != f0_orig
        assert abs(params['F0'] - f0_orig * float(IFTE_K)) < 1e-12
        # Metadata should be set
        assert params['_tcb_converted'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
