"""Tests for GUI parameter parity with the fitter.

Validates:
1. GUI fittable_params list covers all registry-supported parameters
2. Canonicalization is applied before fitting
3. _parse_par_file_parameters finds all supported params in a par file
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


class TestGUIParameterParity:
    """The GUI's fittable parameter set must be a superset of the registry."""

    def test_gui_uses_registry_list(self):
        """_parse_par_file_parameters must use list_fittable_params (not a hard-coded list)."""
        import inspect
        from jug.gui.main_window import MainWindow
        source = inspect.getsource(MainWindow._parse_par_file_parameters)
        assert 'list_fittable_params' in source, (
            "_parse_par_file_parameters should call list_fittable_params(), "
            "not use a hard-coded list"
        )

    def test_registry_covers_key_params(self):
        """Verify the registry contains all params the fitter has derivatives for."""
        from jug.model.parameter_spec import list_fittable_params
        fittable = set(list_fittable_params())

        # Spin
        for p in ['F0', 'F1', 'F2', 'F3']:
            assert p in fittable, f"{p} missing from fittable params"

        # DM
        for p in ['DM', 'DM1', 'DM2']:
            assert p in fittable, f"{p} missing from fittable params"

        # Astrometry
        for p in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
            assert p in fittable, f"{p} missing from fittable params"

        # ELL1 binary
        for p in ['PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'SINI', 'M2']:
            assert p in fittable, f"{p} missing from fittable params"

        # DD/DDK binary
        for p in ['ECC', 'OM', 'T0', 'PBDOT', 'OMDOT', 'GAMMA', 'XDOT', 'EDOT']:
            assert p in fittable, f"{p} missing from fittable params"

        # Orthometric Shapiro
        for p in ['H3', 'H4', 'STIG']:
            assert p in fittable, f"{p} missing from fittable params"

        # DDK Kopeikin
        for p in ['KIN', 'KOM']:
            assert p in fittable, f"{p} missing from fittable params"

        # FD
        for p in ['FD1', 'FD2', 'FD3']:
            assert p in fittable, f"{p} missing from fittable params"

        # Solar wind
        assert 'NE_SW' in fittable, "NE_SW missing from fittable params"

    def test_parse_finds_all_par_file_params(self):
        """_parse_par_file_parameters should find all supported params in mini par file."""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PySide6 not available")

        from jug.gui.main_window import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        par_path = Path(__file__).parent / "data_golden" / "J1909_mini.par"
        if not par_path.exists():
            pytest.skip("mini par file not found")

        window = MainWindow()
        window.par_file = str(par_path)
        found = window._parse_par_file_parameters()

        # The mini par file has these params; all should be found
        expected = {'F0', 'F1', 'DM', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX',
                    'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI'}
        missing = expected - set(found)
        assert not missing, f"Parameters in par file but not found by GUI: {missing}"

    def test_canonicalization_in_on_fit_clicked(self):
        """on_fit_clicked should canonicalize parameter names."""
        import inspect
        from jug.gui.main_window import MainWindow
        source = inspect.getsource(MainWindow.on_fit_clicked)
        assert 'canonicalize_param_name' in source, (
            "on_fit_clicked must canonicalize parameter names"
        )

    def test_validation_in_on_fit_clicked(self):
        """on_fit_clicked should validate parameter names."""
        import inspect
        from jug.gui.main_window import MainWindow
        source = inspect.getsource(MainWindow.on_fit_clicked)
        assert 'validate_fit_param' in source, (
            "on_fit_clicked must validate parameter names"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
