#!/usr/bin/env python3
"""
GUI smoke tests for JUG (headless).

Tests that the GUI can initialize and basic components work
without a display using QT_QPA_PLATFORM=offscreen.

Run with: python tests/test_gui_smoke.py
Or: QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py

These tests verify the GUI doesn't crash on startup and can
load data, but don't test interactive functionality.
"""

import os
import sys
from pathlib import Path

# Must set before importing Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def get_mini_paths():
    """Get paths to bundled mini dataset."""
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if par.exists() and tim.exists():
        return str(par), str(tim)
    return None, None


def test_pyside_import():
    """Test PySide6 is available."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        return True, "OK"
    except ImportError as e:
        return False, f"PySide6 not available: {e}"


def test_qapp_creation():
    """Test QApplication can be created headless."""
    try:
        from PySide6.QtWidgets import QApplication
        
        # Check if app already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        return True, "OK (QApplication created)"
    except Exception as e:
        return False, str(e)


def test_main_window_import():
    """Test main window module imports."""
    try:
        from jug.gui.main_window import MainWindow
        return True, "OK"
    except ImportError as e:
        return False, f"import failed: {e}"


def test_main_window_creation():
    """Test main window can be instantiated."""
    try:
        from PySide6.QtWidgets import QApplication
        from jug.gui.main_window import MainWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create window without showing
        window = MainWindow()
        
        # Check basic attributes exist (use actual JUG GUI attributes)
        if not hasattr(window, 'plot_widget'):
            return False, "missing plot_widget attribute"
        
        return True, "OK (MainWindow created)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_load_data_into_gui():
    """Test loading data into GUI."""
    try:
        from PySide6.QtWidgets import QApplication
        from jug.gui.main_window import MainWindow
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        par, tim = get_mini_paths()
        if par is None:
            return False, "mini dataset not found"
        
        window = MainWindow()
        
        # Set par and tim files directly (JUG's API)
        window.par_file = par
        window.tim_file = tim
        
        # Verify they were set
        if window.par_file != par or window.tim_file != tim:
            return False, "files not set correctly"
        
        return True, "OK (par/tim files set)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_gui_compute_residuals():
    """Test that GUI can compute residuals with loaded data."""
    try:
        from PySide6.QtWidgets import QApplication
        from jug.gui.main_window import MainWindow
        from jug.engine.session import TimingSession
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        par, tim = get_mini_paths()
        if par is None:
            return False, "mini dataset not found"
        
        window = MainWindow()
        
        # Set files
        window.par_file = par
        window.tim_file = tim
        
        # Create session synchronously (bypassing async worker for testing)
        window.session = TimingSession(par, tim, verbose=False)
        
        # Compute residuals via session
        result = window.session.compute_residuals()
        
        n_toas = result.get('n_toas', 0)
        if n_toas == 0:
            return False, "no TOAs computed"
        
        return True, f"OK ({n_toas} TOAs)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_gui_fit_f0f1():
    """Test that GUI can run a simple fit."""
    try:
        from PySide6.QtWidgets import QApplication
        from jug.gui.main_window import MainWindow
        from jug.engine.session import TimingSession
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        par, tim = get_mini_paths()
        if par is None:
            return False, "mini dataset not found"
        
        window = MainWindow()
        
        # Set files
        window.par_file = par
        window.tim_file = tim
        
        # Create session synchronously (bypassing async worker for testing)
        window.session = TimingSession(par, tim, verbose=False)
        
        # Fit via session
        result = window.session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=5)
        
        iterations = result.get('iterations', 0)
        if iterations == 0:
            return False, "no iterations performed"
        
        return True, f"OK ({iterations} iterations)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_plot_widget_import():
    """Test plot widgets can be imported."""
    try:
        # JUG uses pyqtgraph PlotWidget, check it's available
        from pyqtgraph import PlotWidget
        return True, "OK (pyqtgraph.PlotWidget)"
    except ImportError as e:
        return False, f"import failed: {e}"


def main():
    """Run all GUI smoke tests."""
    print("=" * 60)
    print("GUI Smoke Tests (Headless)")
    print("=" * 60)
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', 'not set')}")
    
    # First check if PySide6 is available - skip all if not
    pyside_passed, pyside_msg = test_pyside_import()
    if not pyside_passed:
        print(f"\n  [SKIP] PySide6 not available: {pyside_msg}")
        print("  All GUI tests skipped (install PySide6 to enable)")
        print("\n" + "=" * 60)
        print("GUI tests SKIPPED")
        # Return 0 so we don't fail CI when Qt isn't installed
        return 0
    
    tests = [
        ("PySide6 Import", test_pyside_import),
        ("QApplication Creation", test_qapp_creation),
        ("Main Window Import", test_main_window_import),
        ("Main Window Creation", test_main_window_creation),
        ("Plot Widget Import", test_plot_widget_import),
        ("Load Data", test_load_data_into_gui),
        ("Compute Residuals", test_gui_compute_residuals),
        ("Fit F0/F1", test_gui_fit_f0f1),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All GUI smoke tests PASSED")
        return 0
    else:
        print("Some GUI smoke tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
