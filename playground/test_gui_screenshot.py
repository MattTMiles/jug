#!/usr/bin/env python3
"""
Test script to verify GUI functionality without display.
Simulates loading data programmatically.
"""
from pathlib import Path
from PySide6.QtWidgets import QApplication
from jug.gui.main_window import MainWindow
import sys

# Create application
app = QApplication(sys.argv)

# Create main window
window = MainWindow()

# Simulate loading files
window.par_file = Path('data/pulsars/J1909-3744_tdb.par')
window.tim_file = Path('data/pulsars/J1909-3744.tim')

# Compute residuals
print("Loading data and computing residuals...")
window._compute_initial_residuals()

# Check data loaded
if window.residuals_us is not None:
    print(f"✓ Successfully loaded {len(window.mjd)} TOAs")
    print(f"✓ RMS = {window.rms_us:.6f} μs")
    print(f"✓ Residuals range: {window.residuals_us.min():.3f} to {window.residuals_us.max():.3f} μs")
    print()
    print("GUI Phase 1 (MVP) is working correctly!")
else:
    print("✗ Failed to load data")
    sys.exit(1)

# Clean exit (don't show GUI)
sys.exit(0)
