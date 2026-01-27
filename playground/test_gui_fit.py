#!/usr/bin/env python3
"""
Quick test of GUI fit worker functionality.
"""
import os
# Force JAX to use CPU (must be before any JAX imports)
os.environ['JAX_PLATFORMS'] = 'cpu'

from pathlib import Path
from jug.gui.workers.fit_worker import FitWorker
from PySide6.QtCore import QThreadPool, QCoreApplication
import sys


def test_fit_worker():
    """Test the fit worker with J1909-3744 data."""

    # Setup
    app = QCoreApplication(sys.argv)

    par_file = Path("data/pulsars/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")

    print(f"Testing fit worker with:")
    print(f"  PAR: {par_file}")
    print(f"  TIM: {tim_file}")
    print(f"  Parameters: F0, F1")
    print()

    # Create worker
    worker = FitWorker(par_file, tim_file, ['F0', 'F1'])

    # Connect signals
    def on_result(result):
        print("✓ Fit completed successfully!")
        print(f"  F0 = {result['final_params']['F0']:.15f} Hz")
        print(f"  F1 = {result['final_params']['F1']:.6e} Hz/s")
        print(f"  Final RMS = {result['final_rms']:.6f} μs")
        print(f"  Iterations = {result['iterations']}")
        print(f"  Converged = {result['converged']}")
        print(f"  Time = {result['total_time']:.2f} s")
        app.quit()

    def on_error(error_msg):
        print(f"✗ Fit failed:")
        print(error_msg)
        app.quit()

    worker.signals.result.connect(on_result)
    worker.signals.error.connect(on_error)

    # Run worker
    thread_pool = QThreadPool()
    thread_pool.start(worker)

    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    test_fit_worker()
