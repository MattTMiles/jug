"""
Background worker for running parameter fits without blocking the GUI.
"""
from pathlib import Path
from PySide6.QtCore import QRunnable, QObject, Signal, Slot
import numpy as np


class WorkerSignals(QObject):
    """Signals for communicating between worker thread and main thread."""

    # Emitted when fit completes successfully
    result = Signal(dict)

    # Emitted if an error occurs
    error = Signal(str)

    # Emitted when worker finishes (success or error)
    finished = Signal()

    # Emitted with progress updates (iteration number, current RMS)
    progress = Signal(int, float)


class FitWorker(QRunnable):
    """
    Worker thread for running parameter fitting in the background.

    This prevents the GUI from freezing during long-running fits.
    """

    def __init__(self, par_file: Path, tim_file: Path, fit_params: list[str]):
        """
        Initialize the fit worker.

        Parameters
        ----------
        par_file : Path
            Path to .par file
        tim_file : Path
            Path to .tim file
        fit_params : list[str]
            List of parameters to fit (e.g., ['F0', 'F1', 'DM'])
        """
        super().__init__()
        self.signals = WorkerSignals()
        self.par_file = par_file
        self.tim_file = tim_file
        self.fit_params = fit_params
        self.is_running = True

    @Slot()
    def run(self):
        """
        Run the fit in the background thread.

        This method is called when the worker is started via QThreadPool.
        """
        try:
            from jug.fitting.optimized_fitter import fit_parameters_optimized

            # Run the fit (auto-detect device - will use GPU if available)
            result = fit_parameters_optimized(
                par_file=self.par_file,
                tim_file=self.tim_file,
                fit_params=self.fit_params,
                device=None,  # Auto-detect (GPU if available, else CPU)
                verbose=False  # Don't print to console in GUI mode
            )

            # Copy numpy arrays to ensure thread safety
            result_safe = {
                'final_params': dict(result['final_params']),
                'uncertainties': dict(result['uncertainties']),
                'covariance': np.array(result['covariance']),
                'final_rms': float(result['final_rms']),
                'iterations': int(result['iterations']),
                'converged': bool(result['converged']),
                'total_time': float(result['total_time']),
            }

            # Emit success signal
            self.signals.result.emit(result_safe)

        except Exception as e:
            # Emit error signal
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)

        finally:
            # Always emit finished signal
            self.signals.finished.emit()
