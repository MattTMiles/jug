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
    
    Updated to use TimingSession for better performance.
    """

    def __init__(self, session, fit_params: list[str]):
        """
        Initialize the fit worker.

        Parameters
        ----------
        session : TimingSession
            Existing timing session (with cached data)
        fit_params : list[str]
            List of parameters to fit (e.g., ['F0', 'F1', 'DM'])
        """
        super().__init__()
        self.signals = WorkerSignals()
        self.session = session
        self.fit_params = fit_params
        self.is_running = True

    @Slot()
    def run(self):
        """
        Run the fit in the background thread.

        This method is called when the worker is started via QThreadPool.
        """
        try:
            # Use session.fit_parameters (reuses cached data)
            result = self.session.fit_parameters(
                fit_params=self.fit_params,
                verbose=False
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
