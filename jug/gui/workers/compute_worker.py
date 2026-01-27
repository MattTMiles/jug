"""
Compute Worker - Background Residual Computation
=================================================

Computes residuals using an existing session in background thread.
"""
from PySide6.QtCore import QRunnable, QObject, Signal, Slot
import numpy as np


class ComputeWorkerSignals(QObject):
    """Signals for compute worker."""
    
    # Emitted when computation completes
    result = Signal(dict)
    
    # Emitted if error occurs
    error = Signal(str)
    
    # Emitted when worker finishes
    finished = Signal()
    
    # Progress updates
    progress = Signal(str)


class ComputeWorker(QRunnable):
    """
    Worker for computing residuals in background.
    
    Uses existing TimingSession for fast computation.
    """
    
    def __init__(self, session, params=None):
        """
        Initialize compute worker.
        
        Parameters
        ----------
        session : TimingSession
            Existing timing session
        params : dict, optional
            Parameter overrides
        """
        super().__init__()
        self.signals = ComputeWorkerSignals()
        self.session = session
        self.params = params
    
    @Slot()
    def run(self):
        """Compute residuals in background thread."""
        try:
            self.signals.progress.emit("Computing residuals...")
            
            # Compute residuals (uses cached session data)
            result = self.session.compute_residuals(params=self.params)
            
            # Make thread-safe copy
            result_safe = {
                'residuals_us': np.array(result['residuals_us']),
                'rms_us': float(result['rms_us']),
                'tdb_mjd': np.array(result['tdb_mjd']),
                'errors_us': np.array(result.get('errors_us', [])) if result.get('errors_us') is not None else None,
                'dt_sec': np.array(result.get('dt_sec', [])),
            }
            
            self.signals.progress.emit(f"RMS: {result_safe['rms_us']:.6f} μs")
            
            # Emit result
            self.signals.result.emit(result_safe)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        
        finally:
            self.signals.finished.emit()
