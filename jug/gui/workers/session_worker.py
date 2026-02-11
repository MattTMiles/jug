"""
Session Worker - Background Session Creation
=============================================

Loads files and creates timing session in background thread
to avoid freezing the GUI.
"""
from pathlib import Path
from PySide6.QtCore import QRunnable, QObject, Signal, Slot


class SessionWorkerSignals(QObject):
    """Signals for session worker."""
    
    # Emitted when session is ready
    result = Signal(object)  # TimingSession object
    
    # Emitted if error occurs
    error = Signal(str)
    
    # Emitted when worker finishes
    finished = Signal()
    
    # Progress updates
    progress = Signal(str)  # Status message


class SessionWorker(QRunnable):
    """
    Worker for creating timing session in background.
    
    Prevents UI freeze during file loading and parsing.
    """
    
    def __init__(self, par_file: Path, tim_file: Path, clock_dir=None):
        """
        Initialize session worker.
        
        Parameters
        ----------
        par_file : Path
            Path to .par file
        tim_file : Path
            Path to .tim file
        clock_dir : str, optional
            Clock directory path
        """
        super().__init__()
        self.signals = SessionWorkerSignals()
        self.par_file = par_file
        self.tim_file = tim_file
        self.clock_dir = clock_dir
    
    @Slot()
    def run(self):
        """Create timing session in background thread."""
        try:
            # Configure Astropy IERS before any time operations
            from jug.utils.astropy_config import configure_astropy
            configure_astropy()

            from jug.engine import open_session

            self.signals.progress.emit("Loading files...")
            
            # Create session (this parses files and caches data)
            session = open_session(
                par_file=self.par_file,
                tim_file=self.tim_file,
                clock_dir=self.clock_dir,
                verbose=False
            )
            
            self.signals.progress.emit(f"Loaded {session.get_toa_count()} TOAs")
            
            # Emit session object
            self.signals.result.emit(session)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
        
        finally:
            self.signals.finished.emit()
