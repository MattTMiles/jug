"""
Background worker for JAX warmup to avoid first-fit lag.

This worker runs JIT compilation in the background after data loads,
so the first fit button click doesn't freeze the UI.
"""
from PySide6.QtCore import QRunnable, QObject, Signal, Slot
import numpy as np


class WarmupSignals(QObject):
    """Signals for warmup worker."""
    finished = Signal()
    progress = Signal(str)


class WarmupWorker(QRunnable):
    """
    Background worker for JAX JIT warmup.

    Runs a dummy fit iteration to trigger JIT compilation,
    so the first real fit is fast.
    """

    def __init__(self, n_toas: int):
        """
        Initialize warmup worker.

        Parameters
        ----------
        n_toas : int
            Number of TOAs (used to create dummy arrays with matching shape)
        """
        super().__init__()
        self.signals = WarmupSignals()
        self.n_toas = n_toas

    @Slot()
    def run(self):
        """Run JAX warmup in background thread."""
        try:
            self.signals.progress.emit("Warming up JAX...")

            # Import JAX functions that need warming up
            from jug.utils.jax_setup import ensure_jax_x64
            ensure_jax_x64()
            import jax
            import jax.numpy as jnp

            # Use SMALL fixed size for warmup (JAX JIT works with any size after)
            # This keeps warmup fast (~1-2s) regardless of actual data size
            n = min(100, self.n_toas)  # Small size for fast JIT
            dt_sec = jnp.array(np.random.randn(n) * 86400.0 * 1000)  # ~1000 days span
            errors = jnp.array(np.abs(np.random.randn(n)) * 1e-6 + 1e-6)
            weights = 1.0 / errors**2

            # Import and warm up the JIT-compiled fitting functions
            from jug.fitting.optimized_fitter import (
                full_iteration_jax_f0_f1,
                full_iteration_jax_general,
                compute_spin_phase_jax,
                compute_spin_derivatives_jax,
                wls_solve_jax
            )

            # Warm up F0+F1 iteration (most common case)
            f0 = 339.0  # Typical pulsar frequency
            f1 = -1e-15  # Typical spindown
            try:
                _, _, _ = full_iteration_jax_f0_f1(dt_sec, f0, f1, errors, weights)
            except Exception:
                pass  # Ignore errors during warmup

            # Warm up general iteration for 2-param case
            f_values = jnp.array([f0, f1])
            try:
                _, _, _ = full_iteration_jax_general(dt_sec, f_values, errors, weights)
            except Exception:
                pass

            # Warm up spin phase computation
            try:
                _ = compute_spin_phase_jax(dt_sec, f_values)
            except Exception:
                pass

            # Warm up spin derivatives
            try:
                _ = compute_spin_derivatives_jax(dt_sec, f_values, f0)
            except Exception:
                pass

            # Warm up WLS solver
            M = jnp.column_stack([
                -dt_sec / f0,
                -(dt_sec**2 / 2.0) / f0
            ])
            r = jnp.zeros(n)
            try:
                _, _ = wls_solve_jax(r, errors, M)
            except Exception:
                pass

            self.signals.progress.emit("JAX warmup complete")

        except Exception as e:
            # Warmup errors are not critical - just log and continue
            self.signals.progress.emit(f"JAX warmup skipped: {str(e)[:50]}")

        finally:
            self.signals.finished.emit()
