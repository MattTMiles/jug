"""Guardrail tests to prevent performance and architectural regressions.

These tests ensure that:
1. Dead code stays removed
2. Performance-critical paths use the optimized implementations
3. Numerical correctness constraints are preserved
"""

import importlib.util
import inspect
import pytest


class TestDeadCodeRemainsRemoved:
    """Verify that removed modules stay removed."""

    def test_no_binary_dispatch_import(self):
        """binary_dispatch.py was removed - verify it stays gone."""
        spec = importlib.util.find_spec("jug.delays.binary_dispatch")
        assert spec is None, (
            "jug.delays.binary_dispatch should not exist. "
            "It was replaced by combined.py dispatch + binary_registry.py"
        )


class TestECORROptimizations:
    """Verify ECORR uses vectorized gather/scatter, not Python for-loops."""

    def test_ecorr_no_python_loops_in_whiten_residuals(self):
        """whiten_residuals() should use vectorized indexing, not for-loops."""
        from jug.noise.ecorr import ECORRWhitener
        source = inspect.getsource(ECORRWhitener.whiten_residuals)

        # Should NOT contain "for k, group in enumerate"
        assert "for k, group in enumerate" not in source, (
            "whiten_residuals() still has Python for-loop over epoch groups. "
            "Should use vectorized indexing with _flat_gather_idx instead."
        )

        # Should contain the flat index arrays
        assert "_flat_gather_idx" in source, (
            "whiten_residuals() should use _flat_gather_idx for vectorized gather"
        )

    def test_ecorr_no_python_loops_in_whiten_matrix(self):
        """whiten_matrix() should use vectorized indexing, not for-loops."""
        from jug.noise.ecorr import ECORRWhitener
        source = inspect.getsource(ECORRWhitener.whiten_matrix)

        # Should NOT contain "for k, group in enumerate"
        assert "for k, group in enumerate" not in source, (
            "whiten_matrix() still has Python for-loop over epoch groups. "
            "Should use vectorized indexing with _flat_gather_idx instead."
        )

        # Should contain the flat index arrays
        assert "_flat_gather_idx" in source, (
            "whiten_matrix() should use _flat_gather_idx for vectorized gather"
        )


class TestLongdoublePhasePreserved:
    """Verify phase computation still uses longdouble precision."""

    def test_longdouble_phase_preserved(self):
        """compute_phase_residuals() must use np.longdouble, not JAX."""
        from jug.residuals.simple_calculator import compute_phase_residuals
        source = inspect.getsource(compute_phase_residuals)

        # Must contain longdouble
        assert "longdouble" in source, (
            "compute_phase_residuals() MUST use np.longdouble for dt_sec. "
            "JAX doesn't support 80-bit extended precision. Removing longdouble "
            "introduces a 44ns precision gap that breaks Tempo2 parity."
        )

        # Should NOT use jnp (JAX numpy)
        assert "jnp." not in source, (
            "compute_phase_residuals() should NOT use JAX (jnp.*). "
            "JAX doesn't support longdouble precision."
        )


class TestDesignMatrixPreallocation:
    """Verify design matrix uses pre-allocation, not repeated column_stack."""

    def test_design_matrix_uses_preallocation(self):
        """Design matrix assembly should pre-allocate once, not use multiple column_stack calls."""
        from jug.fitting.optimized_fitter import _run_general_fit_iterations
        source = inspect.getsource(_run_general_fit_iterations)

        # Extract the design matrix assembly section (around line 1924-1970)
        # Look for the "Assemble design matrix" comment to "Solve WLS" comment
        start = source.find("# Count columns first")
        end = source.find("# Solve WLS")
        if start == -1 or end == -1:
            pytest.skip("Could not find design matrix assembly section")

        assembly_section = source[start:end]

        # Should contain pre-allocation pattern
        assert "np.empty" in assembly_section, (
            "Design matrix should use np.empty() for pre-allocation"
        )

        # Count column_stack occurrences - should be 0 in the assembly section
        # (note: column_stack might appear elsewhere in comments, so we're lenient)
        column_stack_count = assembly_section.count("np.column_stack")
        assert column_stack_count == 0, (
            f"Design matrix assembly should not use np.column_stack (found {column_stack_count} times). "
            "Should pre-allocate with np.empty() and fill via slicing."
        )


class TestWLSSolverUsesJAX:
    """Verify augmented WLS solver uses JAX linalg, not numpy linalg."""

    def test_augmented_solver_uses_jax(self):
        """Augmented solver paths should use jnp.linalg, not np.linalg."""
        from jug.fitting.optimized_fitter import _run_general_fit_iterations
        source = inspect.getsource(_run_general_fit_iterations)

        # Extract the "fast" and "exact" augmented solver sections
        # Look for "if n_augmented > 0:" blocks
        fast_start = source.find('if solver_mode == "fast":')
        fast_end = source.find('else:', fast_start + 100)  # Find the first else after fast
        exact_start = source.find('if n_augmented > 0:', fast_end)
        exact_end = source.find('else:', exact_start + 100)

        if fast_start == -1 or exact_start == -1:
            pytest.skip("Could not find augmented solver sections")

        fast_section = source[fast_start:fast_end]
        exact_section = source[exact_start:exact_end]

        # Check both sections for JAX usage in augmented paths
        for section_name, section in [("fast", fast_section), ("exact", exact_section)]:
            # Should contain jnp.linalg.solve or jnp.linalg.lstsq
            assert "jnp.linalg" in section, (
                f"{section_name} augmented solver should use jnp.linalg (JAX), "
                f"not np.linalg (numpy) for GPU-readiness and consistency"
            )
