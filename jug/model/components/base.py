"""
Base Protocol for Timing Components
====================================

Defines the interface that all timing components must implement.

Components are thin wrappers around derivative functions that:
1. Filter parameters to only those relevant to this component
2. Provide consistent interface for design matrix assembly
3. Document what parameters they provide

This is a Protocol (structural subtyping) rather than ABC,
allowing duck-typing while providing type hints.
"""

from typing import Dict, List, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class TimingComponent(Protocol):
    """
    Protocol for timing model components.

    Components wrap derivative computation modules and provide:
    - Parameter filtering (only pass relevant params)
    - Consistent interface for design matrix assembly
    - Documentation of provided parameters

    All methods should be pure (no side effects) and deterministic.
    """

    def provides_params(self) -> List[str]:
        """
        List parameter names this component provides derivatives for.

        Returns
        -------
        list of str
            Parameter names (e.g., ['F0', 'F1', 'F2', 'F3'] for spin)
        """
        ...

    def compute_derivatives(
        self,
        params: Dict,
        toas_mjd: np.ndarray,
        fit_params: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute derivatives for requested parameters.

        Parameters
        ----------
        params : dict
            Full timing model parameters
        toas_mjd : np.ndarray
            TOA times in MJD
        fit_params : list of str
            Parameters to fit (may include params from other components)
        **kwargs
            Additional arguments (e.g., freq_mhz for DM)

        Returns
        -------
        dict
            Mapping from parameter name to derivative column.
            Only includes parameters that this component provides.
            Each value is np.ndarray of shape (n_toas,)

        Notes
        -----
        - Must filter fit_params to only those this component provides
        - Must return bit-for-bit identical results to direct function calls
        - Must not modify input arrays
        """
        ...

    def filter_fit_params(self, fit_params: List[str]) -> List[str]:
        """
        Filter fit_params to only those this component provides.

        Parameters
        ----------
        fit_params : list of str
            Full list of parameters to fit

        Returns
        -------
        list of str
            Subset that this component provides
        """
        ...
