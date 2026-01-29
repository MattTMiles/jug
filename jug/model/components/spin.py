"""
Spin Component
==============

Thin wrapper around derivatives_spin.py that provides:
- Parameter filtering (only F0, F1, F2, ... parameters)
- Consistent interface for design matrix assembly

This component handles spin frequency parameters:
- F0 (spin frequency)
- F1 (spin-down rate)
- F2, F3, ... (higher order spin derivatives)

The actual derivative computation is delegated to the existing
compute_spin_derivatives() function - this wrapper just filters
and routes parameters.
"""

from typing import Dict, List
import numpy as np

from jug.fitting.derivatives_spin import compute_spin_derivatives as _compute_spin_derivatives
from jug.model.parameter_spec import is_spin_param


class SpinComponent:
    """
    Component for spin frequency parameters.

    Wraps derivatives_spin.py with parameter filtering.
    """

    # Parameters this component provides (can be extended dynamically)
    _PROVIDED_PARAMS = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']

    def provides_params(self) -> List[str]:
        """
        List parameter names this component provides derivatives for.

        Returns
        -------
        list of str
            Spin parameter names (F0, F1, F2, ...)
        """
        return self._PROVIDED_PARAMS.copy()

    def filter_fit_params(self, fit_params: List[str]) -> List[str]:
        """
        Filter fit_params to only spin parameters.

        Uses the ParameterSpec registry for robust classification
        rather than string prefix matching.

        Parameters
        ----------
        fit_params : list of str
            Full list of parameters to fit

        Returns
        -------
        list of str
            Only spin parameters
        """
        return [p for p in fit_params if is_spin_param(p)]

    def compute_derivatives(
        self,
        params: Dict,
        toas_mjd: np.ndarray,
        fit_params: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute spin parameter derivatives.

        Filters fit_params to spin parameters only, then delegates
        to the underlying compute_spin_derivatives() function.

        Parameters
        ----------
        params : dict
            Full timing model parameters (must include PEPOCH, F0, etc.)
        toas_mjd : np.ndarray
            TOA times in MJD
        fit_params : list of str
            Parameters to fit (may include non-spin params)
        **kwargs
            Ignored (for interface compatibility)

        Returns
        -------
        dict
            Mapping from spin parameter name to derivative column.
            Each value is np.ndarray of shape (n_toas,)

        Notes
        -----
        This method:
        1. Filters fit_params to only spin parameters
        2. Calls the existing compute_spin_derivatives() exactly as before
        3. Returns results unchanged

        The result is BIT-FOR-BIT identical to calling
        compute_spin_derivatives() directly with filtered params.
        """
        # Filter to spin parameters only
        spin_params = self.filter_fit_params(fit_params)

        if not spin_params:
            return {}

        # Delegate to existing implementation - NO changes to computation
        return _compute_spin_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=spin_params,
        )
