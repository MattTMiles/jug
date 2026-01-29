"""
Dispersion Component
====================

Thin wrapper around derivatives_dm.py that provides:
- Parameter filtering (only DM, DM1, DM2, ... parameters)
- Consistent interface for design matrix assembly

This component handles dispersion measure parameters:
- DM (base dispersion measure)
- DM1 (linear DM derivative)
- DM2, DM3, ... (higher order DM derivatives)

The actual derivative computation is delegated to the existing
compute_dm_derivatives() function - this wrapper just filters
and routes parameters.
"""

from typing import Dict, List
import numpy as np

from jug.fitting.derivatives_dm import compute_dm_derivatives as _compute_dm_derivatives
from jug.model.parameter_spec import is_dm_param


class DispersionComponent:
    """
    Component for dispersion measure parameters.

    Wraps derivatives_dm.py with parameter filtering.
    """

    # Parameters this component provides (can be extended dynamically)
    _PROVIDED_PARAMS = ['DM', 'DM1', 'DM2', 'DM3', 'DM4', 'DM5']

    def provides_params(self) -> List[str]:
        """
        List parameter names this component provides derivatives for.

        Returns
        -------
        list of str
            DM parameter names (DM, DM1, DM2, ...)
        """
        return self._PROVIDED_PARAMS.copy()

    def filter_fit_params(self, fit_params: List[str]) -> List[str]:
        """
        Filter fit_params to only DM parameters.

        Uses the ParameterSpec registry for robust classification
        rather than string prefix matching.

        Parameters
        ----------
        fit_params : list of str
            Full list of parameters to fit

        Returns
        -------
        list of str
            Only DM parameters
        """
        return [p for p in fit_params if is_dm_param(p)]

    def compute_derivatives(
        self,
        params: Dict,
        toas_mjd: np.ndarray,
        fit_params: List[str],
        freq_mhz: np.ndarray = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute DM parameter derivatives.

        Filters fit_params to DM parameters only, then delegates
        to the underlying compute_dm_derivatives() function.

        Parameters
        ----------
        params : dict
            Full timing model parameters (may include DMEPOCH, DM, etc.)
        toas_mjd : np.ndarray
            TOA times in MJD
        fit_params : list of str
            Parameters to fit (may include non-DM params)
        freq_mhz : np.ndarray
            Observing frequencies in MHz (REQUIRED for DM derivatives)
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        dict
            Mapping from DM parameter name to derivative column.
            Each value is np.ndarray of shape (n_toas,)

        Raises
        ------
        ValueError
            If freq_mhz is not provided but DM parameters are requested

        Notes
        -----
        This method:
        1. Filters fit_params to only DM parameters
        2. Validates that freq_mhz is provided
        3. Calls the existing compute_dm_derivatives() exactly as before
        4. Returns results unchanged

        The result is BIT-FOR-BIT identical to calling
        compute_dm_derivatives() directly with filtered params.
        """
        # Filter to DM parameters only
        dm_params = self.filter_fit_params(fit_params)

        if not dm_params:
            return {}

        # Validate frequency data
        if freq_mhz is None:
            raise ValueError(
                "freq_mhz is required for DM derivative computation. "
                f"Requested DM params: {dm_params}"
            )

        # Delegate to existing implementation - NO changes to computation
        return _compute_dm_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            freq_mhz=freq_mhz,
            fit_params=dm_params,
        )
