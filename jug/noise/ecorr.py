"""ECORR (epoch-correlated noise) support.

ECORR models correlated jitter noise within observing epochs.  TOAs that
share the same backend flag value **and** fall within the same MJD window
(default 0.5 days ≈ a single observing session) are grouped into an
*epoch*.  Within each epoch, the noise covariance gains a rank-1 update:

.. math::

    C_{ij} += \\text{ECORR}^2  \\qquad \\forall \\, i, j \\text{ in same epoch}

In matrix notation the full covariance is:

.. math::

    C = \\text{diag}(\\sigma_i^2)  +  U \\, \\text{diag}(J_k) \\, U^T

where :math:`U` is an (N_toa × N_epoch) quantization matrix with
:math:`U_{i,k} = 1` if TOA *i* belongs to epoch *k*, and
:math:`J_k = \\text{ECORR}_k^2`.

The key operation exposed by this module is **block-Cholesky whitening**:
given residuals *r* and design matrix *M*, produce whitened versions
:math:`\\tilde{r} = L^{-1} r` and :math:`\\tilde{M} = L^{-1} M` such
that the standard OLS/WLS solver recovers the GLS solution.  This is done
block-by-block without ever forming the full N×N matrix.

For single-TOA epochs the whitening reduces to the usual diagonal
:math:`1/\\sigma_i` scaling, so the approach degrades gracefully when
ECORR entries are absent or when no TOA groups exist.

References
----------
- van Haasteren & Vallisneri (2014) — "New advances in the Gaussian-process
  approach to pulsar-timing data analysis"
- NANOGrav / enterprise noise conventions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from jug.noise.white import WhiteNoiseEntry, build_backend_mask


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochGroup:
    """A group of TOA indices sharing the same backend and observing epoch.

    Attributes
    ----------
    indices : tuple of int
        TOA indices belonging to this epoch (sorted).
    ecorr_us : float
        ECORR value for this backend in microseconds.
    flag_value : str
        Backend flag value (for diagnostics / debugging).
    """
    indices: Tuple[int, ...]
    ecorr_us: float
    flag_value: str


@dataclass
class ECORRWhitener:
    """Pre-computed block-Cholesky whitening data for ECORR.

    This object stores the Cholesky factors for each epoch block and
    can apply :math:`L^{-1}` to vectors and matrices efficiently.

    Attributes
    ----------
    epoch_groups : list of EpochGroup
        All multi-TOA epoch groups (size ≥ 2).
    singleton_indices : np.ndarray
        Indices of TOAs not in any multi-TOA epoch.
    n_toas : int
        Total number of TOAs.
    """
    epoch_groups: List[EpochGroup]
    singleton_indices: np.ndarray
    n_toas: int
    # Pre-computed lower-triangular Cholesky factors per block
    _L_blocks: List[np.ndarray] = field(default_factory=list, repr=False)
    # Pre-computed inverse diagonal entries for singletons
    _sigma_inv_singletons: Optional[np.ndarray] = field(default=None, repr=False)

    def prepare(self, sigma_sec: np.ndarray) -> None:
        """Pre-compute Cholesky factors from diagonal variances + ECORR.

        Parameters
        ----------
        sigma_sec : np.ndarray, shape (n_toas,)
            Per-TOA uncertainties in seconds (after EFAC/EQUAD scaling).
        """
        self._L_blocks = []
        for group in self.epoch_groups:
            idx = np.array(group.indices)
            n = len(idx)
            # Block covariance: diag(σ²) + J * 11^T
            sig2 = sigma_sec[idx] ** 2
            ecorr_sec = group.ecorr_us * 1e-6
            J = ecorr_sec ** 2
            C_block = np.diag(sig2) + J * np.ones((n, n))
            L_lower, _ = cho_factor(C_block, lower=True)
            self._L_blocks.append(L_lower)

        # Singletons: just 1/σ
        if len(self.singleton_indices) > 0:
            self._sigma_inv_singletons = 1.0 / sigma_sec[self.singleton_indices]
        else:
            self._sigma_inv_singletons = np.array([], dtype=np.float64)

    def whiten_residuals(self, r_sec: np.ndarray) -> np.ndarray:
        """Apply L^{-1} to a residual vector.

        Parameters
        ----------
        r_sec : np.ndarray, shape (n_toas,)

        Returns
        -------
        r_white : np.ndarray, shape (n_toas,)
            Whitened residuals (dimensionless if r was in seconds).
        """
        r_white = np.empty(self.n_toas, dtype=np.float64)

        # Multi-TOA epoch blocks: solve L x = r_block for x = L^{-1} r_block
        for group, L in zip(self.epoch_groups, self._L_blocks):
            idx = np.array(group.indices)
            r_white[idx] = solve_triangular(L, r_sec[idx], lower=True)

        # Singleton TOAs: simple diagonal scaling
        if len(self.singleton_indices) > 0:
            r_white[self.singleton_indices] = (
                r_sec[self.singleton_indices] * self._sigma_inv_singletons
            )

        return r_white

    def whiten_matrix(self, M: np.ndarray) -> np.ndarray:
        """Apply L^{-1} to a design matrix (column-wise).

        Parameters
        ----------
        M : np.ndarray, shape (n_toas, n_params)

        Returns
        -------
        M_white : np.ndarray, shape (n_toas, n_params)
        """
        M_white = np.empty_like(M)

        for group, L in zip(self.epoch_groups, self._L_blocks):
            idx = np.array(group.indices)
            M_white[idx, :] = solve_triangular(L, M[idx, :], lower=True)

        if len(self.singleton_indices) > 0:
            M_white[self.singleton_indices, :] = (
                M[self.singleton_indices, :] * self._sigma_inv_singletons[:, None]
            )

        return M_white

    def chi2(self, r_sec: np.ndarray) -> float:
        """Compute r^T C^{-1} r using the block structure.

        Parameters
        ----------
        r_sec : np.ndarray, shape (n_toas,)
            Residuals in seconds.

        Returns
        -------
        chi2 : float
        """
        r_white = self.whiten_residuals(r_sec)
        return float(np.dot(r_white, r_white))


# ---------------------------------------------------------------------------
# Epoch grouping
# ---------------------------------------------------------------------------

def _group_toas_into_epochs(
    toas_mjd: np.ndarray,
    mask: np.ndarray,
    dt_days: float = 0.5,
) -> List[Tuple[int, ...]]:
    """Group masked TOAs into epochs by time proximity.

    TOAs are sorted by MJD. A new epoch starts whenever the gap between
    consecutive TOAs exceeds *dt_days*.

    Parameters
    ----------
    toas_mjd : np.ndarray, shape (n_toas,)
        MJD of all TOAs.
    mask : np.ndarray of bool, shape (n_toas,)
        Boolean mask selecting TOAs for this backend.
    dt_days : float, optional
        Maximum gap within an epoch (default 0.5 days).

    Returns
    -------
    epochs : list of tuple of int
        Each tuple contains the *original* TOA indices belonging to one epoch.
        Only epochs with ≥ 2 TOAs are returned.
    """
    indices = np.where(mask)[0]
    if len(indices) < 2:
        return []

    # Sort selected indices by MJD
    mjd_selected = toas_mjd[indices]
    order = np.argsort(mjd_selected)
    sorted_idx = indices[order]
    sorted_mjd = mjd_selected[order]

    # Walk through and split on gaps
    epochs: List[List[int]] = [[int(sorted_idx[0])]]
    for i in range(1, len(sorted_idx)):
        if sorted_mjd[i] - sorted_mjd[i - 1] > dt_days:
            epochs.append([])
        epochs[-1].append(int(sorted_idx[i]))

    # Return only multi-TOA epochs
    return [tuple(ep) for ep in epochs if len(ep) >= 2]


def build_ecorr_whitener(
    toas_mjd: np.ndarray,
    toa_flags: List[Dict[str, str]],
    noise_entries: List[WhiteNoiseEntry],
    dt_days: float = 0.5,
) -> Optional[ECORRWhitener]:
    """Build an ECORRWhitener from parsed noise entries and TOA metadata.

    If no ECORR entries are present, returns ``None``.

    Parameters
    ----------
    toas_mjd : np.ndarray, shape (n_toas,)
        MJD of all TOAs.
    toa_flags : list of dict
        Per-TOA flag dictionaries.
    noise_entries : list of WhiteNoiseEntry
        All parsed noise entries (EFAC/EQUAD/ECORR).
    dt_days : float, optional
        Maximum gap within an epoch (default 0.5 days).

    Returns
    -------
    whitener : ECORRWhitener or None
        Ready-to-use whitener, or ``None`` if no ECORR entries found.
    """
    ecorr_entries = [e for e in noise_entries if e.kind == 'ECORR']
    if not ecorr_entries:
        return None

    n_toas = len(toas_mjd)
    all_grouped_indices = set()
    epoch_groups: List[EpochGroup] = []

    for entry in ecorr_entries:
        mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days)
        for ep in epochs:
            epoch_groups.append(EpochGroup(
                indices=ep,
                ecorr_us=entry.value,
                flag_value=entry.flag_value,
            ))
            all_grouped_indices.update(ep)

    # Singletons: TOAs not in any multi-TOA epoch
    singleton_indices = np.array(
        sorted(set(range(n_toas)) - all_grouped_indices),
        dtype=np.intp,
    )

    return ECORRWhitener(
        epoch_groups=epoch_groups,
        singleton_indices=singleton_indices,
        n_toas=n_toas,
    )
