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

# JAX for JIT-compiled block-Cholesky whitening
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from jug.noise.white import WhiteNoiseEntry, build_backend_mask


# ---------------------------------------------------------------------------
# JIT-compiled kernels for batched block-Cholesky whitening
# ---------------------------------------------------------------------------

@jax.jit
def _batched_cholesky(C_padded: jnp.ndarray) -> jnp.ndarray:
    """Cholesky-factorise a batch of padded block-covariance matrices.

    Parameters
    ----------
    C_padded : jnp.ndarray, shape (n_blocks, max_n, max_n)
        Padded block covariance matrices.  Padding entries on the diagonal
        must be 1.0 (identity) so the Cholesky is well-defined.

    Returns
    -------
    L_padded : jnp.ndarray, shape (n_blocks, max_n, max_n)
        Lower-triangular Cholesky factors.
    """
    return jax.vmap(jnp.linalg.cholesky)(C_padded)


@jax.jit
def _batched_solve_vec(
    L_padded: jnp.ndarray,
    r_padded: jnp.ndarray,
) -> jnp.ndarray:
    """Solve L x = r for a batch of lower-triangular systems (vectors).

    Parameters
    ----------
    L_padded : jnp.ndarray, shape (n_blocks, max_n, max_n)
    r_padded : jnp.ndarray, shape (n_blocks, max_n)

    Returns
    -------
    x_padded : jnp.ndarray, shape (n_blocks, max_n)
    """
    return jax.vmap(lambda L, r: jla.solve_triangular(L, r, lower=True))(
        L_padded, r_padded
    )


@jax.jit
def _batched_solve_mat(
    L_padded: jnp.ndarray,
    M_padded: jnp.ndarray,
) -> jnp.ndarray:
    """Solve L X = M for a batch of lower-triangular systems (matrices).

    Parameters
    ----------
    L_padded : jnp.ndarray, shape (n_blocks, max_n, max_n)
    M_padded : jnp.ndarray, shape (n_blocks, max_n, n_params)

    Returns
    -------
    X_padded : jnp.ndarray, shape (n_blocks, max_n, n_params)
    """
    return jax.vmap(lambda L, M: jla.solve_triangular(L, M, lower=True))(
        L_padded, M_padded
    )


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

    All heavy numerical work (Cholesky factorisation, triangular solves)
    is performed via JAX JIT-compiled, ``vmap``-batched kernels.  The
    variable-size epoch blocks are right-padded to a common ``max_n``
    dimension with identity entries so that a single ``vmap`` call
    replaces the former Python for-loop.

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
    # --- pre-computed JAX arrays (set by prepare) --------------------------
    # Batched Cholesky factors, shape (n_blocks, max_n, max_n)
    _L_padded: Optional[jnp.ndarray] = field(default=None, repr=False)
    # Per-block actual sizes, shape (n_blocks,)
    _block_sizes: Optional[np.ndarray] = field(default=None, repr=False)
    # Index array for each block, shape (n_blocks, max_n).
    # Padded entries are set to 0 (safe because they are masked out).
    _block_indices: Optional[np.ndarray] = field(default=None, repr=False)
    # Maximum block size
    _max_n: int = field(default=0, repr=False)
    # Pre-computed inverse diagonal entries for singletons (JAX array)
    _sigma_inv_singletons: Optional[jnp.ndarray] = field(default=None, repr=False)

    def prepare(self, sigma_sec: np.ndarray) -> None:
        """Pre-compute batched Cholesky factors from diagonal variances + ECORR.

        All epoch blocks are padded to the same ``max_n`` dimension and
        stacked into a single 3-D array.  The Cholesky factorisation is
        then computed in one ``vmap``-batched JAX call.

        Parameters
        ----------
        sigma_sec : np.ndarray, shape (n_toas,)
            Per-TOA uncertainties in seconds (after EFAC/EQUAD scaling).
        """
        # Cast to float64 — JAX does not support float128/longdouble
        sigma_sec = np.asarray(sigma_sec, dtype=np.float64)
        n_blocks = len(self.epoch_groups)

        if n_blocks == 0:
            self._L_padded = None
            self._block_sizes = np.array([], dtype=np.int32)
            self._block_indices = np.empty((0, 0), dtype=np.intp)
            self._max_n = 0
            if len(self.singleton_indices) > 0:
                self._sigma_inv_singletons = jnp.array(
                    1.0 / sigma_sec[self.singleton_indices]
                )
            else:
                self._sigma_inv_singletons = jnp.array([], dtype=jnp.float64)
            return

        # Determine max block size for padding
        sizes = np.array([len(g.indices) for g in self.epoch_groups], dtype=np.int32)
        max_n = int(sizes.max())
        self._block_sizes = sizes
        self._max_n = max_n

        # Build padded covariance blocks and index arrays
        C_padded = np.zeros((n_blocks, max_n, max_n), dtype=np.float64)
        idx_padded = np.zeros((n_blocks, max_n), dtype=np.intp)

        for k, group in enumerate(self.epoch_groups):
            idx = np.array(group.indices)
            n = len(idx)
            sig2 = sigma_sec[idx] ** 2
            ecorr_sec = group.ecorr_us * 1e-6
            J = ecorr_sec ** 2
            # Fill the real block
            C_padded[k, :n, :n] = np.diag(sig2) + J * np.ones((n, n))
            # Padding: identity on the diagonal (keeps Cholesky well-defined)
            for i in range(n, max_n):
                C_padded[k, i, i] = 1.0
            idx_padded[k, :n] = idx

        self._block_indices = idx_padded

        # Batched Cholesky via JAX vmap (JIT-compiled)
        self._L_padded = _batched_cholesky(jnp.array(C_padded))

        # Singletons: just 1/σ
        if len(self.singleton_indices) > 0:
            self._sigma_inv_singletons = jnp.array(
                1.0 / sigma_sec[self.singleton_indices]
            )
        else:
            self._sigma_inv_singletons = jnp.array([], dtype=jnp.float64)

    def whiten_residuals(self, r_sec: np.ndarray) -> np.ndarray:
        """Apply L^{-1} to a residual vector.

        Uses a single ``vmap``-batched triangular solve over all epoch
        blocks simultaneously, then scatters the results back.

        Parameters
        ----------
        r_sec : np.ndarray, shape (n_toas,)

        Returns
        -------
        r_white : np.ndarray, shape (n_toas,)
            Whitened residuals (dimensionless if r was in seconds).
        """
        # Cast to float64 — JAX does not support float128/longdouble
        r_sec = np.asarray(r_sec, dtype=np.float64)
        r_white = np.empty(self.n_toas, dtype=np.float64)

        if self._L_padded is not None and len(self.epoch_groups) > 0:
            # Gather residuals into padded block array
            r_padded = np.zeros(
                (len(self.epoch_groups), self._max_n), dtype=np.float64
            )
            for k, group in enumerate(self.epoch_groups):
                n = self._block_sizes[k]
                idx = self._block_indices[k, :n]
                r_padded[k, :n] = r_sec[idx]

            # Batched triangular solve (JAX JIT + vmap)
            x_padded = np.asarray(
                _batched_solve_vec(self._L_padded, jnp.array(r_padded))
            )

            # Scatter results back
            for k, group in enumerate(self.epoch_groups):
                n = self._block_sizes[k]
                idx = self._block_indices[k, :n]
                r_white[idx] = x_padded[k, :n]

        # Singleton TOAs: simple diagonal scaling
        if len(self.singleton_indices) > 0:
            r_white[self.singleton_indices] = np.asarray(
                jnp.array(r_sec[self.singleton_indices])
                * self._sigma_inv_singletons
            )

        return r_white

    def whiten_matrix(self, M: np.ndarray) -> np.ndarray:
        """Apply L^{-1} to a design matrix (column-wise).

        Uses a single ``vmap``-batched triangular solve over all epoch
        blocks simultaneously.

        Parameters
        ----------
        M : np.ndarray, shape (n_toas, n_params)

        Returns
        -------
        M_white : np.ndarray, shape (n_toas, n_params)
        """
        # Cast to float64 — JAX does not support float128/longdouble
        M = np.asarray(M, dtype=np.float64)
        M_white = np.empty_like(M)
        n_params = M.shape[1]

        if self._L_padded is not None and len(self.epoch_groups) > 0:
            # Gather design matrix rows into padded block array
            M_padded = np.zeros(
                (len(self.epoch_groups), self._max_n, n_params), dtype=np.float64
            )
            for k, group in enumerate(self.epoch_groups):
                n = self._block_sizes[k]
                idx = self._block_indices[k, :n]
                M_padded[k, :n, :] = M[idx, :]

            # Batched triangular solve (JAX JIT + vmap)
            X_padded = np.asarray(
                _batched_solve_mat(self._L_padded, jnp.array(M_padded))
            )

            # Scatter results back
            for k, group in enumerate(self.epoch_groups):
                n = self._block_sizes[k]
                idx = self._block_indices[k, :n]
                M_white[idx, :] = X_padded[k, :n, :]

        if len(self.singleton_indices) > 0:
            inv = np.asarray(self._sigma_inv_singletons)
            M_white[self.singleton_indices, :] = (
                M[self.singleton_indices, :] * inv[:, None]
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
