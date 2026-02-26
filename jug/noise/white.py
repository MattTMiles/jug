"""White noise models: EFAC, EQUAD, ECORR.

This module implements Tempo2-style white noise scaling for TOA uncertainties:

- **EFAC** (Error Factor): Multiplicative scaling of TOA uncertainties.
- **EQUAD** (Error added in Quadrature): Additional white noise added in
  quadrature to TOA uncertainties.
- **ECORR** (Epoch-Correlated noise): Correlated noise within a single
  observation epoch (same backend, same MJD). ECORR requires block-diagonal
  covariance and is deferred to a future milestone.

The modified uncertainty is:

.. math::

    \\sigma_{\\text{eff}}^2 = \\text{EFAC}^2 \\cdot (\\sigma^2 + \\text{EQUAD}^2)

where :math:`\\sigma` is the original TOA uncertainty, EFAC and EQUAD are
per-backend noise parameters, and the result is in the same units as
:math:`\\sigma` (microseconds).

Noise parameters are parsed from par files in one of these formats:

    T2EFAC  -f <flag_value> <value>
    T2EQUAD -f <flag_value> <value_us>
    ECORR   -f <flag_value> <value_us>

    EFAC  <flag_name> <flag_value> <value>      (Tempo1-style)
    EQUAD <flag_name> <flag_value> <value_us>

Both ``T2EFAC``/``EFAC`` and ``T2EQUAD``/``EQUAD`` are supported as aliases.
The ``-f`` flag selects TOAs matching the ``-f`` flag in the tim file.

References
----------
- Tempo2 documentation (Hobbs et al. 2006)
- NANOGrav noise pipeline conventions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# JAX for JIT-compiled EFAC/EQUAD scaling
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhiteNoiseEntry:
    """A single EFAC, EQUAD, or ECORR specification from a par file.

    Attributes
    ----------
    kind : str
        One of ``'EFAC'``, ``'EQUAD'``, or ``'ECORR'``.
    flag_name : str
        The TIM-file flag used for selection (e.g. ``'f'``, ``'be'``, ``'sys'``).
        Stored *without* the leading ``-``.
    flag_value : str
        The value that the flag must match (e.g. ``'L-wide_PUPPI'``).
    value : float
        The noise parameter value. EFAC is dimensionless; EQUAD and ECORR are
        in microseconds.
    """

    kind: str       # 'EFAC', 'EQUAD', or 'ECORR'
    flag_name: str  # e.g. 'f', 'be', 'sys'
    flag_value: str
    value: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_noise_lines(lines: Sequence[str]) -> List[WhiteNoiseEntry]:
    """Parse EFAC/EQUAD/ECORR lines from a par file.

    Handles both Tempo2 and TempoNest-style formats:

    - ``T2EFAC  -f <flag_value> <value>``  (flag_name is ``'f'``)
    - ``T2EQUAD -f <flag_value> <value>``
    - ``ECORR   -f <flag_value> <value>``
    - ``EFAC  <flag_name> <flag_value> <value>`` (Tempo1-style, ``-`` optional)
    - ``EQUAD <flag_name> <flag_value> <value>``

    Parameters
    ----------
    lines : sequence of str
        Raw lines from a par file.

    Returns
    -------
    list of WhiteNoiseEntry
        Parsed noise entries.
    """
    entries: List[WhiteNoiseEntry] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        keyword = parts[0].upper()

        # -----------------------------------------------------------------
        # Tempo2 format:  T2EFAC -<flag> <value> <number>
        #                 T2EQUAD -<flag> <value> <log10_equad_seconds>
        #                 ECORR   -<flag> <value> <number>
        # -----------------------------------------------------------------
        if keyword == 'T2EFAC':
            if len(parts) < 4:
                continue
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            try:
                value = float(parts[3])
            except ValueError:
                continue
            entries.append(WhiteNoiseEntry('EFAC', flag_name, flag_value, value))

        elif keyword == 'T2EQUAD':
            # Tempo2 convention: value is EQUAD in microseconds.
            # Some TempoNest-generated files erroneously use T2EQUAD with
            # log10(seconds) values -- detect via negative values (EQUAD in mus
            # is always positive).
            if len(parts) < 4:
                continue
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            try:
                raw_val = float(parts[3])
            except ValueError:
                continue
            if raw_val < 0:
                # Negative -> must be log10(seconds) (TempoNest convention)
                equad_us = 10**raw_val * 1e6
            else:
                equad_us = raw_val  # microseconds
            entries.append(WhiteNoiseEntry('EQUAD', flag_name, flag_value, equad_us))

        elif keyword == 'ECORR':
            if len(parts) < 4:
                continue
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            try:
                value = float(parts[3])
            except ValueError:
                continue
            entries.append(WhiteNoiseEntry('ECORR', flag_name, flag_value, value))

        elif keyword == 'DMEFAC':
            # DMEFAC: DM uncertainty scale factor per backend
            # Format: DMEFAC -f <flag_value> <value>
            # Identical structure to T2EFAC but applies to DM measurements
            if len(parts) < 4:
                continue
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            try:
                value = float(parts[3])
            except ValueError:
                continue
            entries.append(WhiteNoiseEntry('DMEFAC', flag_name, flag_value, value))

        elif keyword == 'TNECORR':
            # TempoNest convention: value is log10(ECORR in seconds)
            if len(parts) < 4:
                continue
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            try:
                log10_val = float(parts[3])
            except ValueError:
                continue
            ecorr_us = 10**log10_val * 1e6  # log10(seconds) -> microseconds
            entries.append(WhiteNoiseEntry('ECORR', flag_name, flag_value, ecorr_us))

        # -----------------------------------------------------------------
        # Tempo1/TempoNest format:  EFAC <flag_name> <flag_value> <number>
        #                           EQUAD <flag_name> <flag_value> <number>
        #   (or with the flag prefixed by '-')
        # -----------------------------------------------------------------
        elif keyword == 'EFAC':
            if len(parts) >= 4:
                flag_name = parts[1].lstrip('-')
                flag_value = parts[2]
                try:
                    value = float(parts[3])
                except ValueError:
                    continue
                entries.append(WhiteNoiseEntry('EFAC', flag_name, flag_value, value))
            elif len(parts) == 2:
                # Global EFAC (no flag selector, applies to all TOAs)
                try:
                    value = float(parts[1])
                except ValueError:
                    continue
                entries.append(WhiteNoiseEntry('EFAC', '*', '*', value))

        elif keyword == 'EQUAD':
            if len(parts) >= 4:
                flag_name = parts[1].lstrip('-')
                flag_value = parts[2]
                try:
                    value = float(parts[3])
                except ValueError:
                    continue
                entries.append(WhiteNoiseEntry('EQUAD', flag_name, flag_value, value))
            elif len(parts) == 2:
                # Global EQUAD (no flag selector)
                try:
                    value = float(parts[1])
                except ValueError:
                    continue
                entries.append(WhiteNoiseEntry('EQUAD', '*', '*', value))

    return entries


def parse_noise_params_from_file(par_path: str) -> List[WhiteNoiseEntry]:
    """Read a par file and extract all white noise entries.

    Parameters
    ----------
    par_path : str or Path
        Path to the par file.

    Returns
    -------
    list of WhiteNoiseEntry
    """
    from pathlib import Path
    lines = Path(par_path).read_text().splitlines()
    return parse_noise_lines(lines)


# ---------------------------------------------------------------------------
# TOA-backend matching
# ---------------------------------------------------------------------------

def build_backend_mask(
    toa_flags: List[Dict[str, str]],
    flag_name: str,
    flag_value: str,
) -> np.ndarray:
    """Create a boolean mask selecting TOAs that match a backend flag.

    Parameters
    ----------
    toa_flags : list of dict
        Per-TOA flag dictionaries (from ``SimpleTOA.flags``).
    flag_name : str
        Flag name to match (e.g. ``'f'``, ``'be'``). Use ``'*'`` for global.
    flag_value : str
        Value the flag must have. Use ``'*'`` for global (all TOAs).

    Returns
    -------
    mask : np.ndarray of bool, shape (n_toas,)
    """
    n = len(toa_flags)
    if flag_name == '*' and flag_value == '*':
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    for i, flags in enumerate(toa_flags):
        if flag_name in flags:
            val = flags[flag_name]
            if isinstance(val, list):
                if flag_value in val:
                    mask[i] = True
            elif val == flag_value:
                mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# JIT-compiled EFAC/EQUAD kernel
# ---------------------------------------------------------------------------

@jax.jit
def _apply_efac_equad_jax(
    errors_us: jnp.ndarray,
    efac: jnp.ndarray,
    equad: jnp.ndarray,
) -> jnp.ndarray:
    """sigma_eff = sqrt(EFAC^2 * (sigma^2 + EQUAD^2)), JIT-compiled."""
    return jnp.sqrt(efac**2 * (errors_us**2 + equad**2))


# ---------------------------------------------------------------------------
# Apply EFAC / EQUAD scaling
# ---------------------------------------------------------------------------

def apply_white_noise(
    errors_us: np.ndarray,
    toa_flags: List[Dict[str, str]],
    noise_entries: List[WhiteNoiseEntry],
) -> np.ndarray:
    """Apply EFAC and EQUAD noise scaling to TOA uncertainties.

    The modified uncertainty per TOA is:

    .. math::

        \\sigma_{\\text{eff},i}^2 = \\text{EFAC}_i^2 \\cdot
            (\\sigma_i^2 + \\text{EQUAD}_i^2)

    where EFAC_i and EQUAD_i are determined by the TOA's backend flag.
    If multiple EFAC/EQUAD entries match a TOA, the **last** match wins
    (matching Tempo2's behaviour).

    Parameters
    ----------
    errors_us : np.ndarray, shape (n_toas,)
        Original TOA uncertainties in microseconds.
    toa_flags : list of dict
        Per-TOA flag dictionaries.
    noise_entries : list of WhiteNoiseEntry
        Parsed EFAC/EQUAD entries (ECORR entries are ignored here).

    Returns
    -------
    scaled_errors_us : np.ndarray, shape (n_toas,)
        Modified TOA uncertainties in microseconds.
    """
    n = len(errors_us)
    efac = np.ones(n, dtype=np.float64)   # default EFAC = 1
    equad = np.zeros(n, dtype=np.float64)  # default EQUAD = 0

    for entry in noise_entries:
        if entry.kind == 'EFAC':
            mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
            efac[mask] = entry.value
        elif entry.kind == 'EQUAD':
            mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
            equad[mask] = entry.value
        # ECORR is not applied here (requires block-diagonal covariance)

    # sigma_eff^2 = EFAC^2 * (sigma^2 + EQUAD^2)  -- JIT-compiled via JAX
    return np.asarray(
        _apply_efac_equad_jax(jnp.array(errors_us), jnp.array(efac), jnp.array(equad))
    )
