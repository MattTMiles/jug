"""Writer for pulsar timing .par files.

Writes the current parameter state to a Tempo2/PINT-compatible par file,
including noise parameters in log10 format where appropriate.

All parameters stored in the JUG params dict are written back faithfully:
spin derivatives (F0–F20), DM derivatives (DM–DM20), binary frequency
derivatives (FB0–FB20), FD parameters, JUMPs, FDJUMPs, DMX ranges,
white noise (EFAC/EQUAD/ECORR), and correlated noise (red, DM, chromatic,
band, group).  Any unrecognised keys are written at the end so that no
information is silently dropped.
"""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


# Metadata keys (written first, in this order)
_METADATA_ORDER = [
    'PSR', 'EPHEM', 'CLK', 'CLOCK', 'UNITS', 'TIMEEPH', 'T2CMETHOD',
    'DILATEFREQ', 'DMDATA', 'NTOA', 'START', 'FINISH', 'CHI2', 'INFO',
    'PLANET_SHAPIRO', 'CORRECT_TROPOSPHERE', 'NE_SW', 'SWM',
    'CLK_CORR_CHAIN',
]

# Binary parameters — fixed-name keys written in this order when present.
# FB0–FB20 are handled dynamically after these.
_BINARY_FIXED = [
    'BINARY', 'PB', 'A1', 'ECC', 'E', 'T0', 'OM', 'OMDOT', 'PBDOT',
    'GAMMA', 'M2', 'SINI', 'KIN', 'KOM',
    'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT', 'TASC',
    'A1DOT', 'XDOT', 'EDOT',
    'DR', 'DTH', 'A0', 'B0',
    'SHAPMAX', 'DTHETA', 'LNEDOT',
    'H3', 'H4', 'STIGMA', 'NHARM',
    'K96',
]

# TZR parameters
_TZR_PARAMS = ['TZRMJD', 'TZRSITE', 'TZRFRQ']

# All correlated-noise keywords (any case variant) handled by dedicated
# write sections.  Used to exclude them from the catch-all.
_NOISE_PARAM_KEYS = {
    # Tempo2-native
    'RNAMP', 'RNIDX', 'RNC',
    # JUG-native (all-caps)
    'REDAMP', 'REDGAM', 'REDC',
    'DMAMP', 'DMGAM', 'DMC',
    'CHROMAMP', 'CHROMGAM', 'CHROMIDX', 'CHROMC',
    'BANDNOISE', 'GROUPNOISE', 'GROUPSETSPAN',
    # TempoNest (mixed-case and uppercase)
    'TNREDAMP', 'TNREDGAM', 'TNREDC',
    'TNDMAMP', 'TNDMGAM', 'TNDMC',
    'TNCHROMAMP', 'TNCHROMGAM', 'TNCHROMIDX', 'TNCHROMC',
    'TNBANDNOISE', 'TNGROUPNOISE', 'TNGROUPSETSPAN',
    # Enterprise convention
    'RN_LOG10_A', 'RN_GAMMA', 'RN_NCOEFF',
    'DM_LOG10_A', 'DM_GAMMA', 'DM_NCOEFF',
    'CHROM_LOG10_A', 'CHROM_GAMMA', 'CHROM_IDX', 'CHROM_NCOEFF',
}

# Internal/derived keys that must never be written to the par file.
_SKIP_KEYS = {
    '_high_precision', '_noise_lines', '_jump_lines', '_fdjump_lines',
    '_ecliptic_coords', '_ecliptic_frame',
    '_ecliptic_lon_deg', '_ecliptic_lat_deg',
    '_ecliptic_pmlon', '_ecliptic_pmlat', '_par_timescale',
    '_fit_flags',
    # Position keys handled by the dedicated position section
    'RAJ', 'DECJ', 'PMRA', 'PMDEC',
    'ELONG', 'ELAT', 'PMELONG', 'PMELAT',
    'LAMBDA', 'BETA', 'PMLAMBDA', 'PMBETA',
    'ECL',
}

# Regex for numbered-series keys handled by dedicated sections.
_NUMBERED_SERIES_RE = re.compile(
    r'^(F|DM|FB|FD|FDJUMP|DMX_|DMXR1_|DMXR2_|JUMP)\d')


def _is_noise_model_key(key: str) -> bool:
    """Return True if *key* is a correlated-noise parameter written elsewhere."""
    return key.upper() in _NOISE_PARAM_KEYS


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt_line(key: str, value: str, fit_flag: Optional[int] = None,
              uncertainty: Optional[float] = None) -> str:
    """Format a single par file line: KEY VALUE [FIT_FLAG [UNCERTAINTY]]."""
    line = f"{key:<24s}{value}"
    if fit_flag is not None:
        line += f" {fit_flag}"
        if uncertainty is not None and fit_flag == 1:
            line += f" {uncertainty}"
    return line


def _repr_num(value: Any) -> str:
    """repr() a numeric value, converting numpy scalars to Python floats."""
    if isinstance(value, np.floating):
        return repr(float(value))
    if isinstance(value, np.integer):
        return repr(int(value))
    return repr(value)


def _val_str(value: Any, high_prec: Optional[str] = None,
             fmt: str = '') -> str:
    """Convert a parameter value to string, using high-precision if available."""
    if high_prec is not None:
        return high_prec
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if fmt:
            return format(v, fmt)
        return repr(v)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _write_param(lines: List[str], key: str, params: Dict[str, Any],
                 hp: Dict[str, str], fit_params: Set[str],
                 uncertainties: Dict[str, float]) -> None:
    """Write a single parameter line with fit flag and uncertainty."""
    val = params[key]
    fit = 1 if key in fit_params else 0
    unc = uncertainties.get(key)
    lines.append(_fmt_line(key, _val_str(val, hp.get(key)), fit, unc))


def _collect_numbered(params: Dict[str, Any], prefix: str) -> List[str]:
    """Return sorted list of keys matching prefix+digit (e.g. F0, F1, …)."""
    keys = [k for k in params
            if k.startswith(prefix) and k[len(prefix):].isdigit()]
    return sorted(keys, key=lambda k: int(k[len(prefix):]))


# ── Main entry point ─────────────────────────────────────────────────────

def write_par_file(
    params: Dict[str, Any],
    path: Union[str, Path],
    uncertainties: Optional[Dict[str, float]] = None,
    fit_params: Optional[Set[str]] = None,
) -> None:
    """Write a par file from a JUG params dict.

    Parameters
    ----------
    params : dict
        Parameter dictionary (as produced by ``parse_par_file`` and
        updated by the fitter).
    path : str or Path
        Output file path.
    uncertainties : dict, optional
        Map of parameter name to 1-sigma uncertainty.
    fit_params : set of str, optional
        Set of parameter names that were fitted (written with fit flag ``1``).
    """
    path = Path(path)
    if uncertainties is None:
        uncertainties = {}
    if fit_params is None:
        fit_params = set()

    hp = params.get('_high_precision', {})
    lines: List[str] = []
    written: set = set()  # track which keys we've already emitted

    # --- Header ---
    lines.append(f"# Created by JUG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('#')

    # --- Metadata ---
    for key in _METADATA_ORDER:
        if key in params:
            lines.append(_fmt_line(key, _val_str(params[key])))
            written.add(key)

    lines.append('')

    # --- Position / astrometry ---
    _write_position(params, lines, hp, fit_params, uncertainties, written)
    lines.append('')

    # --- Spin parameters: F0–F20, PEPOCH ---
    for key in _collect_numbered(params, 'F'):
        _write_param(lines, key, params, hp, fit_params, uncertainties)
        written.add(key)
    if 'PEPOCH' in params:
        lines.append(_fmt_line('PEPOCH', _val_str(params['PEPOCH'], hp.get('PEPOCH'))))
        written.add('PEPOCH')

    lines.append('')

    # --- DM parameters: DM, DM1–DM20, DMEPOCH ---
    if 'DM' in params:
        _write_param(lines, 'DM', params, hp, fit_params, uncertainties)
        written.add('DM')
    for key in _collect_numbered(params, 'DM'):
        _write_param(lines, key, params, hp, fit_params, uncertainties)
        written.add(key)
    if 'DMEPOCH' in params:
        lines.append(_fmt_line('DMEPOCH', _val_str(params['DMEPOCH'], hp.get('DMEPOCH'))))
        written.add('DMEPOCH')

    lines.append('')

    # --- Parallax ---
    if 'PX' in params:
        _write_param(lines, 'PX', params, hp, fit_params, uncertainties)
        written.add('PX')

    # --- Binary parameters ---
    _write_binary(params, lines, hp, fit_params, uncertainties, written)

    # --- FD parameters ---
    fd_keys = _collect_numbered(params, 'FD')
    if fd_keys:
        lines.append('')
        for key in fd_keys:
            _write_param(lines, key, params, hp, fit_params, uncertainties)
            written.add(key)

    lines.append('')

    # --- TZR parameters ---
    for key in _TZR_PARAMS:
        if key in params:
            lines.append(_fmt_line(key, _val_str(params[key], hp.get(key))))
            written.add(key)

    lines.append('')

    # --- JUMPs ---
    _write_jumps(params, lines, fit_params, uncertainties, written)

    # --- FDJUMPs ---
    _write_fdjumps(params, lines, fit_params, uncertainties, written)

    # --- DMX parameters ---
    _write_dmx_block(params, lines, uncertainties, fit_params, written)

    # --- Noise: EFAC / EQUAD / ECORR ---
    _write_noise_block(params, lines)

    # --- Red noise ---
    _write_red_noise(params, lines, written)

    # --- DM noise ---
    _write_dm_noise(params, lines, written)

    # --- Chromatic noise ---
    _write_chromatic_noise(params, lines, written)

    # --- Catch-all: write any remaining public keys not yet emitted ---
    _write_remaining(params, lines, hp, fit_params, uncertainties, written)

    # Write file
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


# ── Section writers ───────────────────────────────────────────────────────

def _write_position(params, lines, hp, fit_params, uncertainties, written):
    """Write position / proper-motion block (ecliptic or equatorial)."""
    is_ecliptic = params.get('_ecliptic_coords', False)
    if is_ecliptic:
        ecl_frame = params.get('_ecliptic_frame', params.get('ECL', 'IERS2010'))
        lines.append(_fmt_line('ECL', str(ecl_frame)))
        written.add('ECL')

        elong = params.get('_ecliptic_lon_deg', params.get('ELONG', 0.0))
        elat = params.get('_ecliptic_lat_deg', params.get('ELAT', 0.0))
        pmelong = params.get('_ecliptic_pmlon', params.get('PMELONG', 0.0))
        pmelat = params.get('_ecliptic_pmlat', params.get('PMELAT', 0.0))

        # Map fitter names (RAJ/DECJ/PMRA/PMDEC) to ecliptic equivalents
        _fitter_map = {
            'ELONG': 'RAJ', 'ELAT': 'DECJ',
            'PMELONG': 'PMRA', 'PMELAT': 'PMDEC',
        }
        for key, val in [('ELONG', elong), ('ELAT', elat),
                         ('PMELONG', pmelong), ('PMELAT', pmelat)]:
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            equiv = _fitter_map.get(key)
            if equiv and equiv in fit_params:
                fit = 1
                unc = uncertainties.get(equiv, unc)
            lines.append(_fmt_line(key, _repr_num(val), fit, unc))
        written.update(['ELONG', 'ELAT', 'PMELONG', 'PMELAT',
                        'RAJ', 'DECJ', 'PMRA', 'PMDEC'])
    else:
        raj = params.get('RAJ', '00:00:00')
        decj = params.get('DECJ', '+00:00:00')
        lines.append(_fmt_line('RAJ', str(raj),
                               1 if 'RAJ' in fit_params else 0,
                               uncertainties.get('RAJ')))
        lines.append(_fmt_line('DECJ', str(decj),
                               1 if 'DECJ' in fit_params else 0,
                               uncertainties.get('DECJ')))
        written.update(['RAJ', 'DECJ'])
        if 'PMRA' in params:
            lines.append(_fmt_line('PMRA', _repr_num(params['PMRA']),
                                   1 if 'PMRA' in fit_params else 0,
                                   uncertainties.get('PMRA')))
            written.add('PMRA')
        if 'PMDEC' in params:
            lines.append(_fmt_line('PMDEC', _repr_num(params['PMDEC']),
                                   1 if 'PMDEC' in fit_params else 0,
                                   uncertainties.get('PMDEC')))
            written.add('PMDEC')

    if 'POSEPOCH' in params:
        lines.append(_fmt_line('POSEPOCH', _val_str(params['POSEPOCH'], hp.get('POSEPOCH'))))
        written.add('POSEPOCH')


def _write_binary(params, lines, hp, fit_params, uncertainties, written):
    """Write binary model parameters including FB0–FB20."""
    binary_type = params.get('BINARY')
    if not binary_type:
        return

    lines.append('')
    for key in _BINARY_FIXED:
        if key in params:
            val = params[key]
            if key == 'BINARY':
                lines.append(_fmt_line(key, str(val)))
            elif isinstance(val, str):
                lines.append(_fmt_line(key, val))
            else:
                fit = 1 if key in fit_params else 0
                unc = uncertainties.get(key)
                lines.append(_fmt_line(key, _repr_num(val), fit, unc))
            written.add(key)

    # FB0–FB20 (dynamic)
    for key in _collect_numbered(params, 'FB'):
        if key not in written:
            _write_param(lines, key, params, hp, fit_params, uncertainties)
            written.add(key)


def _write_jumps(params, lines, fit_params, uncertainties, written):
    """Write JUMP lines, updating values from params[JUMPn]."""
    jump_lines = params.get('_jump_lines', [])
    if not jump_lines:
        return

    for idx, jl in enumerate(jump_lines):
        parts = jl.strip().split()
        jump_key = f'JUMP{idx + 1}'
        current_val = params.get(jump_key)

        if parts[1].upper() == 'MJD':
            prefix = f"JUMP            MJD {parts[2]} {parts[3]}"
            val_str = _repr_num(current_val) if current_val is not None else parts[4]
        else:
            prefix = f"JUMP            {parts[1]} {parts[2]}"
            val_str = _repr_num(current_val) if current_val is not None else parts[3]

        fit = 1 if jump_key in fit_params else 0
        unc = uncertainties.get(jump_key)
        line = f"{prefix}    {val_str} {fit}"
        if unc is not None and fit == 1:
            line += f" {unc}"
        lines.append(line)
        written.add(jump_key)
    lines.append('')


def _write_fdjumps(params, lines, fit_params, uncertainties, written):
    """Write FDJUMP lines, updating values from params[FDJUMPn_m]."""
    fdjump_lines = params.get('_fdjump_lines', [])
    if not fdjump_lines:
        return

    # Check for FDJUMP_SCALE
    if 'FDJUMP_SCALE' in params:
        lines.append(_fmt_line('FDJUMP_SCALE', str(params['FDJUMP_SCALE'])))
        written.add('FDJUMP_SCALE')

    for raw_line in fdjump_lines:
        fparts = raw_line.strip().split()
        key_raw = fparts[0].upper()
        if key_raw == 'FDJUMP_SCALE':
            continue  # already handled

        # Parse: FDJUMPn -flag flagval value [fit_flag]
        m = re.match(r'FDJUMP(\d+)', key_raw)
        if not m:
            lines.append(raw_line)
            continue

        fd_idx = m.group(1)
        flag = fparts[1]
        flag_val = fparts[2]

        # Find the internal key for this FDJUMP entry
        fdjump_keys = sorted(
            [k for k in params if k.startswith(f'FDJUMP{fd_idx}_')],
            key=lambda k: int(k.split('_')[1]) if '_' in k else 0
        )

        if fdjump_keys:
            # Use first matching key's value (there may be multiple per group)
            for fk in fdjump_keys:
                if fk in params:
                    val = params[fk]
                    fit = 1 if fk in fit_params else 0
                    unc = uncertainties.get(fk)
                    line = f"FDJUMP{fd_idx}        {flag} {flag_val}    {_repr_num(val)}"
                    if fit:
                        line += f" {fit}"
                        if unc is not None:
                            line += f" {unc}"
                    else:
                        line += " 0"
                    lines.append(line)
                    written.add(fk)
                    break
        else:
            # No internal key found — preserve raw line
            lines.append(raw_line)

    # Mark all FDJUMP keys
    for k in list(params.keys()):
        if k.startswith('FDJUMP'):
            written.add(k)
    lines.append('')


def _write_dmx_block(params, lines, uncertainties, fit_params, written):
    """Write DMX range and value blocks."""
    if 'DMX' in params:
        lines.append(_fmt_line('DMX', _val_str(params['DMX'])))
        written.add('DMX')

    # Collect DMX indices
    dmx_indices = set()
    for key in params:
        if key.startswith('DMX_') and key[4:].isdigit():
            dmx_indices.add(key[4:])
        elif key.startswith('DMXR1_') and key[6:].isdigit():
            dmx_indices.add(key[6:])

    if not dmx_indices:
        return

    lines.append('')
    for idx in sorted(dmx_indices):
        r1_key = f'DMXR1_{idx}'
        r2_key = f'DMXR2_{idx}'
        val_key = f'DMX_{idx}'

        if r1_key in params:
            lines.append(_fmt_line(r1_key, f'{float(params[r1_key]):.13f}'))
            written.add(r1_key)
        if r2_key in params:
            lines.append(_fmt_line(r2_key, f'{float(params[r2_key]):.13f}'))
            written.add(r2_key)
        if val_key in params:
            fit = 1 if val_key in fit_params else 0
            unc = uncertainties.get(val_key)
            lines.append(_fmt_line(val_key, _repr_num(params[val_key]), fit, unc))
            written.add(val_key)


def _write_noise_block(params: Dict[str, Any], lines: List[str]) -> None:
    """Write EFAC / T2EQUAD / TNECORR lines.

    EFAC is dimensionless (written as-is).
    EQUAD and ECORR are converted to log10(seconds) and written as
    T2EQUAD and TNECORR respectively, so the keyword matches the unit
    convention that JUG's reader expects.
    """
    noise_lines = params.get('_noise_lines', [])

    # GROUPSETSPAN modifier (standalone key, not a noise line)
    gss = params.get('GROUPSETSPAN', params.get('TNGROUPSETSPAN'))
    has_group_flag = gss is not None

    if not noise_lines and not has_group_flag:
        return

    lines.append('')
    if has_group_flag:
        lines.append(_fmt_line('GROUPSETSPAN', str(int(gss))))

    for raw_line in noise_lines:
        parts = raw_line.strip().split()
        keyword = parts[0].upper()

        if keyword in ('T2EFAC', 'EFAC'):
            flag = parts[1]
            flag_val = parts[2]
            value = float(parts[3])
            lines.append(f"EFAC            {flag} {flag_val}    {repr(value)}")

        elif keyword in ('T2EQUAD', 'EQUAD'):
            flag = parts[1]
            flag_val = parts[2]
            equad_us = float(parts[3])
            log10_sec = math.log10(equad_us * 1e-6)
            lines.append(f"T2EQUAD         {flag} {flag_val}    {log10_sec}")

        elif keyword in ('ECORR',):
            flag = parts[1]
            flag_val = parts[2]
            ecorr_us = float(parts[3])
            log10_sec = math.log10(ecorr_us * 1e-6)
            lines.append(f"TNECORR         {flag} {flag_val}    {log10_sec}")

        elif keyword == 'TNECORR':
            lines.append(raw_line)

        elif keyword in ('DMEFAC', 'DMJUMP'):
            lines.append(raw_line)

        elif keyword in ('TNBANDNOISE', 'BANDNOISE'):
            # Normalize to JUG-native keyword
            parts[0] = 'BANDNOISE'
            lines.append(' '.join(parts))

        elif keyword in ('TNGROUPNOISE', 'GROUPNOISE'):
            # Normalize to JUG-native keyword
            parts[0] = 'GROUPNOISE'
            lines.append(' '.join(parts))

        elif keyword in ('TNGROUPSETSPAN', 'GROUPSETSPAN'):
            parts[0] = 'GROUPSETSPAN'
            lines.append(' '.join(parts))

        else:
            # Unknown noise keyword — preserve raw line
            lines.append(raw_line)


def _write_correlated_noise(params: Dict[str, Any], lines: List[str],
                            written: set,
                            key_map: List[tuple],
                            count_keys: Set[str]) -> None:
    """Write a group of correlated-noise parameters.

    Parameters
    ----------
    key_map : list of (candidates, output_key)
        Each entry maps one or more candidate param names to the canonical
        output keyword.  The first candidate found in *params* wins.
    count_keys : set of str
        Output keys that represent integer counts (written with ``int()``).
    """
    found = any(k in params for candidates, _ in key_map for k in candidates)
    if not found:
        return

    lines.append('')
    for key_candidates, out_key in key_map:
        for k in key_candidates:
            if k in params:
                val = params[k]
                if out_key in count_keys:
                    lines.append(_fmt_line(out_key, str(int(val))))
                else:
                    lines.append(_fmt_line(out_key, _repr_num(val)))
                written.add(k)
                break


def _write_red_noise(params: Dict[str, Any], lines: List[str],
                     written: set) -> None:
    """Write red noise as REDAMP / REDGAM / REDC.

    If the input used Tempo2-native RNAMP/RNIDX, converts to log10_A/gamma
    first.  All output uses JUG-native all-caps keywords.
    """
    # Tempo2-native RNAMP/RNIDX → convert to log10_A/gamma
    if 'RNAMP' in params:
        rnamp = float(params['RNAMP'])
        _SEC_PER_YR = 365.25 * 86400.0
        log10_A = math.log10(
            2.0 * math.pi * math.sqrt(3.0) / (_SEC_PER_YR * 1e6) * rnamp)
        lines.append('')
        lines.append(_fmt_line('REDAMP', repr(log10_A)))
        written.update(['RNAMP', 'RNIDX', 'RNC'])
        if 'RNIDX' in params:
            gamma = -float(params['RNIDX'])
            lines.append(_fmt_line('REDGAM', repr(gamma)))
        n_harm = params.get('RNC', params.get('TNREDC', params.get('TNRedC')))
        if n_harm is not None:
            lines.append(_fmt_line('REDC', str(int(n_harm))))
        return

    # Already in log10_A/gamma convention (any alias)
    _write_correlated_noise(params, lines, written, [
        (('REDAMP', 'TNREDAMP', 'TNRedAmp', 'RN_log10_A'), 'REDAMP'),
        (('REDGAM', 'TNREDGAM', 'TNRedGam', 'RN_gamma'), 'REDGAM'),
        (('REDC', 'TNREDC', 'TNRedC', 'RN_ncoeff'), 'REDC'),
    ], count_keys={'REDC'})


def _write_dm_noise(params: Dict[str, Any], lines: List[str],
                    written: set) -> None:
    """Write DM noise as DMAMP / DMGAM / DMC (enterprise convention).

    JUG outputs DM noise in the enterprise convention (no offset).
    If the source value came from a TempoNest key (TNDMAMP), the
    TNDM_OFFSET is subtracted so the written value is correct.
    """
    from jug.noise.red_noise import TNDM_OFFSET

    # Amplitude — find source key and apply offset if needed
    amp_val = None
    for k in ('DMAMP', 'TNDMAMP', 'TNDMAmp', 'DM_log10_A'):
        if k in params:
            amp_val = float(params[k])
            if k.upper().startswith('TN'):
                amp_val -= TNDM_OFFSET
            written.add(k)
            break

    # Gamma
    gam_val = None
    for k in ('DMGAM', 'TNDMGAM', 'TNDMGam', 'DM_gamma'):
        if k in params:
            gam_val = float(params[k])
            written.add(k)
            break

    # Count
    count_val = None
    for k in ('DMC', 'TNDMC', 'TNDMc', 'DM_ncoeff'):
        if k in params:
            count_val = int(params[k])
            written.add(k)
            break

    if amp_val is None and gam_val is None:
        return

    lines.append('')
    if amp_val is not None:
        lines.append(_fmt_line('DMAMP', repr(amp_val)))
    if gam_val is not None:
        lines.append(_fmt_line('DMGAM', repr(gam_val)))
    if count_val is not None:
        lines.append(_fmt_line('DMC', str(count_val)))


def _write_chromatic_noise(params: Dict[str, Any], lines: List[str],
                           written: set) -> None:
    """Write chromatic noise as CHROMAMP / CHROMGAM / CHROMIDX / CHROMC."""
    _write_correlated_noise(params, lines, written, [
        (('CHROMAMP', 'TNCHROMAMP', 'TNChromAmp', 'CHROM_log10_A'), 'CHROMAMP'),
        (('CHROMGAM', 'TNCHROMGAM', 'TNChromGam', 'CHROM_gamma'), 'CHROMGAM'),
        (('CHROMIDX', 'TNCHROMIDX', 'TNChromIdx', 'CHROM_idx'), 'CHROMIDX'),
        (('CHROMC', 'TNCHROMC', 'TNChromC', 'CHROM_ncoeff'), 'CHROMC'),
    ], count_keys={'CHROMC'})


def _write_remaining(params, lines, hp, fit_params, uncertainties, written):
    """Write any params not yet emitted (catch-all to prevent silent drops)."""
    remaining = []
    for key in sorted(params.keys()):
        if key in written:
            continue
        if key in _SKIP_KEYS:
            continue
        if key.startswith('_'):
            continue
        if _is_noise_model_key(key):
            # Already handled or internal alias — skip
            continue
        if _NUMBERED_SERIES_RE.match(key):
            # DMX/JUMP/FD/FDJUMP already handled above
            continue
        remaining.append(key)

    if remaining:
        lines.append('')
        lines.append('# Additional parameters')
        for key in remaining:
            val = params[key]
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            if isinstance(val, str):
                lines.append(_fmt_line(key, val, fit if fit else None, unc))
            else:
                lines.append(_fmt_line(key, _val_str(val, hp.get(key)),
                                       fit, unc))
