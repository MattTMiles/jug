"""Writer for pulsar timing .par files.

Writes the current parameter state to a Tempo2/PINT-compatible par file,
including noise parameters in log10 format where appropriate.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


# Parameters that need high-precision string representations
_HIGH_PRECISION_PARAMS = {'F0', 'F1', 'F2', 'F3', 'PEPOCH', 'POSEPOCH', 'DMEPOCH', 'TZRMJD'}

# Metadata keys (written first, in this order)
_METADATA_ORDER = ['PSR', 'EPHEM', 'CLK', 'CLOCK', 'UNITS', 'TIMEEPH', 'T2CMETHOD',
                   'DILATEFREQ', 'DMDATA', 'NTOA', 'START', 'FINISH', 'CHI2', 'INFO']

# Position/astrometry keys (ecliptic)
_ECLIPTIC_POS = ['ELONG', 'ELAT', 'PMELONG', 'PMELAT']

# Position/astrometry keys (equatorial)
_EQUATORIAL_POS = ['RAJ', 'DECJ', 'PMRA', 'PMDEC']

# Spin parameters
_SPIN_PARAMS = ['F0', 'F1', 'F2', 'F3', 'PEPOCH', 'POSEPOCH']

# DM parameters
_DM_PARAMS = ['DM', 'DM1', 'DM2', 'DM3', 'DMEPOCH']

# Binary parameters (written in this order if present)
_BINARY_PARAMS = [
    'BINARY', 'PB', 'A1', 'ECC', 'E', 'T0', 'OM', 'OMDOT', 'PBDOT',
    'GAMMA', 'M2', 'SINI', 'KIN', 'KOM',
    'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT', 'TASC',
    'FB0', 'FB1', 'FB2', 'FB3', 'FB4', 'FB5',
    'A1DOT', 'XDOT', 'EDOT',
    'DR', 'DTH', 'A0', 'B0',
    'SHAPMAX', 'DTHETA', 'LNEDOT',
    'H3', 'H4', 'STIGMA', 'NHARM',
    'K96',
]

# Miscellaneous parameters
_MISC_PARAMS = ['PX', 'NE_SW', 'SWM', 'PLANET_SHAPIRO', 'CORRECT_TROPOSPHERE']

# FD parameters (frequency-dependent delays)
_FD_PREFIX = 'FD'

# TZR parameters
_TZR_PARAMS = ['TZRMJD', 'TZRSITE', 'TZRFRQ']

# Keys to skip (internal/derived)
_SKIP_KEYS = {
    '_high_precision', '_noise_lines', '_jump_lines', '_ecliptic_coords',
    '_ecliptic_frame', '_ecliptic_lon_deg', '_ecliptic_lat_deg',
    '_ecliptic_pmlon', '_ecliptic_pmlat', '_par_timescale',
    'RAJ', 'DECJ', 'PMRA', 'PMDEC',  # handled specially
    'ELONG', 'ELAT', 'PMELONG', 'PMELAT',  # handled specially
    'LAMBDA', 'BETA', 'PMLAMBDA', 'PMBETA',  # internal ecliptic
    'RNAMP', 'RNIDX',  # written as TNRedAmp/TNRedGam
}


def _fmt_line(key: str, value: str, fit_flag: Optional[int] = None,
              uncertainty: Optional[float] = None) -> str:
    """Format a single par file line: KEY VALUE [FIT_FLAG [UNCERTAINTY]]."""
    line = f"{key:<24s}{value}"
    if fit_flag is not None:
        line += f" {fit_flag}"
        if uncertainty is not None and fit_flag == 1:
            line += f" {uncertainty}"
    return line


def _val_str(value: Any, high_prec: Optional[str] = None,
             fmt: str = '') -> str:
    """Convert a parameter value to string, using high-precision if available."""
    if high_prec is not None:
        return high_prec
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if fmt:
            return format(value, fmt)
        # Default: full precision
        return repr(value)
    return str(value)


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

    # --- Header ---
    lines.append(f"# Created by JUG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('#')

    # --- Metadata ---
    for key in _METADATA_ORDER:
        if key in params:
            lines.append(_fmt_line(key, _val_str(params[key])))

    lines.append('')

    # --- Position ---
    is_ecliptic = params.get('_ecliptic_coords', False)
    if is_ecliptic:
        ecl_frame = params.get('_ecliptic_frame', params.get('ECL', 'IERS2010'))
        lines.append(_fmt_line('ECL', str(ecl_frame)))

        elong = params.get('_ecliptic_lon_deg', params.get('ELONG', 0.0))
        elat = params.get('_ecliptic_lat_deg', params.get('ELAT', 0.0))
        pmelong = params.get('_ecliptic_pmlon', params.get('PMELONG', 0.0))
        pmelat = params.get('_ecliptic_pmlat', params.get('PMELAT', 0.0))

        for key, val in [('ELONG', elong), ('ELAT', elat),
                         ('PMELONG', pmelong), ('PMELAT', pmelat)]:
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            # Map fitter names to ecliptic equivalents
            if key == 'ELONG' and 'RAJ' in fit_params:
                fit = 1
                unc = uncertainties.get('RAJ')
            if key == 'ELAT' and 'DECJ' in fit_params:
                fit = 1
                unc = uncertainties.get('DECJ')
            if key == 'PMELONG' and 'PMRA' in fit_params:
                fit = 1
                unc = uncertainties.get('PMRA')
            if key == 'PMELAT' and 'PMDEC' in fit_params:
                fit = 1
                unc = uncertainties.get('PMDEC')
            lines.append(_fmt_line(key, repr(val), fit, unc))
    else:
        raj = params.get('RAJ', '00:00:00')
        decj = params.get('DECJ', '+00:00:00')
        lines.append(_fmt_line('RAJ', str(raj),
                               1 if 'RAJ' in fit_params else 0,
                               uncertainties.get('RAJ')))
        lines.append(_fmt_line('DECJ', str(decj),
                               1 if 'DECJ' in fit_params else 0,
                               uncertainties.get('DECJ')))
        if 'PMRA' in params:
            lines.append(_fmt_line('PMRA', repr(params['PMRA']),
                                   1 if 'PMRA' in fit_params else 0,
                                   uncertainties.get('PMRA')))
        if 'PMDEC' in params:
            lines.append(_fmt_line('PMDEC', repr(params['PMDEC']),
                                   1 if 'PMDEC' in fit_params else 0,
                                   uncertainties.get('PMDEC')))

    # POSEPOCH
    if 'POSEPOCH' in params:
        lines.append(_fmt_line('POSEPOCH', _val_str(params['POSEPOCH'], hp.get('POSEPOCH'))))

    lines.append('')

    # --- Spin parameters ---
    for key in ('F0', 'F1', 'F2', 'F3'):
        if key in params:
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            lines.append(_fmt_line(key, _val_str(params[key], hp.get(key)), fit, unc))
    if 'PEPOCH' in params:
        lines.append(_fmt_line('PEPOCH', _val_str(params['PEPOCH'], hp.get('PEPOCH'))))

    lines.append('')

    # --- DM parameters ---
    for key in _DM_PARAMS:
        if key in params:
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            if key == 'DMEPOCH':
                lines.append(_fmt_line(key, _val_str(params[key], hp.get(key))))
            else:
                lines.append(_fmt_line(key, _val_str(params[key], hp.get(key)), fit, unc))

    lines.append('')

    # --- Parallax ---
    if 'PX' in params:
        lines.append(_fmt_line('PX', repr(params['PX']),
                               1 if 'PX' in fit_params else 0,
                               uncertainties.get('PX')))

    # --- Binary parameters ---
    binary_type = params.get('BINARY')
    if binary_type:
        lines.append('')
        for key in _BINARY_PARAMS:
            if key in params:
                val = params[key]
                if key == 'BINARY':
                    lines.append(_fmt_line(key, str(val)))
                elif isinstance(val, str):
                    lines.append(_fmt_line(key, val))
                else:
                    fit = 1 if key in fit_params else 0
                    unc = uncertainties.get(key)
                    lines.append(_fmt_line(key, repr(val), fit, unc))

    # --- FD parameters ---
    fd_keys = sorted([k for k in params if k.startswith(_FD_PREFIX) and k[2:].isdigit()],
                     key=lambda k: int(k[2:]))
    if fd_keys:
        lines.append('')
        for key in fd_keys:
            fit = 1 if key in fit_params else 0
            unc = uncertainties.get(key)
            lines.append(_fmt_line(key, repr(params[key]), fit, unc))

    # --- Miscellaneous ---
    for key in _MISC_PARAMS:
        if key in params and key != 'PX':
            val = params[key]
            if isinstance(val, str):
                lines.append(_fmt_line(key, val))
            else:
                lines.append(_fmt_line(key, repr(val)))

    lines.append('')

    # --- TZR parameters ---
    for key in _TZR_PARAMS:
        if key in params:
            lines.append(_fmt_line(key, _val_str(params[key], hp.get(key))))

    lines.append('')

    # --- JUMPs ---
    jump_lines = params.get('_jump_lines', [])
    if jump_lines:
        for idx, jl in enumerate(jump_lines):
            parts = jl.strip().split()
            jump_key = f'JUMP{idx + 1}'
            current_val = params.get(jump_key)

            # Reconstruct the JUMP line with updated value
            if parts[1].upper() == 'MJD':
                # MJD-based: JUMP MJD start end offset [fit] [unc]
                prefix = f"JUMP            MJD {parts[2]} {parts[3]}"
                val_str = repr(current_val) if current_val is not None else parts[4]
            else:
                # Flag-based: JUMP -flag val offset [fit] [unc]
                prefix = f"JUMP            {parts[1]} {parts[2]}"
                val_str = repr(current_val) if current_val is not None else parts[3]

            fit = 1 if jump_key in fit_params else 0
            unc = uncertainties.get(jump_key)
            line = f"{prefix}    {val_str}"
            if fit is not None:
                line += f" {fit}"
                if unc is not None and fit == 1:
                    line += f" {unc}"
            lines.append(line)
        lines.append('')

    # --- DMX parameters ---
    _write_dmx_block(params, lines, uncertainties, fit_params)

    # --- Noise: EFAC / EQUAD / ECORR (log10 format for EQUAD/ECORR) ---
    _write_noise_block(params, lines)

    # --- Red noise: TNRedAmp / TNRedGam ---
    _write_red_noise(params, lines)

    # --- DM noise: TNDMAmp / TNDMGam ---
    _write_dm_noise(params, lines)

    # Write file
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


def _write_dmx_block(params: Dict[str, Any], lines: List[str],
                     uncertainties: Dict[str, float],
                     fit_params: Set[str]) -> None:
    """Write DMX range and value blocks."""
    # Standalone DMX key (reference DM for DMX offsets)
    if 'DMX' in params:
        lines.append(_fmt_line('DMX', _val_str(params['DMX'])))

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

        # DMXR1/DMXR2 are MJDs — use fixed-point with 13 decimals to
        # preserve full float64 precision (matches standard par format).
        if r1_key in params:
            lines.append(_fmt_line(r1_key, f'{params[r1_key]:.13f}'))
        if r2_key in params:
            lines.append(_fmt_line(r2_key, f'{params[r2_key]:.13f}'))
        if val_key in params:
            fit = 1 if val_key in fit_params else 0
            unc = uncertainties.get(val_key)
            lines.append(_fmt_line(val_key, repr(params[val_key]), fit, unc))


def _write_noise_block(params: Dict[str, Any], lines: List[str]) -> None:
    """Write EFAC / T2EQUAD / TNECORR lines.

    EFAC is dimensionless (written as-is).
    EQUAD and ECORR are converted to log10(seconds) and written as
    T2EQUAD and TNECORR respectively, so the keyword matches the unit
    convention that JUG's reader expects.
    """
    noise_lines = params.get('_noise_lines', [])
    if not noise_lines:
        return

    lines.append('')
    for raw_line in noise_lines:
        parts = raw_line.strip().split()
        keyword = parts[0].upper()

        if keyword in ('T2EFAC', 'EFAC'):
            # EFAC: dimensionless, write as-is
            flag = parts[1]
            flag_val = parts[2]
            value = float(parts[3])
            lines.append(f"EFAC            {flag} {flag_val}    {repr(value)}")

        elif keyword in ('T2EQUAD', 'EQUAD'):
            # EQUAD in microseconds -> log10(seconds)
            flag = parts[1]
            flag_val = parts[2]
            equad_us = float(parts[3])
            log10_sec = math.log10(equad_us * 1e-6)
            lines.append(f"T2EQUAD         {flag} {flag_val}    {log10_sec}")

        elif keyword in ('ECORR',):
            # ECORR in microseconds -> TNECORR as log10(seconds)
            flag = parts[1]
            flag_val = parts[2]
            ecorr_us = float(parts[3])
            log10_sec = math.log10(ecorr_us * 1e-6)
            lines.append(f"TNECORR         {flag} {flag_val}    {log10_sec}")

        elif keyword == 'TNECORR':
            # Already in log10(seconds) — write as-is
            lines.append(raw_line)

        elif keyword in ('DMEFAC', 'DMJUMP'):
            # Pass through unchanged
            lines.append(raw_line)

        else:
            # Unknown noise keyword — preserve raw line
            lines.append(raw_line)


def _write_red_noise(params: Dict[str, Any], lines: List[str]) -> None:
    """Write red noise as TNRedAmp (log10_A) and TNRedGam."""
    if 'RNAMP' not in params and 'RNIDX' not in params:
        return

    lines.append('')

    if 'RNAMP' in params:
        rnamp = float(params['RNAMP'])
        _SEC_PER_YR = 365.25 * 86400.0
        log10_A = math.log10(2.0 * math.pi * math.sqrt(3.0) / (_SEC_PER_YR * 1e6) * rnamp)
        lines.append(_fmt_line('TNRedAmp', repr(log10_A)))

    if 'RNIDX' in params:
        gamma = -float(params['RNIDX'])  # Sign flip: RNIDX -> TNRedGam
        lines.append(_fmt_line('TNRedGam', repr(gamma)))

    # Number of harmonics
    n_harm = params.get('RNC', params.get('TNREDC', params.get('TNRedC')))
    if n_harm is not None:
        lines.append(_fmt_line('TNRedC', str(int(n_harm))))


def _write_dm_noise(params: Dict[str, Any], lines: List[str]) -> None:
    """Write DM noise as TNDMAmp / TNDMGam if present."""
    has_dm_amp = any(k in params for k in ('TNDMAMP', 'TNDMAmp', 'DM_log10_A'))
    has_dm_gam = any(k in params for k in ('TNDMGAM', 'TNDMGam', 'DM_gamma'))

    if not has_dm_amp and not has_dm_gam:
        return

    lines.append('')

    for key_candidates, out_key in [
        (('TNDMAMP', 'TNDMAmp', 'DM_log10_A'), 'TNDMAmp'),
        (('TNDMGAM', 'TNDMGam', 'DM_gamma'), 'TNDMGam'),
        (('TNDMC', 'TNDMc'), 'TNDMC'),
    ]:
        for k in key_candidates:
            if k in params:
                lines.append(_fmt_line(out_key, repr(params[k])))
                break
