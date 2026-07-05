"""Irwin & Fukushima (1999) time ephemeris reader (tempo2 ``ifteph.C``).

Loads ``TIMEEPH_short.te405`` from the JUG data directory and evaluates
``IFTE_DeltaT`` / ``IFTE_DeltaTDot`` for tempo2-native ``tt2tb`` corrections.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jug.utils.constants import SECS_PER_DAY

IFTE_LC = 1.48082686742e-8
IFTE_MJD0 = 43144.0003725
IFTE_TEPH0_SEC = -65.564518e-6

_DEFAULT_EPHEM = (
    Path(__file__).resolve().parent.parent.parent / "data" / "ephemeris" / "TIMEEPH_short.te405"
)


@dataclass
class _InterpInfo:
    pc: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=np.float64))
    vc: np.ndarray = field(default_factory=lambda: np.zeros(18, dtype=np.float64))
    twot: float = 0.0
    np_: int = 2
    nv: int = 3


@dataclass
class _IFTEState:
    f: object | None = None
    start_jd: float = 0.0
    end_jd: float = 0.0
    step_jd: float = 0.0
    swap_endian: bool = False
    reclen: int = 0
    irec: int = -1
    buf: np.ndarray = field(default_factory=lambda: np.zeros(322, dtype=np.float64))
    iinfo: _InterpInfo = field(default_factory=_InterpInfo)
    ipt: np.ndarray = field(default_factory=lambda: np.zeros((2, 3), dtype=np.int64))


_STATE = _IFTEState()


def _swap8(data: bytes) -> bytes:
    return data[7::-1]


def _read_double(f, swap: bool) -> float:
    raw = f.read(8)
    if len(raw) < 8:
        raise EOFError("unexpected EOF in IFTE ephemeris")
    if swap:
        raw = _swap8(raw)
    return struct.unpack("<d", raw)[0]


def _read_int(f, swap: bool) -> int:
    raw = f.read(4)
    if len(raw) < 4:
        raise EOFError("unexpected EOF in IFTE ephemeris")
    if swap:
        raw = raw[3::-1]
    return struct.unpack("<i", raw)[0]


def ifte_init(path: str | Path | None = None) -> None:
    """Open the IFTE coefficient file (idempotent)."""
    global _STATE
    fname = Path(path) if path is not None else _DEFAULT_EPHEM
    if _STATE.f is not None:
        return
    if not fname.exists():
        raise FileNotFoundError(f"IFTE ephemeris not found: {fname}")

    f = open(fname, "rb")
    f.read(252)
    f.read(12)
    raw_start = f.read(8)
    raw_end = f.read(8)
    raw_step = f.read(8)
    raw_ncon = f.read(4)
    start_jd = struct.unpack("<d", raw_start)[0]
    end_jd = struct.unpack("<d", raw_end)[0]
    step_jd = struct.unpack("<d", raw_step)[0]
    ncon = struct.unpack("<i", raw_ncon)[0]
    swap = ncon != 2
    if swap:
        start_jd = struct.unpack("<d", _swap8(raw_start))[0]
        end_jd = struct.unpack("<d", _swap8(raw_end))[0]
        step_jd = struct.unpack("<d", _swap8(raw_step))[0]
        ncon = struct.unpack("<i", _swap8(raw_ncon))[0]
    if ncon != 2:
        raise ValueError(f"unexpected IFTE ncon={ncon} in {fname}")

    ipt_raw = f.read(24)
    ipt = np.frombuffer(ipt_raw, dtype="<i4").reshape(2, 3).copy()
    if swap:
        ipt = ipt.byteswap()

    reclen = 4 * 2 * (ipt[1, 0] - 1 + 3 * ipt[1, 1] * ipt[1, 2])
    f.seek(reclen, 0)
    _read_double(f, swap)
    _read_double(f, swap)

    iinfo = _InterpInfo()
    iinfo.pc[0] = 1.0
    iinfo.pc[1] = 0.0
    iinfo.vc[1] = 1.0

    _STATE = _IFTEState(
        f=f,
        start_jd=start_jd,
        end_jd=end_jd,
        step_jd=step_jd,
        swap_endian=swap,
        reclen=reclen,
        irec=-1,
        buf=np.zeros(reclen // 8, dtype=np.float64),
        iinfo=iinfo,
        ipt=ipt,
    )


def ifte_close() -> None:
    global _STATE
    if _STATE.f is not None:
        _STATE.f.close()
    _STATE = _IFTEState()


def _ifte_interp(
    iinfo: _InterpInfo,
    coef: np.ndarray,
    t: tuple[float, float],
    ncf: int,
    ncm: int,
    na: int,
    ifl: int,
) -> np.ndarray:
    posvel = np.zeros(ncm * ifl, dtype=np.float64)
    dna = float(na)
    frac_t0, int_t0 = math.modf(t[0])
    temp = dna * t[0]
    temp_frac, temp_int = math.modf(temp)
    l = int(temp_int - int_t0)
    tc = 2.0 * (temp_frac + frac_t0) - 1.0

    if tc != iinfo.pc[1]:
        iinfo.np_ = 2
        iinfo.nv = 3
        iinfo.pc[1] = tc
        iinfo.twot = tc + tc

    if iinfo.np_ < ncf:
        for i in range(ncf - iinfo.np_):
            idx = iinfo.np_ + i
            iinfo.pc[idx] = iinfo.twot * iinfo.pc[idx - 1] - iinfo.pc[idx - 2]
        iinfo.np_ = ncf

    for i in range(ncm):
        coeff_ptr = ncf * (i + l * ncm + 1)
        pc_ptr = ncf
        val = 0.0
        for _ in range(ncf):
            pc_ptr -= 1
            coeff_ptr -= 1
            val += iinfo.pc[pc_ptr] * coef[coeff_ptr]
        posvel[i] = val

    if ifl <= 1:
        return posvel

    vfac = (dna + dna) / t[1]
    iinfo.vc[2] = iinfo.twot + iinfo.twot
    if iinfo.nv < ncf:
        for i in range(ncf - iinfo.nv):
            idx = iinfo.nv + i
            iinfo.vc[idx] = (
                iinfo.twot * iinfo.vc[idx - 1]
                + iinfo.pc[idx - 1]
                + iinfo.pc[idx - 1]
                - iinfo.vc[idx - 2]
            )
        iinfo.nv = ncf

    for i in range(ncm):
        tval = 0.0
        coeff_ptr = ncf * (i + l * ncm + 1)
        vc_ptr = ncf
        for _ in range(ncf):
            vc_ptr -= 1
            coeff_ptr -= 1
            tval += iinfo.vc[vc_ptr] * coef[coeff_ptr]
        posvel[i + ncm] = tval * vfac
    return posvel


def _ifte_get_vals(jd0: float, jd1: float, kind: int) -> np.ndarray:
    if _STATE.f is None:
        ifte_init()

    whole0 = math.floor(jd0 - 0.5)
    frac0 = jd0 - 0.5 - whole0
    whole1 = math.floor(jd1)
    frac1 = jd1 - whole1
    whole0 += whole1 + 0.5
    frac0 += frac1
    whole1 = math.floor(frac0)
    frac1 = frac0 - whole1
    whole0 += whole1
    jd0 = whole0
    jd1 = frac1

    if jd0 < _STATE.start_jd:
        raise ValueError(
            f"IFTE request JD={jd0} before ephemeris start {_STATE.start_jd}"
        )

    irec = int(math.floor((jd0 - _STATE.start_jd) / _STATE.step_jd)) + 2
    if jd0 == _STATE.end_jd:
        irec -= 1
    t0 = (jd0 - (_STATE.start_jd + _STATE.step_jd * (irec - 2)) + jd1) / _STATE.step_jd
    t = (t0, _STATE.step_jd)

    ncoeff = _STATE.reclen // 8
    if irec != _STATE.irec:
        _STATE.f.seek(_STATE.reclen * irec, 0)
        raw = _STATE.f.read(_STATE.reclen)
        if len(raw) < _STATE.reclen:
            raise EOFError("IFTE record read truncated")
        buf = np.frombuffer(raw, dtype="<f8", count=ncoeff).copy()
        if _STATE.swap_endian:
            buf = buf.byteswap()
        _STATE.buf = buf
        _STATE.irec = irec

    if kind == 1:
        return _ifte_interp(
            _STATE.iinfo,
            _STATE.buf[_STATE.ipt[0, 0] - 1 :],
            t,
            _STATE.ipt[0, 1],
            1,
            _STATE.ipt[0, 2],
            2,
        )
    return _ifte_interp(
        _STATE.iinfo,
        _STATE.buf[_STATE.ipt[1, 0] - 1 :],
        t,
        _STATE.ipt[1, 1],
        3,
        _STATE.ipt[1, 2],
        2,
    )


def ifte_delta_t(jd0: float, jd1: float) -> float:
    """``IFTE_DeltaT`` in days (tempo2 ``IFTE_DeltaT`` native units)."""
    return float(_ifte_get_vals(jd0, jd1, 1)[0])


def ifte_delta_t_sec(mjd_tt: float) -> float:
    """``IF_deltaT`` from ``tt2tdb.C`` — IFTE offset in seconds."""
    jd0 = 2400000.0 + math.floor(mjd_tt)
    jd1 = 0.5 + (mjd_tt - math.floor(mjd_tt))
    return ifte_delta_t(jd0, jd1) * SECS_PER_DAY


def ifte_delta_t_mjd(mjd_tt: float | np.ndarray) -> np.ndarray:
    """Vectorized ``IF_deltaT(mjd_tt)`` from ``tt2tdb.C`` (seconds)."""
    mjd = np.asarray(mjd_tt, dtype=np.float64)
    out = np.empty_like(mjd)
    flat = mjd.ravel()
    for i, m in enumerate(flat):
        out.ravel()[i] = ifte_delta_t_sec(float(m))
    return out.reshape(mjd.shape)


def ifte_delta_t_dot(jd0: float, jd1: float) -> float:
    return float(_ifte_get_vals(jd0, jd1, 1)[1])
