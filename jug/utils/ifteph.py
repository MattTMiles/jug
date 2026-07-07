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
    # ``ifteph.C`` IFTEinterp: ``modf(t[0], &dt1)`` then ``tc = 2*(modf(temp,&temp1)+dt1)-1``
    frac_t0, int_t0 = math.modf(t[0])
    temp = dna * t[0]
    temp_frac, temp_int = math.modf(temp)
    l = int(temp - int_t0)
    tc = 2.0 * (temp_frac + int_t0) - 1.0

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


@dataclass(frozen=True)
class IFTECoeffTables:
    """Static IFTE coefficient records for host and JAX evaluation."""

    records: np.ndarray
    start_jd: float
    end_jd: float
    step_jd: float
    ipt: np.ndarray
    coef_offset: int
    ncf: int
    na: int


def load_ifte_coeff_tables(path: str | Path | None = None) -> IFTECoeffTables:
    """Load all IFTE Chebyshev records (``ifteph.C`` record 2..N)."""
    ifte_init(path)
    st = _STATE
    ncoeff = st.reclen // 8
    irec_min = 2
    irec_max = int(math.floor((st.end_jd - st.start_jd) / st.step_jd)) + 1
    records = np.empty((irec_max - irec_min + 1, ncoeff), dtype=np.float64)
    for idx, irec in enumerate(range(irec_min, irec_max + 1)):
        st.f.seek(st.reclen * irec, 0)
        raw = st.f.read(st.reclen)
        if len(raw) < st.reclen:
            raise EOFError("IFTE record read truncated")
        buf = np.frombuffer(raw, dtype="<f8", count=ncoeff).copy()
        if st.swap_endian:
            buf = buf.byteswap()
        records[idx] = buf
    return IFTECoeffTables(
        records=records,
        start_jd=st.start_jd,
        end_jd=st.end_jd,
        step_jd=st.step_jd,
        ipt=st.ipt.copy(),
        coef_offset=int(st.ipt[0, 0] - 1),
        ncf=int(st.ipt[0, 1]),
        na=int(st.ipt[0, 2]),
    )


def pack_ifte_tables_jax(tables: IFTECoeffTables):
    """Pack ``IFTECoeffTables`` into JAX arrays."""
    import jax.numpy as jnp

    return {
        "ifte_records": jnp.asarray(tables.records, dtype=jnp.float64),
        "ifte_start_jd": jnp.asarray(tables.start_jd, dtype=jnp.float64),
        "ifte_end_jd": jnp.asarray(tables.end_jd, dtype=jnp.float64),
        "ifte_step_jd": jnp.asarray(tables.step_jd, dtype=jnp.float64),
        "ifte_ipt": jnp.asarray(tables.ipt, dtype=jnp.int32),
    }


def _ifte_combine_jd_parts_jax(jd0, jd1):
    """``IFTE_get_Vals`` JD splitting (``ifteph.C`` lines 226–237)."""
    import jax.numpy as jnp

    whole0 = jnp.floor(jd0 - 0.5)
    frac0 = jd0 - 0.5 - whole0
    whole1 = jnp.floor(jd1)
    frac1 = jd1 - whole1
    whole0 = whole0 + whole1 + 0.5
    frac0 = frac0 + frac1
    whole1 = jnp.floor(frac0)
    frac1 = frac0 - whole1
    whole0 = whole0 + whole1
    return whole0, frac1


def _ifte_interp_pos_jax(coef, t_frac, ncf, na):
    """Position-only ``IFTEinterp`` (``ifteph.C`` lines 400–444)."""
    import jax
    import jax.numpy as jnp

    dna = jnp.float64(na)
    int_t0 = jnp.floor(t_frac)
    temp = dna * t_frac
    int_temp = jnp.floor(temp)
    frac_temp = temp - int_temp
    l = (int_temp - int_t0).astype(jnp.int32)
    tc = 2.0 * (frac_temp + int_t0) - 1.0

    twot = tc + tc
    pc = jnp.zeros(ncf, dtype=jnp.float64)
    pc = pc.at[0].set(1.0)
    pc = pc.at[1].set(tc)

    def cheb_step(i, arr):
        return arr.at[i].set(twot * arr[i - 1] - arr[i - 2])

    pc = jax.lax.fori_loop(2, ncf, cheb_step, pc)
    k = jnp.arange(ncf, dtype=jnp.int32)
    cidx = ncf * (l + 1) - 1 - k
    return jnp.sum(pc[ncf - 1 - k] * coef[cidx])


def _ifte_delta_t_days_jax(jd0, jd1, records, start_jd, end_jd, step_jd, coef_offset, ncf, na):
    """``IFTE_get_Vals`` + ``IFTE_DeltaT`` in days (``ifteph.C`` / ``tt2tdb.C``)."""
    import jax.numpy as jnp

    jd0, jd1 = _ifte_combine_jd_parts_jax(jd0, jd1)
    irec = jnp.floor((jd0 - start_jd) / step_jd).astype(jnp.int32) + 2
    irec = jnp.where(jd0 == end_jd, irec - 1, irec)
    t_frac = (jd0 - (start_jd + step_jd * (irec - 2)) + jd1) / step_jd
    buf = records[irec - 2]
    coef = buf[coef_offset:]
    return _ifte_interp_pos_jax(coef, t_frac, ncf, na)


def ifte_delta_t_sec_jax(
    mjd_tt,
    *,
    ifte_records,
    ifte_start_jd,
    ifte_end_jd,
    ifte_step_jd,
    ifte_coef_offset: int,
    ifte_ncf: int,
    ifte_na: int,
):
    """``IF_deltaT(mjd_tt)`` inside JAX (``tt2tdb.C`` wrapper × ``SECS_PER_DAY``)."""
    import jax
    import jax.numpy as jnp

    mjd = jnp.asarray(mjd_tt, dtype=jnp.float64)

    def one(m):
        jd0 = 2400000.0 + jnp.floor(m)
        jd1 = 0.5 + (m - jnp.floor(m))
        delta_days = _ifte_delta_t_days_jax(
            jd0,
            jd1,
            ifte_records,
            ifte_start_jd,
            ifte_end_jd,
            ifte_step_jd,
            ifte_coef_offset,
            ifte_ncf,
            ifte_na,
        )
        return delta_days * SECS_PER_DAY

    if mjd.ndim == 0:
        return one(mjd)
    return jax.vmap(one)(mjd)


def ifte_delta_t_dot(jd0: float, jd1: float) -> float:
    return float(_ifte_get_vals(jd0, jd1, 1)[1])
