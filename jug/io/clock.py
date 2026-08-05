"""Clock correction file handling and interpolation.

This module provides functions to parse and interpolate tempo2-style clock
correction files.  The core clock-chain logic uses a graph-based approach
matching Tempo2's design: each ``.clk`` file declares its ``FROM`` and ``TO``
timescales in the first comment line (e.g. ``# UTC(AO) UTC(GPS)``), and
:class:`ClockGraph` builds a directed graph over all available files and uses
Dijkstra's algorithm to find the shortest correction path from any
``UTC(obs)`` to ``UTC``, exactly as Tempo2 does.
"""

from functools import lru_cache
from pathlib import Path
from bisect import bisect_left
import heapq
import os
import sys
import warnings
import numpy as np


def resolve_clock_dir(
    clock_dir: Path | str | None = None,
    *,
    compatibility: str | None = None,
) -> Path:
    """Resolve the clock file directory for PINT-family chain discovery."""
    if clock_dir is not None:
        return Path(clock_dir)
    module_dir = Path(__file__).resolve().parent
    return module_dir.parent.parent / "data" / "clock"


# ---------------------------------------------------------------------------
# Leap-second-aware MJD scale conversion
# ---------------------------------------------------------------------------
# Clock-correction files are tabulated against UTC MJD stamps (typically integer
# day stamps). On a UTC day that contains a leap second, the day is 86401 s
# rather than 86400 s long; a query MJD that falls inside such a day is not
# linearly equidistant in physical seconds from the surrounding integer-day
# clock entries unless we account for the inserted second.
#
# Astropy ``Time`` (used by PINT) handles this transparently via SOFA; JUG
# previously interpolated directly on UTC MJD, giving ~ few-fs disagreement
# with PINT on TOAs that fall on leap-second days.
#
# Fix: map both query and clock-file MJDs onto a continuous (TAI-like) scale by
# adding cumulative leap-second offsets, then interpolate on that scale.  The
# absolute zero-point cancels because the same transformation is applied to
# both sides, so we omit the TAI-UTC base offset (10 s at MJD 41499) and just
# add the cumulative count of leap seconds inserted at or before each MJD.

# UTC MJD at the start of each day following a leap-second insertion.  For
# every entry M, all UTC MJDs >= M have accumulated one additional leap second
# relative to MJDs < M.  Table covers leaps from 1972-06-30 through 2016-12-31.
_LEAP_INSERTION_MJDS = np.array([
    41499, 41683, 42048, 42413, 42778, 43144, 43509, 43874, 44239,
    44786, 45151, 45516, 46247, 47161, 47892, 48257, 48804, 49169,
    49534, 50083, 50630, 51179, 53736, 54832, 56109, 57204, 57754,
], dtype=np.int64)


def utc_mjd_to_continuous(mjd):
    """Map UTC MJD to a continuous (leap-second-aware) time coordinate.

    Adds the cumulative count of leap seconds inserted at or before ``mjd``,
    expressed in days, to convert from a UTC abscissa (where leap-second days
    are 86401 s long) to a continuous abscissa suitable for linear
    interpolation against clock-file MJDs treated the same way.

    The absolute offset cancels when both query and clock-file MJDs are
    converted; only the relative leap count between them matters.

    Parameters
    ----------
    mjd : float or np.ndarray
        UTC MJD value(s).

    Returns
    -------
    Same shape as input; continuous-MJD scale.
    """
    mjd_arr = np.asarray(mjd, dtype=np.float64)
    n_leaps = np.searchsorted(_LEAP_INSERTION_MJDS, mjd_arr, side='right')
    return mjd_arr + n_leaps / 86400.0


# ---------------------------------------------------------------------------
# Leap-second-aware MJD scale conversion
# ---------------------------------------------------------------------------
# Clock-correction files are tabulated against UTC MJD stamps (typically integer
# day stamps). On a UTC day that contains a leap second, the day is 86401 s
# rather than 86400 s long; a query MJD that falls inside such a day is not
# linearly equidistant in physical seconds from the surrounding integer-day
# clock entries unless we account for the inserted second.
#
# Astropy ``Time`` (used by PINT) handles this transparently via SOFA; JUG
# previously interpolated directly on UTC MJD, giving ~ few-fs disagreement
# with PINT on TOAs that fall on leap-second days.
#
# Fix: map both query and clock-file MJDs onto a continuous (TAI-like) scale by
# adding cumulative leap-second offsets, then interpolate on that scale.  The
# absolute zero-point cancels because the same transformation is applied to
# both sides, so we omit the TAI-UTC base offset (10 s at MJD 41499) and just
# add the cumulative count of leap seconds inserted at or before each MJD.

# UTC MJD at the start of each day following a leap-second insertion.  For
# every entry M, all UTC MJDs >= M have accumulated one additional leap second
# relative to MJDs < M.  Table covers leaps from 1972-06-30 through 2016-12-31.
_LEAP_INSERTION_MJDS = np.array([
    41499, 41683, 42048, 42413, 42778, 43144, 43509, 43874, 44239,
    44786, 45151, 45516, 46247, 47161, 47892, 48257, 48804, 49169,
    49534, 50083, 50630, 51179, 53736, 54832, 56109, 57204, 57754,
], dtype=np.int64)


def utc_mjd_to_continuous(mjd):
    """Map UTC MJD to a continuous (leap-second-aware) time coordinate.

    Adds the cumulative count of leap seconds inserted at or before ``mjd``,
    expressed in days, to convert from a UTC abscissa (where leap-second days
    are 86401 s long) to a continuous abscissa suitable for linear
    interpolation against clock-file MJDs treated the same way.

    The absolute offset cancels when both query and clock-file MJDs are
    converted; only the relative leap count between them matters.

    Parameters
    ----------
    mjd : float or np.ndarray
        UTC MJD value(s).

    Returns
    -------
    Same shape as input; continuous-MJD scale.
    """
    mjd_arr = np.asarray(mjd, dtype=np.float64)
    n_leaps = np.searchsorted(_LEAP_INSERTION_MJDS, mjd_arr, side='right')
    return mjd_arr + n_leaps / 86400.0


# ---------------------------------------------------------------------------
# Graph-based clock chain (Tempo2-style Dijkstra path finding)
# ---------------------------------------------------------------------------

def _read_clock_header(path) -> tuple[str, str, int] | None:
    """Read the FROM/TO timescale pair (and optional weight) from the first
    comment line of a clock file.

    Tempo2 clock files begin with a line like::

        # UTC(AO) UTC(GPS)
        # UTC(PKS) UTC(AUS) 100   ← high weight = avoid this path

    The optional third field is the Tempo2 hop weight (default 1 if absent).
    Higher weight means Dijkstra prefers other paths.

    Returns ``(from_scale, to_scale, weight)`` both timescales upper-cased,
    or ``None`` if the header cannot be parsed.

    Timescale normalization: ``UTC(USNO)`` is collapsed to ``UTC``. The IPTA/
    Tempo2 ``gps2utc.clk`` declares its target as the USNO realization of UTC
    (``# UTC(GPS) UTC(USNO)``); without collapsing it, JUG's Dijkstra cannot
    terminate on ``gps2utc.clk`` and reroutes GPS->UTC observatories onto a
    *different* UTC realization (e.g. VLA via ``vla2nist.clk -> nist2utc.clk``),
    which disagrees with Tempo2/PINT by ~µs. USNO is the canonical UTC
    realization the GPS chain targets, so it IS ``UTC`` for routing purposes.
    (A pure data refresh that only relabels this header would otherwise silently
    flip the clock chain -- the cause of the 2026-06-22 NG-GBT/VLA regression.)
    """
    def _norm(scale: str) -> str:
        return "UTC" if scale == "UTC(USNO)" else scale
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    parts = line.lstrip('#').split()
                    if len(parts) >= 2:
                        weight = 1
                        if len(parts) >= 3:
                            try:
                                weight = int(parts[2])
                            except ValueError:
                                pass
                        return _norm(parts[0].upper()), _norm(parts[1].upper()), weight
                    return None
    except OSError:
        pass
    return None


class ClockGraph:
    """Directed graph of Tempo2-style clock correction files.

    Scans a directory for ``*.clk`` files, reads their ``# FROM TO`` headers,
    and builds a directed graph where each edge is one clock file.  The
    :meth:`correction_chain` method uses Dijkstra's algorithm (matching
    Tempo2's ``getClockCorrectionSequence``) to find the shortest path from
    ``UTC(obs)`` to a target timescale (default ``UTC``), then merges the
    corrections along that path into a single combined clock dict.

    Parameters
    ----------
    clock_dir : str or Path
        Directory containing ``.clk`` files.
    target : str, optional
        Target timescale to route towards (default ``"UTC"``).
    """

    def __init__(self, clock_dir, target: str = "UTC"):
        self.clock_dir = Path(clock_dir)
        self.target = target.upper()
        self._target_set = {self.target}
        # edges: list of (from_scale, to_scale, path)
        self._edges: list[tuple[str, str, Path]] = []
        self._build()
        # Lazily-filled per-edge (start, end) MJD coverage; the edge/file set
        # is fixed once _build() has run, so no invalidation is needed.
        self._coverage: list[tuple[float, float] | None] = [None] * len(self._edges)
        # correction_chain results per (src, mjd_min, mjd_max); values are
        # template dicts — correction_chain returns shallow copies so caller
        # key-additions cannot poison the cache.
        self._chain_cache: dict[tuple, dict | None] = {}

    def _build(self):
        """Scan the clock directory and build the edge list."""
        for clk_file in sorted(self.clock_dir.glob("*.clk")):
            header = _read_clock_header(clk_file)
            if header is None:
                continue
            from_scale, to_scale, weight = header
            # Skip files whose FROM == TO (no-op) and TAI/TT/UT1 terminals
            # that aren't on the path to UTC.
            self._edges.append((from_scale, to_scale, clk_file, weight))

    def _edge_coverage(self, edge_idx: int) -> tuple[float, float]:
        """(start, end) MJD covered by the clock file of *edge_idx* (memoized)."""
        cov = self._coverage[edge_idx]
        if cov is None:
            path = self._edges[edge_idx][2]
            st = os.stat(path)
            cov = _clock_file_span(str(path), (st.st_mtime_ns, st.st_size))
            self._coverage[edge_idx] = cov
        return cov

    def _shortest_path(self, src: str, mjd: float | None = None) -> list[int] | None:
        """Dijkstra from *src* to the target; returns edge indices or None.

        When *mjd* is given, only edges whose clock file covers that epoch are
        traversable — mirroring tempo2 ``makeClockCorrectionSequence``, which
        rebuilds the chain per epoch from the files valid at that SAT.
        """
        dst = self.target

        # Build adjacency: node → list of (neighbour, edge_index, weight)
        adj: dict[str, list[tuple[str, int, int]]] = {}
        for i, (frm, to, _, weight) in enumerate(self._edges):
            if mjd is not None:
                lo, hi = self._edge_coverage(i)
                if not (lo <= mjd <= hi):
                    continue
            adj.setdefault(frm, []).append((to, i, weight))
            # Edges are directed; Tempo2 also supports reverse traversal when
            # the path can be inverted (additive inverse), but we only support
            # the forward direction here for simplicity and correctness.

        # Dijkstra (weighted: respect Tempo2 hop-weights from clock file headers)
        dist: dict[str, int] = {src: 0}
        prev: dict[str, tuple[str, int] | None] = {src: None}
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, 10**9):
                continue
            if u in self._target_set:
                dst = u  # record which alias was actually reached
                break
            for v, edge_idx, weight in adj.get(u, []):
                nd = d + weight
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    prev[v] = (u, edge_idx)
                    heapq.heappush(heap, (nd, v))

        # If the loop broke early, dst was updated to the reached alias.
        # If the heap exhausted without breaking, fall back to whichever alias
        # was reached with the shortest distance.
        if dst not in dist:
            reached = min(
                (n for n in self._target_set if n in dist),
                key=lambda n: dist[n],
                default=None,
            )
            if reached is None:
                return None  # no path found
            dst = reached

        # Reconstruct path
        edge_indices: list[int] = []
        node = dst
        while prev[node] is not None:
            parent, eidx = prev[node]
            edge_indices.append(eidx)
            node = parent
        edge_indices.reverse()
        return edge_indices

    def correction_chain(self, obs_scale: str,
                         mjd_min: float | None = None,
                         mjd_max: float | None = None) -> dict | None:
        """Return the merged clock correction from ``obs_scale`` to ``self.target``.

        Uses Dijkstra's algorithm over the graph of clock files, choosing the
        path with the lowest total hop weight (matching Tempo2).

        When ``mjd_min``/``mjd_max`` are given, the chain is resolved
        *per epoch* the way tempo2 ``getClockCorrectionSequence`` does: only
        files covering a given epoch are usable, so different MJD ranges may
        route through different files (e.g. UTC(AO) goes via UTC(NIST) before
        MJD 50155 and via UTC(GPS) after).  The result is a piecewise merged
        table spanning ``[mjd_min, mjd_max]``.

        Parameters
        ----------
        obs_scale : str
            Starting timescale, e.g. ``"UTC(meerkat)"`` or ``"UTC(AO)"``.
            Case-insensitive.
        mjd_min, mjd_max : float, optional
            Data MJD range for epoch-aware chain resolution.  When omitted,
            a single epoch-blind chain is used (previous behaviour).

        Returns
        -------
        dict or None
            A merged clock dict ``{'mjd': array, 'offset': array}`` representing
            the sum of corrections along the shortest path, or ``None`` if no
            path exists.  Also sets ``dict['chain']`` to the list of file names
            used, for diagnostic purposes.
        """
        src = obs_scale.upper()

        key = (
            src,
            None if mjd_min is None else float(mjd_min),
            None if mjd_max is None else float(mjd_max),
        )
        if key in self._chain_cache:
            cached = self._chain_cache[key]
            # Shallow copy: arrays are shared, but callers adding/overwriting
            # keys (e.g. 'chain') cannot mutate the cached template.
            return None if cached is None else dict(cached)

        result = self._correction_chain_uncached(src, mjd_min, mjd_max)
        self._chain_cache[key] = result
        return None if result is None else dict(result)

    def _correction_chain_uncached(self, src: str,
                                   mjd_min: float | None,
                                   mjd_max: float | None) -> dict | None:
        if src in self._target_set:
            # Already at target — zero correction
            return {'mjd': np.array([0.0, 1e6]), 'offset': np.array([0.0, 0.0]),
                    'chain': []}

        if mjd_min is None or mjd_max is None:
            edge_indices = self._shortest_path(src)
            if edge_indices is None:
                return None
            chain_files = [self._edges[i][2] for i in edge_indices]
            return self._merge_chain(chain_files)

        return self._correction_chain_piecewise(src, float(mjd_min), float(mjd_max))

    def _correction_chain_piecewise(self, src: str, mjd_min: float,
                                    mjd_max: float) -> dict | None:
        """Epoch-aware chain: re-run Dijkstra on each coverage interval."""
        # Pad the data range slightly so interpolation at the exact endpoints
        # stays inside the table.
        lo_all = mjd_min - 1.0
        hi_all = mjd_max + 1.0

        # Interval breakpoints: coverage boundaries of every file, clipped
        # to the data range.
        cuts = {lo_all, hi_all}
        for i in range(len(self._edges)):
            for b in self._edge_coverage(i):
                if lo_all < b < hi_all:
                    cuts.add(b)
        bounds = sorted(cuts)

        seg_mjd: list[np.ndarray] = []
        seg_off: list[np.ndarray] = []
        chain_names: list[str] = []
        prev_path: tuple[int, ...] | None = None

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mid = 0.5 * (lo + hi)
            edge_indices = self._shortest_path(src, mjd=mid)
            if edge_indices is None:
                # tempo2 CLK4: "Trying assuming UTC = <obs>" → zero correction
                grid = np.array([lo, hi])
                offs = np.zeros(2)
                path_key: tuple[int, ...] = ()
            else:
                files = [self._edges[i][2] for i in edge_indices]
                clocks = [parse_clock_file(f) for f in files]
                grid_pts = [c['mjd'][(c['mjd'] >= lo) & (c['mjd'] <= hi)]
                            for c in clocks]
                grid = np.unique(np.concatenate(grid_pts + [np.array([lo, hi])]))
                grid_cont = utc_mjd_to_continuous(grid)
                offs = np.zeros_like(grid)
                for clk in clocks:
                    # Clamped linear interpolation (files cover the interval by
                    # construction; clamping only matters at exact boundaries,
                    # where interpolate_clock_vectorized would zero the value).
                    offs += np.interp(
                        grid_cont,
                        utc_mjd_to_continuous(clk['mjd']),
                        clk['offset'],
                    )
                path_key = tuple(edge_indices)
                for f in files:
                    if f.name not in chain_names:
                        chain_names.append(f.name)

            if seg_mjd and prev_path != path_key:
                # Chain switch: duplicate the boundary so the merged table has
                # a step there instead of interpolating across the switch.
                eps = 1e-8  # ~0.9 ms on the MJD axis; clock tables vary slowly
                last = seg_mjd[-1]
                if last[-1] >= grid[0]:
                    grid = grid.copy()
                    grid[0] = last[-1] + eps
            elif seg_mjd:
                # Same chain continuing: drop the duplicated boundary sample.
                grid = grid[1:]
                offs = offs[1:]
            seg_mjd.append(grid)
            seg_off.append(offs)
            prev_path = path_key

        if not seg_mjd:
            return None
        mjd_all = np.concatenate(seg_mjd)
        off_all = np.concatenate(seg_off)
        order = np.argsort(mjd_all, kind="stable")
        return {
            'mjd': mjd_all[order],
            'offset': off_all[order],
            'chain': chain_names,
        }

    @staticmethod
    def _merge_chain(files: list[Path]) -> dict:
        """Load and sum clock corrections along a chain of files."""
        from jug.io.clock import interpolate_clock_vectorized  # local import avoids circular

        if not files:
            return {'mjd': np.array([0.0, 1e6]), 'offset': np.array([0.0, 0.0]),
                    'chain': []}

        clocks = [parse_clock_file(f) for f in files]

        if len(clocks) == 1:
            clocks[0]['chain'] = [files[0].name]
            return clocks[0]

        # Merge MJD grids (union), preserving duplicate MJDs for step functions
        mjd_grid = np.sort(np.unique(np.concatenate([c['mjd'] for c in clocks])))
        combined = np.zeros_like(mjd_grid)
        for clk in clocks:
            combined += interpolate_clock_vectorized(clk, mjd_grid)

        return {
            'mjd': mjd_grid,
            'offset': combined,
            'chain': [f.name for f in files],
        }


def _clock_span_line_value(line: str):
    """MJD of *line* if ``_parse_clock_file_cached`` would keep it, else None.

    Must mirror that parser's filter exactly -- comment/blank skip, >=2 fields,
    and BOTH columns numeric. (Only checking column 0 mis-reads e.g.
    pks2aus.clk, whose prose header has a line starting with a bare number.)
    """
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None
    parts = stripped.split()
    if len(parts) < 2:
        return None
    try:
        mjd = float(parts[0])
        float(parts[1])
    except ValueError:
        return None
    return mjd


@lru_cache(maxsize=None)
def _clock_file_span(path_str: str, _stat_key: tuple) -> tuple:
    """(first, last) MJD of a clock file, without parsing the whole thing.

    _edge_coverage needs only two numbers per file, but the piecewise chain
    asks for them for EVERY file in the clock directory in order to place its
    interval breakpoints. Doing that via parse_clock_file reads ~65 MB and
    costs ~450 ms; seeking the head and tail costs ~13 ms for the same answer
    (verified identical across all 77 bundled clock files).

    Files on the selected chain are still parsed in full by the caller -- this
    only avoids parsing the ~75 that merely get their coverage inspected.
    """
    size = os.path.getsize(path_str)
    chunk = 1 << 16
    first = last = None
    with open(path_str, 'rb') as fh:
        pos = 0
        while first is None and pos < size:
            fh.seek(pos)
            lines = fh.read(chunk).decode('utf-8', 'ignore').splitlines()
            if pos + chunk < size and lines:
                lines = lines[:-1]          # drop a possibly-truncated line
            for ln in lines:
                v = _clock_span_line_value(ln)
                if v is not None:
                    first = v
                    break
            pos += chunk
        off = max(0, size - chunk)
        while last is None:
            fh.seek(off)
            lines = fh.read(size - off).decode('utf-8', 'ignore').splitlines()
            if off > 0 and lines:
                lines = lines[1:]           # drop the partial leading line
            for ln in reversed(lines):
                v = _clock_span_line_value(ln)
                if v is not None:
                    last = v
                    break
            if off == 0:
                break
            off = max(0, off - chunk)
    if first is None or last is None:
        return (float('inf'), float('-inf'))
    return (float(first), float(last))


@lru_cache(maxsize=None)
def _parse_clock_file_cached(path_str: str) -> tuple:
    """Internal cached clock file parser.

    Returns ``(mjd_array, offset_array)`` as read-only float64 arrays shared
    by every ``parse_clock_file`` call for the same path. The cache is
    unbounded on purpose: it holds one entry per distinct clock file
    (~dozens), and a bounded LRU smaller than the clock directory thrashes at
    ~100% miss during the ClockGraph Dijkstra sweeps, which visit every file
    per shortest-path call (measured: 18k+ full re-parses per residual
    computation, ~57% of total runtime). File contents are assumed stable for
    the process lifetime — the same semantics ClockGraph already has for the
    edge list.

    Sentinel entries at the end of clock files (e.g. MJD 60000 or 99999
    with offset=0) are kept as-is, matching Tempo2 behaviour: linear
    interpolation between the last real entry and the sentinel is used
    for MJDs that fall in the gap.
    """
    mjds = []
    offsets = []
    path = Path(path_str)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mjd = float(parts[0])
                    offset = float(parts[1])
                    mjds.append(mjd)
                    offsets.append(offset)
                except ValueError:
                    continue

    mjd_arr = np.array(mjds, dtype=np.float64)
    offset_arr = np.array(offsets, dtype=np.float64)
    # Shared across callers: fail loudly if anyone tries in-place writes.
    mjd_arr.setflags(write=False)
    offset_arr.setflags(write=False)
    return (mjd_arr, offset_arr)


# str(path) -> resolved absolute path string; avoids a filesystem
# Path.resolve() round-trip on every parse_clock_file call.
_RESOLVED_PATH_CACHE: dict = {}


def parse_clock_file(path: Path | str) -> dict:
    """Parse tempo2-style clock correction file.

    Parameters
    ----------
    path : Path or str
        Path to clock file

    Returns
    -------
    dict
        Dictionary with 'mjd' and 'offset' arrays

    Notes
    -----
    File format: MJD offset(seconds) [optional columns]

    Lines starting with '#' are comments and are skipped.
    
    Results are cached using functools.lru_cache for performance.
    Repeated calls with the same path return cached arrays.

    Examples
    --------
    >>> clock_data = parse_clock_file("data/clock/mk2utc.clk")
    >>> print(f"Clock file has {len(clock_data['mjd'])} entries")
    """
    # Resolve to absolute path string for consistent caching
    key = str(path)
    path_str = _RESOLVED_PATH_CACHE.get(key)
    if path_str is None:
        path_str = str(Path(path).resolve())
        _RESOLVED_PATH_CACHE[key] = path_str

    # Fresh dict per call (callers may add keys, e.g. _merge_chain sets
    # 'chain'); the arrays themselves are shared read-only cache entries.
    mjd_arr, offset_arr = _parse_clock_file_cached(path_str)

    return {
        'mjd': mjd_arr,
        'offset': offset_arr,
        'path': str(path),
    }


def interpolate_clock(clock_data: dict, mjd: float) -> float:
    """Interpolate clock correction at given MJD.

    Uses linear interpolation between adjacent points.

    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays
    mjd : float
        MJD value to interpolate at

    Returns
    -------
    float
        Interpolated clock offset in seconds

    Notes
    -----
    For MJDs outside the clock file range, returns 0 (no correction),
    matching Tempo2's behaviour where the clock chain falls through.

    Examples
    --------
    >>> clock_data = parse_clock_file("mk2utc.clk")
    >>> offset = interpolate_clock(clock_data, 58000.5)
    >>> print(f"Clock correction: {offset:.9f} seconds")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']

    if len(mjds) == 0:
        return 0.0

    # Out-of-range: return 0 (matches Tempo2 behaviour where the clock
    # chain falls through when the file doesn't cover the TOA)
    if mjd <= mjds[0]:
        return 0.0
    if mjd >= mjds[-1]:
        return 0.0

    # Find bracketing points (range-check on raw UTC MJDs; interpolation on
    # leap-second-aware continuous scale below).
    idx = bisect_left(mjds, mjd)
    if idx == 0:
        return offsets[0]

    # Convert bracketing MJDs and query MJD to the continuous scale so the
    # interpolation fraction is correct across leap-second boundaries.
    mjd_cont   = float(utc_mjd_to_continuous(np.asarray([mjd], dtype=float))[0])
    bracket    = utc_mjd_to_continuous(np.asarray([mjds[idx-1], mjds[idx]], dtype=float))
    mjd0, mjd1 = float(bracket[0]), float(bracket[1])
    off0, off1 = offsets[idx-1], offsets[idx]

    frac = (mjd_cont - mjd0) / (mjd1 - mjd0)
    return off0 + frac * (off1 - off0)


def interpolate_clock_vectorized(clock_data: dict, mjd_array: np.ndarray,
                                 clock_name: str = "") -> np.ndarray:
    """Vectorized clock interpolation using np.searchsorted.

    ~10x faster than looping over interpolate_clock() for large arrays.
    Maintains identical accuracy to scalar version.

    For MJDs outside the clock file range, the nearest boundary value
    is returned (constant extrapolation) and a warning is emitted.

    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays (and optional 'path')
    mjd_array : np.ndarray
        Array of MJD values to interpolate
    clock_name : str, optional
        Override name for warning messages (defaults to clock_data['path'])

    Returns
    -------
    np.ndarray
        Interpolated clock offsets in seconds

    Notes
    -----
    This function is optimized for processing many TOAs at once.
    For single values, use interpolate_clock() instead.

    Examples
    --------
    >>> clock_data = parse_clock_file("mk2utc.clk")
    >>> mjds = np.array([58000.0, 58001.0, 58002.0])
    >>> offsets = interpolate_clock_vectorized(clock_data, mjds)
    >>> print(f"Corrections: {offsets}")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']

    # Handle empty clock data
    if len(mjds) == 0:
        return np.zeros_like(mjd_array)

    # Warn if TOAs are outside clock file range (like PINT)
    if len(mjd_array) > 0 and len(mjds) > 1:
        n_before = np.sum(mjd_array < mjds[0])
        n_after = np.sum(mjd_array > mjds[-1])
        label = clock_name or clock_data.get('path', '')
        if label:
            label = Path(label).name
        if n_before > 0:
            warnings.warn(
                f"Clock file '{label}': {n_before} TOA(s) before clock data "
                f"start (MJD {mjds[0]:.1f}); using constant extrapolation",
                stacklevel=2,
            )
        if n_after > 0:
            warnings.warn(
                f"Clock file '{label}': {n_after} TOA(s) after clock data "
                f"end (MJD {mjds[-1]:.1f}); using constant extrapolation",
                stacklevel=2,
            )

    # Find insertion indices (right side gives us the upper bracket).
    # Range-check uses the raw UTC MJDs so the warning thresholds are still
    # human-readable; the actual interpolation is done on the leap-second-aware
    # continuous scale below.
    idx = np.searchsorted(mjds, mjd_array, side='right')

    # Identify out-of-range TOAs (before first or after last entry).
    # Tempo2 drops the clock correction entirely for out-of-range TOAs
    # (the clock chain falls through), so we return 0 for those.
    out_of_range = (idx == 0) | (idx >= len(mjds))

    # Clip to valid range [1, len(mjds)-1] for interpolation
    idx = np.clip(idx, 1, len(mjds) - 1)

    # Convert both clock-file MJDs and query MJDs onto the leap-second-aware
    # continuous scale before computing the interpolation fraction. This makes
    # JUG bit-equivalent to PINT/Astropy on TOAs that fall on leap-second days
    # (otherwise ~5 fs disagreement per leap second between the two scales).
    mjds_cont      = utc_mjd_to_continuous(mjds)
    mjd_array_cont = utc_mjd_to_continuous(mjd_array)

    # Get bracketing points (continuous scale)
    mjd0 = mjds_cont[idx - 1]
    mjd1 = mjds_cont[idx]
    off0 = offsets[idx - 1]
    off1 = offsets[idx]

    # Vectorized linear interpolation
    # Handle edge cases: if mjd0 == mjd1, frac should be 0 (use first offset)
    frac = np.where(mjd1 != mjd0, (mjd_array_cont - mjd0) / (mjd1 - mjd0), 0.0)

    # Clamp frac to [0, 1] for in-range values only
    frac = np.clip(frac, 0.0, 1.0)

    result = off0 + frac * (off1 - off0)

    # Zero out corrections for out-of-range TOAs (matches Tempo2 behaviour)
    result[out_of_range] = 0.0

    return result


def validate_clock_file_coverage(clock_data: dict, mjd_start: float, mjd_end: float, 
                                   file_name: str = "clock file", warn_days: float = 30.0) -> dict:
    """Validate that a clock file covers the required MJD range.
    
    Checks for:
    - Coverage gaps (MJDs outside clock file range)
    - Suspicious constant regions (potential extrapolation)
    - Outdated files (end date too far in the past)
    
    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays
    mjd_start : float
        Start MJD of data requiring coverage
    mjd_end : float
        End MJD of data requiring coverage
    file_name : str, optional
        Name of clock file for warning messages
    warn_days : float, optional
        Warn if file ends more than this many days before mjd_end (default: 30)
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool, True if coverage is adequate
        - 'warnings': list of warning strings
        - 'errors': list of error strings
        - 'coverage_start': MJD where clock file starts
        - 'coverage_end': MJD where clock file ends
        - 'data_start': MJD where data starts
        - 'data_end': MJD where data ends
    
    Examples
    --------
    >>> clock_data = parse_clock_file("tai2tt_bipm2024.clk")
    >>> result = validate_clock_file_coverage(clock_data, 60000.0, 60837.0)
    >>> if not result['valid']:
    ...     for warning in result['warnings']:
    ...         print(f"WARNING: {warning}")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']
    
    warnings = []
    errors = []
    valid = True
    
    if len(mjds) == 0:
        errors.append(f"{file_name}: Clock file is empty")
        return {
            'valid': False,
            'warnings': warnings,
            'errors': errors,
            'coverage_start': None,
            'coverage_end': None,
            'data_start': mjd_start,
            'data_end': mjd_end
        }
    
    coverage_start = mjds[0]
    coverage_end = mjds[-1]
    
    # Check if data is outside clock file range
    if mjd_start < coverage_start:
        errors.append(
            f"{file_name}: Data starts at MJD {mjd_start:.1f} but clock file "
            f"only covers from MJD {coverage_start:.1f} "
            f"({coverage_start - mjd_start:.1f} days before coverage)"
        )
        valid = False
    
    if mjd_end > coverage_end:
        days_past = mjd_end - coverage_end
        if days_past > warn_days:
            errors.append(
                f"{file_name}: Data extends to MJD {mjd_end:.1f} but clock file "
                f"ends at MJD {coverage_end:.1f} "
                f"({days_past:.1f} days of extrapolation). "
                f"Consider updating clock file."
            )
            valid = False
        else:
            warnings.append(
                f"{file_name}: Minor extrapolation ({days_past:.1f} days past clock file end)"
            )
    
    # Check for suspicious constant regions near the end
    # (indicates possible extrapolation in the clock file itself)
    if len(mjds) > 10:
        # Find where real data ends by looking for large gaps or constant regions
        # Check spacing between entries
        mjd_diffs = np.diff(mjds)
        
        # Look for abnormally large gaps (> 100 days suggests jump to extrapolation)
        # Only consider gaps that occur after the data start to avoid false positives
        # from dummy header entries at MJD ~0 in some clock files.
        large_gaps = np.where((mjd_diffs > 100) & (mjds[:-1] > mjd_start - 365))[0]
        if len(large_gaps) > 0:
            # Found a large gap - data before this is real
            real_data_end_idx = large_gaps[0]
            real_data_end = mjds[real_data_end_idx]
            
            if mjd_end > real_data_end:
                errors.append(
                    f"{file_name}: Real data ends at MJD {real_data_end:.1f}, "
                    f"but your data extends to MJD {mjd_end:.1f} "
                    f"({mjd_end - real_data_end:.1f} days using extrapolated values). "
                    f"Clock file has large gap suggesting constant extrapolation. "
                    f"UPDATE CLOCK FILE from IPTA repository!"
                )
                valid = False
        
        # Also check last 10 entries for constant values
        last_offsets = offsets[-10:]
        if np.std(last_offsets) < 1e-12:  # Effectively constant
            # Check if there's variation before the constant region
            if len(mjds) > 20:
                prev_offsets = offsets[-20:-10]
                if np.std(prev_offsets) > 1e-12:  # Previous region was varying
                    warnings.append(
                        f"{file_name}: Last 10 entries are constant at "
                        f"{last_offsets[-1]:.12f} s (possible extrapolation within file)"
                    )
    
    return {
        'valid': valid,
        'warnings': warnings,
        'errors': errors,
        'coverage_start': coverage_start,
        'coverage_end': coverage_end,
        'data_start': mjd_start,
        'data_end': mjd_end
    }


def check_clock_files(mjd_start: float, mjd_end: float,
                      mk_clock: dict, gps_clock: dict, bipm_clock: dict,
                      verbose: bool = True,
                      clock_dir: str = None) -> tuple:
    """Check all clock files for adequate coverage.

    Errors (data outside clock file range) are always printed in bold red,
    regardless of the ``verbose`` flag.  Warnings (minor extrapolation, etc.)
    are printed in bold yellow when ``verbose=True``.

    Parameters
    ----------
    mjd_start : float
        Start MJD of data
    mjd_end : float
        End MJD of data
    mk_clock : dict
        Observatory clock data
    gps_clock : dict
        GPS clock data
    bipm_clock : dict
        BIPM clock data
    verbose : bool, optional
        Print warnings in addition to errors (default: True)
    clock_dir : str, optional
        Path to the clock directory, used in the actionable suggestion message.

    Returns
    -------
    tuple
        ``(valid, issues)`` where *valid* is a bool (True = no hard errors) and
        *issues* is a list of dicts, each with keys ``'severity'`` (``'error'``
        or ``'warning'``) and ``'message'`` (str).

    Examples
    --------
    >>> mk = parse_clock_file("mk2utc.clk")
    >>> gps = parse_clock_file("gps2utc.clk")
    >>> bipm = parse_clock_file("tai2tt_bipm2024.clk")
    >>> ok, issues = check_clock_files(58000.0, 60837.0, mk, gps, bipm)
    """
    _RED   = "\033[1;31m"   # bold red
    _YELLOW = "\033[1;33m"  # bold yellow
    _RESET = "\033[0m"

    all_valid = True
    all_issues = []

    suggestion = (
        f"  -> Copy the correct clock files into {clock_dir}"
        if clock_dir else
        "  -> Copy the correct clock files into your JUG data/clock/ directory"
    )

    for name, clock_data in [
        ("Observatory clock", mk_clock),
        ("GPS clock (gps2utc.clk)", gps_clock),
        ("BIPM clock (tai2tt_bipm*.clk)", bipm_clock),
    ]:
        # Include the actual filename so the user knows which file to update
        filename = Path(clock_data.get('path', '')).name if clock_data.get('path') else ''
        label = f"{name} ({filename})" if filename and filename not in name else name
        result = validate_clock_file_coverage(clock_data, mjd_start, mjd_end, label)

        if not result['valid']:
            all_valid = False

        for error in result['errors']:
            msg = f"CLOCK FILE ERROR: {error}"
            all_issues.append({'severity': 'error', 'message': msg})
            # Always print errors -- they affect timing accuracy
            print(f"{_RED}[!] {msg}{_RESET}")
            print(f"{_RED}{suggestion}{_RESET}")

        for warning in result['warnings']:
            msg = f"CLOCK FILE WARNING: {warning}"
            all_issues.append({'severity': 'warning', 'message': msg})
            if verbose:
                print(f"{_YELLOW}[!]  {msg}{_RESET}")
                print(f"{_YELLOW}{suggestion}{_RESET}")

    return all_valid, all_issues


def compare_clock_files(path_a: Path | str, path_b: Path | str,
                        threshold_us: float = 0.001) -> dict:
    """Compare two clock files and report significant differences.

    Parameters
    ----------
    path_a, path_b : Path or str
        Paths to clock files to compare.
    threshold_us : float, optional
        Difference threshold in microseconds above which to flag (default: 0.001 mus).

    Returns
    -------
    dict
        Keys: ``'max_diff_us'``, ``'mean_diff_us'``, ``'significant'`` (bool),
        ``'a_end_mjd'``, ``'b_end_mjd'``, ``'a_entries'``, ``'b_entries'``,
        ``'summary'`` (human-readable string).
    """
    def _load(p):
        d = parse_clock_file(p)
        return d['mjd'], d['offset']

    mjds_a, off_a = _load(path_a)
    mjds_b, off_b = _load(path_b)

    # Interpolate b onto a's grid within the overlap
    overlap_start = max(mjds_a[0], mjds_b[0])
    overlap_end = min(mjds_a[-1], mjds_b[-1])

    max_diff_us = 0.0
    mean_diff_us = 0.0

    if overlap_end > overlap_start:
        mask = (mjds_a >= overlap_start) & (mjds_a <= overlap_end)
        sample_mjds = mjds_a[mask]
        if len(sample_mjds) > 0:
            interp_b = np.interp(sample_mjds, mjds_b, off_b)
            interp_a = off_a[mask]
            diffs_us = np.abs(interp_a - interp_b) * 1e6
            max_diff_us = float(np.max(diffs_us))
            mean_diff_us = float(np.mean(diffs_us))

    summary = (
        f"{Path(path_a).name}: {len(mjds_a)} entries, ends MJD {mjds_a[-1]:.1f}; "
        f"{Path(path_b).name}: {len(mjds_b)} entries, ends MJD {mjds_b[-1]:.1f}; "
        f"max diff = {max_diff_us:.4f} mus"
    )

    return {
        'max_diff_us': max_diff_us,
        'mean_diff_us': mean_diff_us,
        'significant': max_diff_us > threshold_us,
        'a_end_mjd': float(mjds_a[-1]),
        'b_end_mjd': float(mjds_b[-1]),
        'a_entries': len(mjds_a),
        'b_entries': len(mjds_b),
        'summary': summary,
    }


_IERS_REMEDIATION = (
    'python -c "from astropy.utils.iers import IERS_A; IERS_A.open()" '
    "or bind a host ~/.astropy/cache into the container"
)


def _probe_iers_gcrs_transform(mjd: float) -> None:
    """Smoke-test ITRF→GCRS; raises if Astropy IERS/EOP data is unusable."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    # Fixed geocentric ITRF position (Green Bank approximate); avoids site registry.
    loc = EarthLocation.from_geocentric(
        -849.066 * u.km, -4792.015 * u.km, 3952.036 * u.km
    )
    times = Time([mjd], format="mjd", scale="tdb")
    loc.get_gcrs_posvel(obstime=times)


def iers_strict_enabled(*, iers_policy: str | None = None) -> bool:
    """Return True when IERS preflight should hard-fail (parity/dev), not warn."""
    if iers_policy is not None:
        policy = str(iers_policy).lower()
        if policy == "strict":
            return True
        if policy == "warn":
            return "pytest" in sys.modules
        return False
    return "pytest" in sys.modules


def warn_on_iers_failure(valid: bool, issues: list) -> None:
    """Emit warnings for IERS preflight issues (general fitting / offline use)."""
    if valid:
        return
    messages = [i["message"] for i in issues if i.get("message")]
    if not messages:
        messages = ["IERS/EOP preflight failed."]
    for msg in messages:
        warnings.warn(
            f"{msg} Observatory geometry may be wrong. Try: {_IERS_REMEDIATION}.",
            UserWarning,
            stacklevel=3,
        )


def raise_on_iers_failure(valid: bool, issues: list) -> None:
    """Abort when IERS preflight reports errors."""
    if valid:
        return
    errors = [i["message"] for i in issues if i.get("severity") == "error"]
    detail = errors[0] if errors else "IERS/EOP preflight failed."
    raise RuntimeError(
        f"{detail} Observatory geometry (ITRF→GCRS) requires working Astropy "
        f"IERS data. Try: {_IERS_REMEDIATION}."
    )


def check_iers_coverage(mjd_start: float, mjd_end: float,
                        verbose: bool = True) -> tuple:
    """Check that astropy's IERS Earth-orientation data covers the data MJD range.

    The ITRF->GCRS coordinate transform (used when computing observatory SSB
    positions) relies on IERS UT1-UTC and polar-motion data.  Using predicted
    rather than measured values introduces small but systematic errors.

    After the table-range check, performs a functional ``get_gcrs_posvel``
    probe so missing or corrupt IERS caches fail before geometry computation.

    Parameters
    ----------
    mjd_start, mjd_end : float
        MJD range of TOA data.
    verbose : bool, optional
        Print status even when coverage is fine (default: True).

    Returns
    -------
    tuple
        ``(valid, issues)`` matching the same convention as
        :func:`check_clock_files`.
    """
    _RED    = "\033[1;31m"
    _YELLOW = "\033[1;33m"
    _RESET  = "\033[0m"

    issues = []
    valid = True

    try:
        from astropy.utils import iers as astropy_iers

        tab = astropy_iers.earth_orientation_table.get()
        table_mjds = np.asarray(tab['MJD'])
        if table_mjds.size == 0:
            raise ValueError("IERS table is empty")
        table_end = float(table_mjds[-1])

        # Find end of *measured* (vs predicted) UT1-UTC
        measured_end = table_end
        for col in ('UT1_UTC_B', 'UT1_UTC_A', 'UT1_UTC'):
            if col in tab.colnames:
                vals = np.asarray(tab[col], dtype=float)
                finite_mask = np.isfinite(vals)
                if np.any(finite_mask):
                    measured_end = float(table_mjds[finite_mask][-1])
                break

        table_type = type(tab).__name__

        if mjd_end > table_end:
            days_past = mjd_end - table_end
            msg = (
                f"EOP/IERS ERROR: Data extends to MJD {mjd_end:.1f} but IERS "
                f"table ({table_type}) ends at MJD {table_end:.1f} "
                f"({days_past:.1f} days beyond coverage). "
                f"Coordinate transforms may be wrong."
            )
            issues.append({'severity': 'error', 'message': msg})
            valid = False
            print(f"{_RED}[!] {msg}{_RESET}")
            print(f"{_RED}  -> Run: {_IERS_REMEDIATION}{_RESET}")
        elif mjd_end > measured_end:
            days_predicted = mjd_end - measured_end
            msg = (
                f"EOP/IERS WARNING: Data extends {days_predicted:.1f} days past the "
                f"end of measured IERS data (MJD {measured_end:.1f}). "
                f"Coordinate transforms use predicted EOP values in this range."
            )
            issues.append({'severity': 'warning', 'message': msg})
            if verbose:
                print(f"{_YELLOW}[!]  {msg}{_RESET}")
                print(f"{_YELLOW}  -> Download fresh IERS-A: {_IERS_REMEDIATION}{_RESET}")
        else:
            if verbose:
                print(
                    f"   [x] IERS data ({table_type}) covers MJD {mjd_end:.1f} "
                    f"with measured data to MJD {measured_end:.1f}"
                )

        probe_mjd = 0.5 * (float(mjd_start) + float(mjd_end))
        try:
            _probe_iers_gcrs_transform(probe_mjd)
        except Exception as exc:
            msg = (
                f"EOP/IERS ERROR: ITRF→GCRS transform failed at MJD {probe_mjd:.1f} "
                f"({exc}). Astropy IERS/EOP data may be missing or corrupt."
            )
            issues.append({'severity': 'error', 'message': msg})
            valid = False
            print(f"{_RED}[!] {msg}{_RESET}")
            print(f"{_RED}  -> Run: {_IERS_REMEDIATION}{_RESET}")

    except Exception as e:
        msg = f"EOP/IERS ERROR: Could not check IERS coverage: {e}"
        issues.append({'severity': 'error', 'message': msg})
        valid = False
        print(f"{_RED}[!] {msg}{_RESET}")
        print(f"{_RED}  -> Run: {_IERS_REMEDIATION}{_RESET}")

    return valid, issues
