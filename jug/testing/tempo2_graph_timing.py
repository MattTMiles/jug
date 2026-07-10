"""Benchmark tempo2-native JAX graph modes (pack build + JIT compile/eval).

Used by ``tools/run_tempo2_graph_timing_wsrt167.py`` and tempo2 timing tests.
Records wall times only — no hard parity thresholds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from jug.fitting.jax_residual_delta import (
    _prepare_residual_delta_jax,
    compute_autodiff_designmatrix_from_setup,
    make_residual_delta_jax_fn,
)
from jug.fitting.optimized_fitter import GeneralFitSetup
from jug.residuals.tempo2.delta_pack import build_delta_pack_for_setup
from jug.residuals.tempo2.graph_config import (
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
    TEMPO2_GRAPH_STAGED_BCLT,
)
from jug.utils.jax_setup import ensure_jax_x64

DEFAULT_GRAPH_MODES = (
    TEMPO2_GRAPH_STAGED_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
)


@dataclass
class Tempo2GraphModeTiming:
    """Wall-clock timings for one graph mode (seconds)."""

    mode: str
    n_toa: int
    n_fit_params: int
    pack_build_sec: float
    prepare_first_sec: float
    residual_first_jit_sec: float
    residual_warm_sec: float
    jac_first_jit_sec: float
    jac_warm_sec: float
    wls_autodiff_path_sec: float
    prepare_cache_hit_sec: float
    pack_build_calls_wls_path: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_toa": self.n_toa,
            "n_fit_params": self.n_fit_params,
            "pack_build_sec": self.pack_build_sec,
            "prepare_first_sec": self.prepare_first_sec,
            "residual_first_jit_sec": self.residual_first_jit_sec,
            "residual_warm_sec": self.residual_warm_sec,
            "jac_first_jit_sec": self.jac_first_jit_sec,
            "jac_warm_sec": self.jac_warm_sec,
            "wls_autodiff_path_sec": self.wls_autodiff_path_sec,
            "prepare_cache_hit_sec": self.prepare_cache_hit_sec,
            "pack_build_calls_wls_path": self.pack_build_calls_wls_path,
            "total_first_jit_sec": (
                self.residual_first_jit_sec + self.jac_first_jit_sec
            ),
        }


@dataclass
class Tempo2GraphTimingReport:
    """Full benchmark report with speedup ratios vs a reference mode."""

    fixture_id: str
    fit_params: tuple[str, ...]
    host_residuals_sec: float
    setup_from_cache_sec: float
    modes: dict[str, Tempo2GraphModeTiming] = field(default_factory=dict)

    @property
    def setup_build_sec(self) -> float:
        """Legacy alias: host + cache setup (end-to-end from cold host)."""
        return self.host_residuals_sec + self.setup_from_cache_sec

    def speedup_vs(self, reference: str, target: str) -> dict[str, float]:
        ref = self.modes[reference]
        tgt = self.modes[target]
        ratios: dict[str, float] = {}
        for key in (
            "pack_build_sec",
            "residual_first_jit_sec",
            "jac_first_jit_sec",
            "total_first_jit_sec",
            "wls_autodiff_path_sec",
        ):
            r = getattr(ref, key, None) if hasattr(ref, key) else ref.as_dict().get(key)
            t = getattr(tgt, key, None) if hasattr(tgt, key) else tgt.as_dict().get(key)
            if key == "total_first_jit_sec":
                r = ref.residual_first_jit_sec + ref.jac_first_jit_sec
                t = tgt.residual_first_jit_sec + tgt.jac_first_jit_sec
            if r and t and t > 0:
                ratios[key] = float(r) / float(t)
        return ratios

    def summary_lines(self, *, reference: str = TEMPO2_GRAPH_FIXED_STATE_BCLT) -> list[str]:
        lines = [
            f"fixture={self.fixture_id} fit_params={list(self.fit_params)} "
            f"host_residuals_sec={self.host_residuals_sec:.3f} "
            f"setup_from_cache_sec={self.setup_from_cache_sec:.3f}",
            f"{'mode':<22} {'pack':>8} {'res_jit':>8} {'jac_jit':>8} "
            f"{'sum_jit':>8} {'wls_path':>8} {'prep_hit':>8}",
        ]
        for mode in DEFAULT_GRAPH_MODES:
            if mode not in self.modes:
                continue
            m = self.modes[mode]
            total = m.residual_first_jit_sec + m.jac_first_jit_sec
            lines.append(
                f"{mode:<22} {m.pack_build_sec:8.3f} {m.residual_first_jit_sec:8.3f} "
                f"{m.jac_first_jit_sec:8.3f} {total:8.3f} {m.wls_autodiff_path_sec:8.3f} "
                f"{m.prepare_cache_hit_sec:8.3f}"
            )
        if (
            reference in self.modes
            and TEMPO2_GRAPH_FIXED_STATE_STRIPPED in self.modes
        ):
            sp = self.speedup_vs(reference, TEMPO2_GRAPH_FIXED_STATE_STRIPPED)
            lines.append(
                f"stripped/fixed_state_bclt speedup: "
                f"pack={sp.get('pack_build_sec', 0):.2f}x "
                f"res_jit={sp.get('residual_first_jit_sec', 0):.2f}x "
                f"jac_jit={sp.get('jac_first_jit_sec', 0):.2f}x "
                f"sum_jit={sp.get('total_first_jit_sec', 0):.2f}x "
                f"wls={sp.get('wls_autodiff_path_sec', 0):.2f}x"
            )
        return lines

    def to_json(self) -> str:
        payload = {
            "fixture_id": self.fixture_id,
            "fit_params": list(self.fit_params),
            "host_residuals_sec": self.host_residuals_sec,
            "setup_from_cache_sec": self.setup_from_cache_sec,
            "setup_build_sec": self.setup_build_sec,
            "modes": {k: v.as_dict() for k, v in self.modes.items()},
        }
        if (
            TEMPO2_GRAPH_FIXED_STATE_BCLT in self.modes
            and TEMPO2_GRAPH_FIXED_STATE_STRIPPED in self.modes
        ):
            payload["speedup_stripped_vs_fixed_state_bclt"] = self.speedup_vs(
                TEMPO2_GRAPH_FIXED_STATE_BCLT,
                TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
            )
        if (
            TEMPO2_GRAPH_STAGED_BCLT in self.modes
            and TEMPO2_GRAPH_FIXED_STATE_STRIPPED in self.modes
        ):
            payload["speedup_stripped_vs_staged_bclt"] = self.speedup_vs(
                TEMPO2_GRAPH_STAGED_BCLT,
                TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
            )
        return json.dumps(payload, indent=2, sort_keys=True)


def _clear_jax_caches() -> None:
    jax.clear_caches()


def _timed_block_until_ready(fn, arg) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(arg))
    return time.perf_counter() - t0


def benchmark_graph_mode_on_setup(
    setup: GeneralFitSetup,
    fit_params: Sequence[str],
    *,
    mode: str,
    clear_jax_before: bool = True,
) -> Tempo2GraphModeTiming:
    """Benchmark one graph mode on a pre-built ``GeneralFitSetup``."""
    fit_params = tuple(str(p).upper() for p in fit_params)
    setup.tempo2_native = mode
    setup.residual_delta_jax_cache = None

    if clear_jax_before:
        _clear_jax_caches()

    t0 = time.perf_counter()
    pack = build_delta_pack_for_setup(setup)
    pack_build_sec = time.perf_counter() - t0
    if pack is None:
        raise RuntimeError(f"build_delta_pack_for_setup returned None for mode={mode!r}")

    t1 = time.perf_counter()
    _, residual_fn, jac_fn = _prepare_residual_delta_jax(
        setup=setup, fit_params=fit_params
    )
    prepare_first_sec = time.perf_counter() - t1

    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    residual_first_jit_sec = _timed_block_until_ready(residual_fn, zero)
    residual_warm_sec = _timed_block_until_ready(residual_fn, zero)
    jac_first_jit_sec = _timed_block_until_ready(jac_fn, zero)
    jac_warm_sec = _timed_block_until_ready(jac_fn, zero)

    setup.residual_delta_jax_cache = None
    pack_calls = 0
    original = build_delta_pack_for_setup

    def counting_pack(s):
        nonlocal pack_calls
        pack_calls += 1
        return original(s)

    import jug.fitting.jax_residual_delta as jrd

    t_wls = time.perf_counter()
    with patch.object(jrd, "build_delta_pack_for_setup", side_effect=counting_pack):
        make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
        compute_autodiff_designmatrix_from_setup(setup=setup, fit_params=fit_params)
    wls_autodiff_path_sec = time.perf_counter() - t_wls

    t_cache = time.perf_counter()
    _prepare_residual_delta_jax(setup=setup, fit_params=fit_params)
    prepare_cache_hit_sec = time.perf_counter() - t_cache

    return Tempo2GraphModeTiming(
        mode=mode,
        n_toa=len(np.asarray(setup.tdb_mjd)),
        n_fit_params=len(fit_params),
        pack_build_sec=pack_build_sec,
        prepare_first_sec=prepare_first_sec,
        residual_first_jit_sec=residual_first_jit_sec,
        residual_warm_sec=residual_warm_sec,
        jac_first_jit_sec=jac_first_jit_sec,
        jac_warm_sec=jac_warm_sec,
        wls_autodiff_path_sec=wls_autodiff_path_sec,
        prepare_cache_hit_sec=prepare_cache_hit_sec,
        pack_build_calls_wls_path=pack_calls,
    )


def benchmark_wsrt167_graph_modes(
    par_path: Path,
    tim_path: Path,
    fit_params: Sequence[str],
    *,
    fixture_id: str = "wsrt167",
    modes: Sequence[str] = DEFAULT_GRAPH_MODES,
    clear_jax_between_modes: bool = True,
    jug_result: dict | None = None,
    params: dict | None = None,
    toas: list | None = None,
) -> Tempo2GraphTimingReport:
    """Benchmark each graph mode; amortize host residuals when ``jug_result`` given."""
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.residuals.simple_calculator import compute_residuals_simple

    ensure_jax_x64()
    fit_params = tuple(str(p).upper() for p in fit_params)

    if params is None:
        params = parse_par_file(par_path)
    if toas is None:
        toas = parse_tim_file_mjds(tim_path)

    if jug_result is None:
        t_host = time.perf_counter()
        jug_result = compute_residuals_simple(
            par_path,
            tim_path,
            verbose=False,
            compatibility="tempo2",
        )
        host_residuals_sec = time.perf_counter() - t_host
    else:
        host_residuals_sec = 0.0

    import sys

    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from tempo2_test_helpers import (
        build_fit_setup_from_jug_cache,
        session_cached_data_from_jug,
    )

    cached = session_cached_data_from_jug(jug_result, toas)
    t_setup = time.perf_counter()
    setup = build_fit_setup_from_jug_cache(
        params=params,
        session_cached_data=cached,
        fit_params=list(fit_params),
        tempo2_native=TEMPO2_GRAPH_STAGED_BCLT,
    )
    setup_from_cache_sec = time.perf_counter() - t_setup

    report = Tempo2GraphTimingReport(
        fixture_id=fixture_id,
        fit_params=fit_params,
        host_residuals_sec=host_residuals_sec,
        setup_from_cache_sec=setup_from_cache_sec,
    )
    for mode in modes:
        report.modes[mode] = benchmark_graph_mode_on_setup(
            setup,
            fit_params,
            mode=mode,
            clear_jax_before=clear_jax_between_modes,
        )
    return report


def write_timing_report(path: Path, report: Tempo2GraphTimingReport) -> None:
    path.write_text(report.to_json() + "\n")