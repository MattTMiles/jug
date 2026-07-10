"""Deterministic-signal (jug.signals) integration tests.

Covers:
1. Signal-free par: signal_delay_sec is all zeros and residuals are
   unaffected by the wiring (gate works).
2. CHROMEV par: waveform detected, applied to total_delay_sec, exposed in
   the result dict, and consistent between evaluate-only and post-fit paths.
3. Waveform matches a direct independent evaluation (barycentric freqs).
"""
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

JUG_ROOT = os.path.join(os.path.dirname(__file__), "..")
PAR = os.path.join(JUG_ROOT, "data/pulsars/NG_data/NG_15yr_partim/J1738+0333_PINT_20220302.nb.par")
TIM = os.path.join(JUG_ROOT, "data/pulsars/NG_data/NG_15yr_partim/J1738+0333_PINT_20220302.nb.tim")
CLOCK = os.path.join(JUG_ROOT, "data/clock")

CHROMEV = {
    "CHROMEV_EPOCH": 57000.0,
    "CHROMEV_AMP": 2e-6,
    "CHROMEV_TAU": 50.0,
    "CHROMEV_IDX": 2.0,
}

pytestmark = pytest.mark.skipif(
    not (os.path.exists(PAR) and os.path.exists(TIM)),
    reason="J1738+0333 NG15 data not available",
)


@pytest.fixture(scope="module")
def chromev_par(tmp_path_factory):
    d = tmp_path_factory.mktemp("sig")
    par = d / "J1738+0333.par"
    shutil.copy(PAR, par)
    with open(par, "a") as f:
        for k, v in CHROMEV.items():
            f.write(f"{k} {v}\n")
    tim = d / "J1738+0333.tim"
    os.symlink(os.path.abspath(TIM), tim)
    return str(par), str(tim)


def _session(par, tim):
    from jug.engine.session import TimingSession
    return TimingSession(par, tim, clock_dir=CLOCK)


def test_signal_free_par_zero_vector():
    s = _session(PAR, TIM)
    comps = s.compute_residuals(subtract_tzr=True)
    sig = np.asarray(comps["signal_delay_sec"])
    assert sig.shape == np.asarray(comps["residuals_us"]).shape
    assert np.all(sig == 0.0)


def test_chromev_detected_and_applied(chromev_par):
    par, tim = chromev_par
    s = _session(par, tim)
    comps = s.compute_residuals(subtract_tzr=True)
    sig = np.asarray(comps["signal_delay_sec"])
    assert np.count_nonzero(sig) > 0
    # amplitude at 1400 MHz reference, decaying: peak <= amp * (fmin/1400)^-2
    assert 1e-6 < np.max(np.abs(sig)) < 2e-5

    # waveform must match direct evaluation at BARYCENTRIC freqs
    from jug.io.par_reader import parse_par_file
    from jug.signals import detect_signals
    signals = detect_signals(parse_par_file(par))
    assert len(signals) == 1
    direct = signals[0].compute_waveform(
        np.asarray(comps["tdb_mjd"], dtype=float),
        np.asarray(comps["freq_bary_mhz"], dtype=float),
    )
    assert np.allclose(sig, direct, rtol=0, atol=1e-15)


def test_chromev_in_total_delay_and_fit_parity(chromev_par):
    par, tim = chromev_par
    # total delay with signal == total delay without + waveform
    s_clean = _session(PAR, TIM)
    c_clean = s_clean.compute_residuals(subtract_tzr=True)
    s_sig = _session(par, tim)
    c_sig = s_sig.compute_residuals(subtract_tzr=True)
    dtot = (np.asarray(c_sig["total_delay_sec"])
            - np.asarray(c_clean["total_delay_sec"]))
    sig = np.asarray(c_sig["signal_delay_sec"])
    assert np.allclose(dtot, sig, rtol=0, atol=1e-12)

    # evaluate-vs-fit consistency: signal survives the fit unchanged and the
    # fit converges to a sane WRMS (DMX absorbs the idx-2 event on clean data)
    s_sig.fit_parameters(max_iter=5)
    c_post = s_sig.compute_residuals(subtract_tzr=True, force_recompute=True)
    assert np.allclose(np.asarray(c_post["signal_delay_sec"]), sig,
                       rtol=0, atol=1e-15)
    assert c_post["weighted_rms_us"] < 2.0


def test_chromev_params_recognized_no_warning(chromev_par, capsys):
    par, tim = chromev_par
    from jug.io.par_reader import parse_par_file
    from jug.residuals.simple_calculator import _warn_unrecognized_params
    unknown = _warn_unrecognized_params(parse_par_file(par), verbose=False)
    assert not any(k.startswith("CHROMEV_") for k in unknown)
