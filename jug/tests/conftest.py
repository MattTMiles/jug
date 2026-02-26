"""Shared fixtures for JUG test suite.

Provides cached TimingSession instances and path helpers to avoid
redundant file parsing across tests.
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Force determinism BEFORE any JAX imports
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
JUG_ROOT = Path(__file__).resolve().parent.parent.parent  # .../JUG
DATA_PULSARS = JUG_ROOT / "data" / "pulsars"
DATA_GOLDEN = JUG_ROOT / "tests" / "data_golden"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _pulsar_paths(name: str) -> tuple:
    """Return (par, tim) paths for a pulsar under data/pulsars/."""
    par = DATA_PULSARS / f"{name}_tdb.par"
    tim = DATA_PULSARS / f"{name}.tim"
    return par, tim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def j1909_session():
    """A TimingSession for J1909-3744 (ELL1, 10 408 TOAs).

    Cached at session scope -- reusable across all tests.
    """
    from jug.engine.session import TimingSession

    par, tim = _pulsar_paths("J1909-3744")
    if not par.exists() or not tim.exists():
        pytest.skip("J1909-3744 data not found")
    return TimingSession(par, tim, verbose=False)


@pytest.fixture(scope="session")
def j0614_session():
    """A TimingSession for J0614-3329 (DD, 10 099 TOAs)."""
    from jug.engine.session import TimingSession

    par, tim = _pulsar_paths("J0614-3329")
    if not par.exists() or not tim.exists():
        pytest.skip("J0614-3329 data not found")
    return TimingSession(par, tim, verbose=False)


@pytest.fixture(scope="session")
def j1909_prefit(j1909_session):
    """Pre-fit residuals result dict for J1909-3744 (subtract_tzr=False)."""
    return j1909_session.compute_residuals(subtract_tzr=False)


@pytest.fixture(scope="session")
def j0614_prefit(j0614_session):
    """Pre-fit residuals result dict for J0614-3329 (subtract_tzr=False)."""
    return j0614_session.compute_residuals(subtract_tzr=False)


@pytest.fixture(scope="session")
def j1909_params(j1909_session):
    """Parameter dict for J1909-3744."""
    return dict(j1909_session.params)


@pytest.fixture(scope="session")
def j0614_params(j0614_session):
    """Parameter dict for J0614-3329."""
    return dict(j0614_session.params)
