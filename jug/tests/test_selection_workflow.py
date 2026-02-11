"""Tests for jug.engine.selection and jug.engine.session_workflow."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jug.engine.selection import (
    SelectionState,
    AveragedTOA,
    epoch_average,
)
from jug.engine.session_workflow import (
    HistoryEntry,
    SessionWorkflow,
)


# ===================================================================
# SelectionState
# ===================================================================

class TestSelectionState:
    def test_initial_state(self):
        s = SelectionState(n_toas=10)
        assert s.n_active == 10
        assert s.n_deleted == 0
        assert s.n_selected == 0

    def test_delete_indices(self):
        s = SelectionState(n_toas=10)
        s.delete_indices([2, 5])
        assert s.n_deleted == 2
        assert s.n_active == 8
        assert s.deleted[2] and s.deleted[5]

    def test_undelete(self):
        s = SelectionState(n_toas=5)
        s.delete_indices([0, 1, 2])
        s.undelete_indices([1])
        assert s.n_deleted == 2

    def test_undelete_all(self):
        s = SelectionState(n_toas=5)
        s.delete_indices([0, 1, 2])
        s.undelete_all()
        assert s.n_deleted == 0

    def test_delete_by_mjd_range(self):
        s = SelectionState(n_toas=5)
        mjd = np.array([58000, 58100, 58200, 58300, 58400.0])
        count = s.delete_by_mjd_range(mjd, 58050, 58250)
        assert count == 2  # indices 1, 2
        assert s.deleted[1] and s.deleted[2]

    def test_delete_by_flag(self):
        s = SelectionState(n_toas=3)
        flags = [{"be": "MKBF"}, {"be": "PTUSE"}, {"be": "MKBF"}]
        count = s.delete_by_flag(flags, "be", "PTUSE")
        assert count == 1
        assert s.deleted[1]

    def test_select_indices(self):
        s = SelectionState(n_toas=5)
        s.select_indices([1, 3])
        assert s.n_selected == 2

    def test_deselect_all(self):
        s = SelectionState(n_toas=5)
        s.select_indices([0, 1, 2])
        s.deselect_all()
        assert s.n_selected == 0

    def test_toggle(self):
        s = SelectionState(n_toas=3)
        s.toggle_selection([0, 1])
        assert s.selected[0] and s.selected[1]
        s.toggle_selection([0])
        assert not s.selected[0]
        assert s.selected[1]

    def test_snapshot_roundtrip(self):
        s = SelectionState(n_toas=5)
        s.delete_indices([1])
        s.select_indices([3])
        snap = s.snapshot()
        s2 = SelectionState.from_snapshot(snap)
        assert s2.n_toas == 5
        assert s2.deleted[1]
        assert s2.selected[3]


# ===================================================================
# Epoch averaging
# ===================================================================

class TestEpochAverage:
    def test_basic_averaging(self):
        mjd = np.array([58000.0, 58000.1, 58100.0, 58100.1])
        res = np.array([1.0, 3.0, 5.0, 7.0])
        err = np.array([1.0, 1.0, 1.0, 1.0])

        avgs = epoch_average(mjd, res, err, dt_days=0.5)
        assert len(avgs) == 2  # two epochs
        # Equal weights → simple mean
        np.testing.assert_allclose(avgs[0].residual_us, 2.0, atol=1e-10)
        np.testing.assert_allclose(avgs[1].residual_us, 6.0, atol=1e-10)

    def test_error_propagation(self):
        mjd = np.array([58000.0, 58000.1])
        res = np.array([1.0, 3.0])
        err = np.array([2.0, 2.0])

        avgs = epoch_average(mjd, res, err, dt_days=0.5)
        # σ_avg = 1/√(Σ 1/σ²) = 1/√(1/4 + 1/4) = 1/√(0.5) ≈ 1.414
        expected_err = 1.0 / np.sqrt(2 * (1.0 / 4.0))
        np.testing.assert_allclose(avgs[0].error_us, expected_err, rtol=1e-10)

    def test_single_toa_epoch(self):
        mjd = np.array([58000.0, 59000.0])
        res = np.array([1.0, 2.0])
        err = np.array([1.0, 1.0])

        avgs = epoch_average(mjd, res, err, dt_days=0.5)
        assert len(avgs) == 2  # each TOA is its own epoch

    def test_active_mask(self):
        mjd = np.array([58000.0, 58000.1, 58100.0])
        res = np.array([1.0, 3.0, 5.0])
        err = np.array([1.0, 1.0, 1.0])
        mask = np.array([True, False, True])

        avgs = epoch_average(mjd, res, err, active_mask=mask, dt_days=0.5)
        assert len(avgs) == 2  # TOA 1 excluded → epoch 0 has only TOA 0

    def test_per_backend(self):
        mjd = np.array([58000.0, 58000.1, 58000.2])
        res = np.array([1.0, 3.0, 5.0])
        err = np.array([1.0, 1.0, 1.0])
        backends = ["A", "B", "A"]

        avgs = epoch_average(mjd, res, err, backends=backends, dt_days=0.5)
        # Backend A: TOA 0, 2 → one epoch
        # Backend B: TOA 1 → one epoch
        assert len(avgs) == 2

    def test_empty(self):
        avgs = epoch_average(
            np.array([]), np.array([]), np.array([])
        )
        assert avgs == []


# ===================================================================
# SessionWorkflow
# ===================================================================

class TestSessionWorkflow:
    def test_record_action(self):
        wf = SessionWorkflow()
        wf.record_action(
            "fit",
            {"F0": 100.0},
            {"F0": 100.1},
            {"rms": 1.5},
        )
        assert len(wf.history) == 1
        assert wf.fit_count == 1

    def test_undo_redo(self):
        wf = SessionWorkflow()
        wf.record_action("manual_edit", {"F0": 100.0}, {"F0": 100.1})
        wf.record_action("fit", {"F0": 100.1}, {"F0": 100.2})

        assert wf.can_undo
        params = wf.undo()
        assert params["F0"] == 100.1  # params_before of 2nd action

        assert wf.can_redo
        params = wf.redo()
        assert params["F0"] == 100.2  # params_after of 2nd action

    def test_undo_nothing(self):
        wf = SessionWorkflow()
        assert wf.undo() is None

    def test_redo_nothing(self):
        wf = SessionWorkflow()
        assert wf.redo() is None

    def test_redo_discarded_on_new_action(self):
        wf = SessionWorkflow()
        wf.record_action("a", {"x": 1}, {"x": 2})
        wf.record_action("b", {"x": 2}, {"x": 3})
        wf.undo()
        # Now record a new action — redo should be gone
        wf.record_action("c", {"x": 2}, {"x": 4})
        assert not wf.can_redo

    def test_json_roundtrip(self):
        wf = SessionWorkflow()
        wf.init_selection(10)
        wf.selection.delete_indices([1, 3])
        wf.record_action("fit", {"F0": 100.0}, {"F0": 100.1})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        wf.save_json(path)
        wf2 = SessionWorkflow.load_json(path)

        assert wf2.fit_count == 1
        assert len(wf2.history) == 1
        assert wf2.selection.n_deleted == 2

        path.unlink()

    def test_summary(self):
        wf = SessionWorkflow()
        wf.init_selection(20)
        s = wf.summary()
        assert "0 actions" in s
        assert "20/20" in s


# ===================================================================
# HistoryEntry
# ===================================================================

class TestHistoryEntry:
    def test_creation(self):
        e = HistoryEntry(
            action="fit",
            timestamp=1234567890.0,
            params_before={"F0": 100.0},
            params_after={"F0": 100.1},
        )
        assert e.action == "fit"
        assert e.params_before["F0"] == 100.0
