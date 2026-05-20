"""Tests for operator latch and SafetyController."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety_logic import SafetyController, SafetyState, decide


class TestOperatorLatch:
    def test_stop_latches_until_reset(self):
        ctrl = SafetyController(
            confirm_frames=1,
            release_frames=1,
            enable_operator_latch=True,
        )
        snap1 = ctrl.step(1.0)
        assert snap1.filtered.state is SafetyState.STOP
        assert snap1.operator_latched is True
        assert snap1.effective.state is SafetyState.STOP

        # Filtered releases after 1 clear frame; latch still holds effective STOP.
        snap2 = ctrl.step(5.0)
        snap3 = ctrl.step(5.0)
        assert snap3.filtered.state is SafetyState.CLEAR
        assert snap3.effective.state is SafetyState.STOP
        assert snap3.operator_latched is True

        ctrl.reset_operator_latch()
        snap4 = ctrl.step(5.0)
        assert snap4.effective.state is SafetyState.CLEAR
        assert snap4.operator_latched is False

    def test_reset_clears_filter(self):
        ctrl = SafetyController(confirm_frames=5, release_frames=5)
        ctrl.step(1.0)
        ctrl.reset_operator_latch()
        snap = ctrl.step(5.0)
        assert snap.filtered.state is SafetyState.CLEAR

    def test_latch_disabled_follows_filtered(self):
        ctrl = SafetyController(
            confirm_frames=1,
            release_frames=1,
            enable_operator_latch=False,
        )
        ctrl.step(1.0)
        snap = ctrl.step(5.0)
        assert snap.operator_latched is False
        assert snap.effective.state is snap.filtered.state

    def test_failsafe_latches_and_forces_stop(self):
        ctrl = SafetyController(enable_operator_latch=True)
        snap = ctrl.step(None, failsafe=True)
        assert snap.effective.state is SafetyState.STOP
        assert snap.operator_latched is True

    def test_snapshot_as_dict(self):
        ctrl = SafetyController(confirm_frames=1, release_frames=1)
        snap = ctrl.step(1.5)
        d = snap.as_dict()
        assert d["state"] == "STOP"
        assert d["effective_state"] == "STOP"
        assert d["raw_state"] == "STOP"
        assert "operator_latched" in d
