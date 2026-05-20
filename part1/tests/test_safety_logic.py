"""Unit tests for safety_logic.py — pure decision math, no robot/ROS2/GDK needed.

Run from the part1/ directory:
    python3 -m pytest tests/ -v

Or from anywhere on the host (no Docker needed):
    cd part1
    python3 -m pytest tests/test_safety_logic.py -v
"""

import math
import sys
import os

# Allow importing from part1/ root without installing as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from safety_logic import (
    SLOWDOWN_DISTANCE_M,
    STOP_DISTANCE_M,
    SafetyState,
    SafetyDecision,
    decide,
    scripted_distance,
)


# ── decide() — zone boundaries ────────────────────────────────────────────────

class TestDecideZones:
    """Verify correct zone assignment at and around every threshold."""

    def test_no_human_is_clear(self):
        d = decide(None)
        assert d.state is SafetyState.CLEAR
        assert d.velocity_factor == 1.0
        assert d.distance_m is None

    # ── STOP zone ────────────────────────────────────────────────────────────

    def test_stop_at_zero(self):
        d = decide(0.0)
        assert d.state is SafetyState.STOP
        assert d.velocity_factor == 0.0

    def test_stop_well_inside(self):
        d = decide(1.0)
        assert d.state is SafetyState.STOP
        assert d.velocity_factor == 0.0

    def test_stop_at_exact_threshold(self):
        """d == STOP_DISTANCE_M (2.0 m) must still be STOP (≤, not <)."""
        d = decide(STOP_DISTANCE_M)
        assert d.state is SafetyState.STOP
        assert d.velocity_factor == 0.0

    def test_stop_just_inside(self):
        d = decide(STOP_DISTANCE_M - 0.001)
        assert d.state is SafetyState.STOP

    # ── SLOW zone ────────────────────────────────────────────────────────────

    def test_slow_just_outside_stop(self):
        """One millimetre past the stop threshold → SLOW, not STOP."""
        d = decide(STOP_DISTANCE_M + 0.001)
        assert d.state is SafetyState.SLOW
        assert d.velocity_factor > 0.0

    def test_slow_midpoint_velocity(self):
        """At the midpoint of the slowdown zone (3.0 m) v_factor should be 0.5."""
        mid = (STOP_DISTANCE_M + SLOWDOWN_DISTANCE_M) / 2.0  # 3.0 m
        d = decide(mid)
        assert d.state is SafetyState.SLOW
        assert abs(d.velocity_factor - 0.5) < 1e-9

    def test_slow_at_slowdown_threshold(self):
        """d == SLOWDOWN_DISTANCE_M (4.0 m) must still be SLOW (≤, not <)."""
        d = decide(SLOWDOWN_DISTANCE_M)
        assert d.state is SafetyState.SLOW
        assert abs(d.velocity_factor - 1.0) < 1e-9

    def test_slow_velocity_increases_with_distance(self):
        """velocity_factor must be monotonically increasing in the slow zone."""
        distances = [2.1, 2.5, 3.0, 3.5, 3.9, 4.0]
        factors = [decide(d).velocity_factor for d in distances]
        assert factors == sorted(factors), "velocity_factor must increase with distance"

    def test_slow_velocity_in_range(self):
        """All velocity_factors in the slow zone must be in [0.0, 1.0]."""
        for d_m in [2.01, 2.5, 3.0, 3.5, 3.99, 4.0]:
            dec = decide(d_m)
            assert 0.0 <= dec.velocity_factor <= 1.0, f"out of range at {d_m} m"

    # ── CLEAR zone ───────────────────────────────────────────────────────────

    def test_clear_just_outside_slow(self):
        """One millimetre past SLOWDOWN_DISTANCE_M → CLEAR."""
        d = decide(SLOWDOWN_DISTANCE_M + 0.001)
        assert d.state is SafetyState.CLEAR
        assert d.velocity_factor == 1.0

    def test_clear_far_away(self):
        d = decide(10.0)
        assert d.state is SafetyState.CLEAR
        assert d.velocity_factor == 1.0


# ── decide() — output contract ────────────────────────────────────────────────

class TestDecideContract:
    """SafetyDecision must always carry the input distance back unchanged."""

    def test_distance_preserved_in_stop(self):
        d = decide(1.5)
        assert d.distance_m == 1.5

    def test_distance_preserved_in_slow(self):
        d = decide(3.0)
        assert d.distance_m == 3.0

    def test_distance_preserved_in_clear(self):
        d = decide(5.0)
        assert d.distance_m == 5.0

    def test_stop_velocity_is_zero(self):
        for d_m in [0.0, 0.5, 1.0, 1.9, 2.0]:
            assert decide(d_m).velocity_factor == 0.0

    def test_clear_velocity_is_one(self):
        for d_m in [4.001, 5.0, 10.0, 100.0]:
            assert decide(d_m).velocity_factor == 1.0


# ── as_dict() serialisation ───────────────────────────────────────────────────

class TestAsDict:
    """JSON payload must contain the right keys with correct types."""

    def test_keys_present(self):
        d = decide(1.5).as_dict()
        assert {"state", "velocity_factor", "distance_m"} <= d.keys()

    def test_state_is_string(self):
        assert decide(1.5).as_dict()["state"] == "STOP"
        assert decide(3.0).as_dict()["state"] == "SLOW"
        assert decide(5.0).as_dict()["state"] == "CLEAR"

    def test_velocity_factor_rounded(self):
        """as_dict() must round velocity_factor to 4 decimal places."""
        d = decide(3.0).as_dict()
        # 0.5 exactly is fine; check it's not more than 4 d.p. for non-round values
        d2 = decide(2.333).as_dict()
        assert len(str(d2["velocity_factor"]).split(".")[-1]) <= 4

    def test_none_distance_serialises_as_null(self):
        d = decide(None).as_dict()
        assert d["distance_m"] is None


# ── Auto-release (stateless) ──────────────────────────────────────────────────

class TestAutoRelease:
    """STOP must lift automatically the next time distance is outside the zone."""

    def test_stop_then_clear(self):
        assert decide(1.0).state is SafetyState.STOP
        assert decide(5.0).state is SafetyState.CLEAR

    def test_stop_then_slow(self):
        assert decide(1.0).state is SafetyState.STOP
        assert decide(3.0).state is SafetyState.SLOW

    def test_no_hysteresis(self):
        """decide() is stateless — same input always gives same output."""
        assert decide(1.0).state is decide(1.0).state
        assert decide(5.0).state is decide(5.0).state


# ── scripted_distance() ───────────────────────────────────────────────────────

class TestScriptedDistance:
    """Scripted distance must sweep through all three safety zones."""

    def test_range(self):
        """Output must stay within [0.5, 6.0] m over a full period."""
        period_s = 12.0
        samples = [scripted_distance(t) for t in [i * 0.1 for i in range(int(period_s / 0.1))]]
        assert min(samples) >= 0.49  # small float tolerance
        assert max(samples) <= 6.01

    def test_covers_stop_zone(self):
        samples = [scripted_distance(t) for t in [i * 0.1 for i in range(200)]]
        assert any(d <= STOP_DISTANCE_M for d in samples), "must pass through STOP zone"

    def test_covers_slow_zone(self):
        samples = [scripted_distance(t) for t in [i * 0.1 for i in range(200)]]
        assert any(STOP_DISTANCE_M < d <= SLOWDOWN_DISTANCE_M for d in samples)

    def test_covers_clear_zone(self):
        samples = [scripted_distance(t) for t in [i * 0.1 for i in range(200)]]
        assert any(d > SLOWDOWN_DISTANCE_M for d in samples)


# ── Keypoint presets ──────────────────────────────────────────────────────────

class TestKeypointPresets:
    """Verify keypoint preset definitions and PerceptionConfig resolution."""

    def test_imports(self):
        from safety_perception import KEYPOINT_NAMES, KEYPOINT_PRESETS, PerceptionConfig
        assert len(KEYPOINT_NAMES) == 17
        assert "all" in KEYPOINT_PRESETS
        assert "core" in KEYPOINT_PRESETS
        assert "torso" in KEYPOINT_PRESETS

    def test_all_preset_has_17_keypoints(self):
        from safety_perception import KEYPOINT_PRESETS
        assert len(KEYPOINT_PRESETS["all"]) == 17

    def test_core_excludes_wrists_and_ankles(self):
        from safety_perception import KEYPOINT_PRESETS
        core = KEYPOINT_PRESETS["core"]
        assert 9 not in core,  "left_wrist should be excluded from core"
        assert 10 not in core, "right_wrist should be excluded from core"
        assert 15 not in core, "left_ankle should be excluded from core"
        assert 16 not in core, "right_ankle should be excluded from core"

    def test_core_includes_elbows_and_knees(self):
        from safety_perception import KEYPOINT_PRESETS
        core = KEYPOINT_PRESETS["core"]
        assert 7 in core,  "left_elbow should be in core"
        assert 8 in core,  "right_elbow should be in core"
        assert 13 in core, "left_knee should be in core"
        assert 14 in core, "right_knee should be in core"

    def test_torso_only_hips_and_shoulders(self):
        from safety_perception import KEYPOINT_PRESETS
        torso = set(KEYPOINT_PRESETS["torso"])
        assert torso == {5, 6, 11, 12}, f"torso preset unexpected: {torso}"

    def test_config_resolves_preset_name(self):
        from safety_perception import PerceptionConfig
        cfg = PerceptionConfig(keypoints="core")
        indices = cfg.active_keypoint_indices()
        assert isinstance(indices, list)
        assert len(indices) > 0

    def test_config_accepts_custom_list(self):
        from safety_perception import PerceptionConfig
        cfg = PerceptionConfig(keypoints=[5, 6, 11, 12])
        assert cfg.active_keypoint_indices() == [5, 6, 11, 12]

    def test_config_rejects_unknown_preset(self):
        from safety_perception import PerceptionConfig
        cfg = PerceptionConfig(keypoints="bogus")
        with pytest.raises(ValueError, match="Unknown keypoint preset"):
            cfg.active_keypoint_indices()
