"""Tests for the safety loop rate, fail-safe behaviour, and the mock harness.

These tests run entirely without a robot, ROS2, GDK, or Docker. They use a
MockPerceptionModule that injects controlled distance values (or errors) so
every edge-case in the orchestrator can be exercised deterministically.

Run:
    cd part1
    python3 -m pytest tests/test_safety_loop.py -v

For the fail-safe simulation you can also run interactively:
    python3 tests/test_safety_loop.py --demo stale
    python3 tests/test_safety_loop.py --demo stop
    python3 tests/test_safety_loop.py --demo slowdown
"""

from __future__ import annotations

import sys
import os
import time
import threading
import statistics
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from safety_logic import SafetyState, decide, STOP_DISTANCE_M, SLOWDOWN_DISTANCE_M
from safety_perception import StaleSensorError


# ── Mock perception ───────────────────────────────────────────────────────────

class MockPerceptionModule:
    """Drop-in replacement for PerceptionModule with no hardware dependencies.

    Accepts a *script* — an iterable of distance values (float or None) or
    ``StaleSensorError`` sentinels. Each call to ``latest_distance()`` pops
    the next item from the script.

    Usage::

        mock = MockPerceptionModule([5.0, 3.0, 1.5, None, StaleSensorError])
        mock.start()
        assert mock.latest_distance() == 5.0
        assert mock.latest_distance() == 3.0
        # ...
        mock.close()
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self._idx = 0

    def start(self) -> None:
        pass  # nothing to start for the mock

    def latest_distance(self) -> Optional[float]:
        if self._idx >= len(self._script):
            return None  # exhaused → CLEAR

        val = self._script[self._idx]
        self._idx += 1

        if val is StaleSensorError or (
            isinstance(val, type) and issubclass(val, StaleSensorError)
        ):
            raise StaleSensorError("mock: simulated stale sensor")
        if isinstance(val, StaleSensorError):
            raise val
        return val

    def close(self) -> None:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

_FAILSAFE_STOP = decide(0.0)  # STOP, v=0.0 — used as the expected fail-safe decision


def _run_loop_n_ticks(
    perception,
    n: int,
    rate_hz: float = 30.0,
) -> list[tuple]:
    """Drive the safety decision loop for *n* ticks, return (state, v, dist, failsafe) tuples."""
    results = []
    from safety_logic import decide as _decide

    for _ in range(n):
        try:
            dist = perception.latest_distance()
            failsafe = False
        except StaleSensorError:
            dist = None
            failsafe = True

        decision = _FAILSAFE_STOP if failsafe else _decide(dist)
        results.append((decision.state, decision.velocity_factor, dist, failsafe))

    return results


# ── Fail-safe tests ───────────────────────────────────────────────────────────

class TestFailSafe:
    """Stale sensor / perception crash must force STOP."""

    def test_stale_sensor_triggers_stop(self):
        """StaleSensorError from perception → STOP with failsafe=True."""
        mock = MockPerceptionModule([StaleSensorError])
        results = _run_loop_n_ticks(mock, 1)
        state, v, dist, failsafe = results[0]
        assert state is SafetyState.STOP
        assert v == 0.0
        assert failsafe is True

    def test_stale_after_clear_triggers_stop(self):
        """Clear reading followed by stale → STOP (no hysteresis, immediate failsafe)."""
        mock = MockPerceptionModule([5.0, 5.0, StaleSensorError, StaleSensorError])
        results = _run_loop_n_ticks(mock, 4)
        assert results[0][0] is SafetyState.CLEAR
        assert results[1][0] is SafetyState.CLEAR
        assert results[2][0] is SafetyState.STOP   # stale
        assert results[2][3] is True               # failsafe flag
        assert results[3][0] is SafetyState.STOP

    def test_recovery_after_stale(self):
        """After stale → STOP, a fresh clear reading must auto-release."""
        mock = MockPerceptionModule([StaleSensorError, 5.0])
        results = _run_loop_n_ticks(mock, 2)
        assert results[0][0] is SafetyState.STOP   # stale → STOP
        assert results[1][0] is SafetyState.CLEAR  # fresh clear reading → released

    def test_multiple_consecutive_stale_keeps_stop(self):
        """Sustained stale sensor must sustain STOP — never auto-release."""
        mock = MockPerceptionModule([StaleSensorError] * 10)
        results = _run_loop_n_ticks(mock, 10)
        assert all(r[0] is SafetyState.STOP for r in results)
        assert all(r[3] is True for r in results)


# ── STOP signal tests ─────────────────────────────────────────────────────────

class TestStopSignal:
    """STOP must fire at the right distance and release correctly."""

    def test_stop_in_zone(self):
        for dist in [0.5, 1.0, 1.5, 2.0]:
            mock = MockPerceptionModule([dist])
            r = _run_loop_n_ticks(mock, 1)[0]
            assert r[0] is SafetyState.STOP, f"expected STOP at {dist} m"
            assert r[1] == 0.0

    def test_no_stop_outside_zone(self):
        for dist in [2.01, 3.0, 4.0, 4.01, 10.0]:
            mock = MockPerceptionModule([dist])
            r = _run_loop_n_ticks(mock, 1)[0]
            assert r[0] is not SafetyState.STOP, f"unexpected STOP at {dist} m"

    def test_stop_sustained_while_in_zone(self):
        """STOP must persist every tick while human stays inside 2.0 m."""
        mock = MockPerceptionModule([1.5] * 30)
        results = _run_loop_n_ticks(mock, 30)
        assert all(r[0] is SafetyState.STOP for r in results)

    def test_stop_auto_release_at_4m(self):
        """Human moving from 1.5 m → 5.0 m must auto-release STOP → CLEAR."""
        mock = MockPerceptionModule([1.5, 5.0])
        results = _run_loop_n_ticks(mock, 2)
        assert results[0][0] is SafetyState.STOP
        assert results[1][0] is SafetyState.CLEAR

    def test_stop_auto_release_into_slow_zone(self):
        mock = MockPerceptionModule([1.5, 3.0])
        results = _run_loop_n_ticks(mock, 2)
        assert results[0][0] is SafetyState.STOP
        assert results[1][0] is SafetyState.SLOW

    def test_no_human_is_not_stop(self):
        """None distance (no detection) must produce CLEAR, never STOP."""
        mock = MockPerceptionModule([None] * 5)
        results = _run_loop_n_ticks(mock, 5)
        assert all(r[0] is SafetyState.CLEAR for r in results)


# ── Velocity factor tests ─────────────────────────────────────────────────────

class TestVelocityFactor:
    """velocity_factor must be 0.0 in STOP, 1.0 in CLEAR, and ramp in SLOW."""

    def test_stop_velocity_zero(self):
        for d in [0.0, 1.0, 2.0]:
            r = _run_loop_n_ticks(MockPerceptionModule([d]), 1)[0]
            assert r[1] == 0.0

    def test_clear_velocity_one(self):
        for d in [4.001, 5.0, 10.0]:
            r = _run_loop_n_ticks(MockPerceptionModule([d]), 1)[0]
            assert r[1] == 1.0

    def test_slow_velocity_midpoint(self):
        mid = (STOP_DISTANCE_M + SLOWDOWN_DISTANCE_M) / 2.0  # 3.0 m → 0.5
        r = _run_loop_n_ticks(MockPerceptionModule([mid]), 1)[0]
        assert abs(r[1] - 0.5) < 1e-9

    def test_slow_velocity_boundary_at_stop(self):
        """Just past STOP threshold → velocity_factor near 0."""
        r = _run_loop_n_ticks(MockPerceptionModule([STOP_DISTANCE_M + 0.001]), 1)[0]
        assert r[1] < 0.01

    def test_slow_velocity_boundary_at_slowdown(self):
        """At SLOWDOWN threshold → velocity_factor == 1.0."""
        r = _run_loop_n_ticks(MockPerceptionModule([SLOWDOWN_DISTANCE_M]), 1)[0]
        assert abs(r[1] - 1.0) < 1e-9


# ── 30 Hz rate test ───────────────────────────────────────────────────────────

class TestLoopRate:
    """The decision loop must maintain ~30 Hz timing."""

    def test_loop_rate_30hz(self):
        """Run 60 ticks at 30 Hz and verify elapsed time is ~2 s (±10%)."""
        rate_hz = 30.0
        n_ticks = 60
        period = 1.0 / rate_hz
        target_duration = n_ticks * period  # 2.0 s

        tick_times = []
        next_tick = time.monotonic()

        for _ in range(n_ticks):
            tick_times.append(time.monotonic())
            next_tick += period
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()

        actual_duration = tick_times[-1] - tick_times[0]
        assert abs(actual_duration - (target_duration - period)) < target_duration * 0.10, (
            f"loop took {actual_duration:.3f}s, expected ~{target_duration - period:.3f}s"
        )

    def test_inter_tick_jitter(self):
        """Tick-to-tick interval jitter must be < 5 ms (1.5 × one frame at 30 Hz)."""
        rate_hz = 30.0
        n_ticks = 30
        period = 1.0 / rate_hz
        tick_times = []
        next_tick = time.monotonic()

        for _ in range(n_ticks):
            tick_times.append(time.monotonic())
            next_tick += period
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()

        intervals = [tick_times[i+1] - tick_times[i] for i in range(len(tick_times)-1)]
        jitter = statistics.stdev(intervals) * 1000  # ms
        assert jitter < 5.0, f"tick jitter {jitter:.2f} ms exceeds 5 ms threshold"


# ── STOP streaming at 30 Hz ───────────────────────────────────────────────────

class TestStopStreaming:
    """While in STOP state, every tick must publish STOP (no skipping)."""

    def test_stop_published_every_tick_while_in_zone(self):
        """60 ticks with human at 1.5 m → all 60 must be STOP (satisfies 30 Hz req)."""
        mock = MockPerceptionModule([1.5] * 60)
        results = _run_loop_n_ticks(mock, 60)
        stop_count = sum(1 for r in results if r[0] is SafetyState.STOP)
        assert stop_count == 60, f"only {stop_count}/60 ticks were STOP"

    def test_stop_rate_at_30hz(self):
        """Measure actual STOP publish rate over 1 second."""
        rate_hz = 30.0
        duration_s = 1.0
        n_ticks = int(rate_hz * duration_s)
        period = 1.0 / rate_hz

        stop_times = []
        next_tick = time.monotonic()

        for _ in range(n_ticks):
            # Human always inside stop zone.
            decision = decide(1.5)
            if decision.state is SafetyState.STOP:
                stop_times.append(time.monotonic())
            next_tick += period
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()

        elapsed = stop_times[-1] - stop_times[0]
        measured_hz = (len(stop_times) - 1) / elapsed if elapsed > 0 else 0
        assert measured_hz >= 27.0, f"STOP rate {measured_hz:.1f} Hz < 27 Hz (30 Hz ±10%)"
        assert measured_hz <= 33.0, f"STOP rate {measured_hz:.1f} Hz > 33 Hz (30 Hz ±10%)"


# ── Interactive fail-safe demo ────────────────────────────────────────────────

def _demo(scenario: str, rate_hz: float = 30.0, duration_s: float = 4.0) -> None:
    """Interactive demo — prints decisions to stdout for visual inspection.

    Usage:  python3 tests/test_safety_loop.py --demo stale|stop|slowdown|clear
    """
    scripts = {
        "stale":    [5.0] * 30 + [StaleSensorError] * 90,
        "stop":     [5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 1.0, 1.0, 2.5, 5.0] * 10,
        "slowdown": [4.0, 3.5, 3.0, 2.5, 2.1, 2.5, 3.0, 3.5, 4.0] * 15,
        "clear":    [5.0, 6.0, 8.0] * 40,
    }
    if scenario not in scripts:
        print(f"Unknown scenario. Choose from: {list(scripts)}")
        return

    mock = MockPerceptionModule(scripts[scenario])
    period = 1.0 / rate_hz
    n = int(rate_hz * duration_s)
    print(f"Demo: {scenario}  ({rate_hz:.0f} Hz, {duration_s:.0f}s)")
    print("-" * 55)
    last_state = None
    next_tick = time.monotonic()

    for i in range(n):
        try:
            dist = mock.latest_distance()
            failsafe = False
        except StaleSensorError:
            dist = None
            failsafe = True

        decision = _FAILSAFE_STOP if failsafe else decide(dist)
        marker = "*" if decision.state is not last_state else " "
        d_str = "N/A" if dist is None else f"{dist:.2f}m"
        if decision.state is not last_state or i % int(rate_hz) == 0:
            tag = " [FAILSAFE]" if failsafe else ""
            print(
                f"{marker} t={i/rate_hz:5.2f}s  d={d_str:>7}  "
                f"state={decision.state.value:<5}  v={decision.velocity_factor:.2f}{tag}"
            )
        last_state = decision.state
        next_tick += period
        sl = next_tick - time.monotonic()
        if sl > 0:
            time.sleep(sl)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", choices=["stale", "stop", "slowdown", "clear"],
                   help="Run an interactive scenario demo instead of pytest")
    args = p.parse_args()
    if args.demo:
        _demo(args.demo)
    else:
        print("Run with: python3 -m pytest tests/test_safety_loop.py -v")
