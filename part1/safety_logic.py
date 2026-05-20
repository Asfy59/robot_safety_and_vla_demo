"""Pure safety logic shared by every entry point (loop, node, future tools).

Kept tiny and dependency-free so it can be imported in any environment — with
or without ROS2 / GDK / OpenCV. All other modules wrap this; none replace it.

Default thresholds (overridable via config.yaml):
    STOP      d ≤ 2.0 m          → velocity_factor 0.0
    SLOW      2.0 m < d ≤ 4.0 m → linear ramp 0.0 → 1.0
    CLEAR     d > 4.0 m          → velocity_factor 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Module-level defaults — the orchestrators override these from config.yaml.
STOP_DISTANCE_M: float = 2.0
SLOWDOWN_DISTANCE_M: float = 4.0


class SafetyState(str, Enum):
    STOP = "STOP"
    SLOW = "SLOW"
    CLEAR = "CLEAR"


@dataclass(frozen=True)
class SafetyDecision:
    state: SafetyState
    velocity_factor: float
    distance_m: Optional[float]

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "velocity_factor": round(self.velocity_factor, 4),
            "distance_m": (
                None if self.distance_m is None else round(self.distance_m, 3)
            ),
        }


def decide(
    distance_m: Optional[float],
    stop_m: float = STOP_DISTANCE_M,
    slow_m: float = SLOWDOWN_DISTANCE_M,
) -> SafetyDecision:
    """Map a distance reading to a :class:`SafetyDecision`.

    Parameters
    ----------
    distance_m:
        Closest detected human distance.  ``None`` means no humans detected
        (→ CLEAR).  A stale-sensor fail-safe is handled in the orchestrator,
        not here, keeping the pure logic sensor-agnostic.
    stop_m:
        Hard-stop threshold in metres (from ``config.yaml``).
    slow_m:
        Outer edge of the slowdown zone in metres (from ``config.yaml``).
    """
    if distance_m is None:
        return SafetyDecision(SafetyState.CLEAR, 1.0, None)
    if distance_m <= stop_m:
        return SafetyDecision(SafetyState.STOP, 0.0, distance_m)
    if distance_m <= slow_m:
        factor = (distance_m - stop_m) / (slow_m - stop_m)
        return SafetyDecision(SafetyState.SLOW, factor, distance_m)
    return SafetyDecision(SafetyState.CLEAR, 1.0, distance_m)


class DecisionFilter:
    """Hysteresis filter that smooths noisy state transitions.

    Stricter states (CLEAR → SLOW → STOP) latch after ``confirm_frames``
    consecutive agreeing frames.  Looser states (STOP → SLOW → CLEAR) require
    ``release_frames`` consecutive frames before releasing, making the system
    slow to clear but fast to stop.

    This is intentionally conservative: a single STOP reading will always
    immediately produce a STOP output regardless of confirm_frames, because
    safety must never be delayed on the trigger side.
    """

    _STATE_RANK = {SafetyState.CLEAR: 0, SafetyState.SLOW: 1, SafetyState.STOP: 2}

    def __init__(self, confirm_frames: int = 2, release_frames: int = 10) -> None:
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self._latched: SafetyDecision = SafetyDecision(SafetyState.CLEAR, 1.0, None)
        self._pending: Optional[SafetyDecision] = None
        self._pending_count: int = 0

    def update(self, raw: SafetyDecision) -> SafetyDecision:
        """Feed a raw decision; returns the smoothed (filtered) decision."""
        latched_rank = self._STATE_RANK[self._latched.state]
        raw_rank = self._STATE_RANK[raw.state]

        # Moving to a stricter state: fast (confirm_frames), but STOP is
        # instant — never delay a hard stop.
        if raw_rank > latched_rank:
            if raw.state is SafetyState.STOP:
                self._latched = raw
                self._pending = None
                self._pending_count = 0
                return self._latched
            threshold = self.confirm_frames

        # Moving to a looser state: slow (release_frames).
        elif raw_rank < latched_rank:
            threshold = self.release_frames

        # Same state: keep latched, reset pending.
        else:
            self._pending = None
            self._pending_count = 0
            # Update the distance on the latched decision so callers see fresh
            # distance even when state is stable.
            self._latched = raw
            return self._latched

        # Accumulate pending frames toward the threshold.
        if self._pending is not None and self._pending.state == raw.state:
            self._pending_count += 1
        else:
            self._pending = raw
            self._pending_count = 1

        if self._pending_count >= threshold:
            self._latched = raw
            self._pending = None
            self._pending_count = 0

        return self._latched

    def reset(self) -> None:
        """Reset latched state to CLEAR (used after operator acknowledge)."""
        self._latched = SafetyDecision(SafetyState.CLEAR, 1.0, None)
        self._pending = None
        self._pending_count = 0


@dataclass(frozen=True)
class SafetySnapshot:
    """One tick of the safety pipeline: raw → filtered → effective."""

    raw: SafetyDecision
    filtered: SafetyDecision
    effective: SafetyDecision
    operator_latched: bool
    failsafe: bool

    def as_dict(self) -> dict:
        d = self.effective.as_dict()
        d["raw_state"] = self.raw.state.value
        d["filtered_state"] = self.filtered.state.value
        d["effective_state"] = self.effective.state.value
        d["operator_latched"] = self.operator_latched
        d["failsafe"] = self.failsafe
        return d


class SafetyController:
    """Combines distance logic, hysteresis filter, and optional operator latch.

    When ``enable_operator_latch`` is True, entering STOP (from filtered output)
    sets ``operator_latched`` until an operator calls :meth:`reset_operator_latch`.
    While latched, :attr:`effective` stays STOP even if perception later reads
    CLEAR — useful when the human has left but hysteresis or noisy fisheye keeps
    the filtered state from releasing quickly.

    Failsafe always forces STOP and sets the latch; reset clears latch but the
    next failsafe tick will re-latch immediately.
    """

    FAILSAFE = SafetyDecision(SafetyState.STOP, velocity_factor=0.0, distance_m=None)

    def __init__(
        self,
        stop_m: float = STOP_DISTANCE_M,
        slow_m: float = SLOWDOWN_DISTANCE_M,
        confirm_frames: int = 2,
        release_frames: int = 10,
        enable_operator_latch: bool = True,
    ) -> None:
        self.stop_m = stop_m
        self.slow_m = slow_m
        self.enable_operator_latch = enable_operator_latch
        self.filter = DecisionFilter(confirm_frames, release_frames)
        self.operator_latched = False
        self._latest: Optional[SafetySnapshot] = None

    def step(
        self,
        distance_m: Optional[float],
        *,
        failsafe: bool = False,
    ) -> SafetySnapshot:
        raw = self.FAILSAFE if failsafe else decide(distance_m, self.stop_m, self.slow_m)
        filtered = raw if failsafe else self.filter.update(raw)

        if self.enable_operator_latch:
            if failsafe or filtered.state is SafetyState.STOP:
                self.operator_latched = True

        if failsafe or (self.enable_operator_latch and self.operator_latched):
            effective = SafetyDecision(
                SafetyState.STOP, 0.0, filtered.distance_m,
            )
        else:
            effective = filtered

        snap = SafetySnapshot(
            raw=raw,
            filtered=filtered,
            effective=effective,
            operator_latched=self.operator_latched,
            failsafe=failsafe,
        )
        self._latest = snap
        return snap

    def reset_operator_latch(self) -> None:
        """Operator acknowledge: release manual STOP hold and reset hysteresis."""
        self.operator_latched = False
        self.filter.reset()

    @property
    def latest(self) -> Optional[SafetySnapshot]:
        return self._latest


def scripted_distance(t_s: float) -> float:
    """Scripted fake-perception distance. Used until real perception lands.

    Sine wave 0.5 m .. 6.0 m, 12 s period — fast enough to sweep through STOP /
    SLOW / CLEAR a few times per minute, slow enough that the log is readable.
    """

    center = (0.5 + 6.0) / 2.0
    amp = (6.0 - 0.5) / 2.0
    period_s = 12.0
    return center + amp * math.sin(2 * math.pi * t_s / period_s)
