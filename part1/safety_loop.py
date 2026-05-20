"""v1: smallest end-to-end safety loop, stdout only, no ROS2.

Feeds a scripted distance into :func:`safety_logic.decide` at 30 Hz and prints
the result. Kept around as the no-dependency entry point for local testing.

Run:
    python3 part1/safety_loop.py

Stop with Ctrl-C. Each line is one decision; state transitions are flagged
with a "*" so they're obvious in the log.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Optional

from safety_logic import (
    SLOWDOWN_DISTANCE_M,
    STOP_DISTANCE_M,
    SafetyState,
    decide,
    scripted_distance,
)


_running = True


def _handle_sigint(_signo, _frame):
    global _running
    _running = False


def run(rate_hz: float = 30.0) -> None:
    period = 1.0 / rate_hz
    start = time.monotonic()
    last_state: Optional[SafetyState] = None
    next_tick = start

    print(
        f"safety_loop v1 — {rate_hz:.0f} Hz, "
        f"thresholds STOP\u2264{STOP_DISTANCE_M}m, SLOW\u2264{SLOWDOWN_DISTANCE_M}m. "
        f"Ctrl-C to stop.",
        flush=True,
    )

    while _running:
        now = time.monotonic()
        t = now - start
        d = scripted_distance(t)
        decision = decide(d)

        transition = "*" if decision.state is not last_state else " "
        print(
            f"{transition} t={t:6.2f}s  d={d:5.2f}m  "
            f"state={decision.state.value:<5}  v={decision.velocity_factor:.2f}",
            flush=True,
        )
        last_state = decision.state

        # Fixed-rate scheduling that doesn't drift under load.
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)
    try:
        run()
    finally:
        print("safety_loop stopped.", file=sys.stderr)
