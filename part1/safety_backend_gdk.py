"""GDK-first standalone safety backend — no ROS2 required.

Wires safety_config → safety_perception (GDK + YOLO + MJPEG) → safety_logic
and runs the 30 Hz decision loop, printing decisions to stdout.

All parameters are controlled by config.yaml (or a custom file via --config).
No other CLI flags are needed for normal use.

Run inside the Docker container:

    python3 safety_backend_gdk.py                    # uses config.yaml
    python3 safety_backend_gdk.py --config my.yaml   # custom config
    SAFETY_CONFIG=my.yaml python3 safety_backend_gdk.py

For local testing without a robot, set dev.offline: true in config.yaml
(or use a separate offline.yaml).

MJPEG endpoints:

    http://localhost:8080/                  landing page
    http://localhost:8080/stream/annotated  YOLOv8 overlay + safety banner
    http://localhost:8080/stream/color      raw color feed
    http://localhost:8080/stream/depth      colourised depth
    http://localhost:8080/healthz           JSON health probe
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

from safety_config import load_config, AppConfig
from safety_logic import SafetyController, SafetyState, scripted_distance
from safety_perception import PerceptionConfig, PerceptionModule, StaleSensorError

_running = True


def _handle_sigint(_signo, _frame):
    global _running
    _running = False


def run(cfg: AppConfig) -> int:
    global _running

    stop_m = cfg.safety.stop_distance_m
    slow_m = cfg.safety.slowdown_distance_m
    rate_hz = cfg.perception.rate_hz
    offline = cfg.dev.offline

    perception: Optional[PerceptionModule] = None
    safety = SafetyController(
        stop_m=stop_m,
        slow_m=slow_m,
        confirm_frames=cfg.safety.confirm_frames,
        release_frames=cfg.safety.release_frames,
        enable_operator_latch=cfg.safety.enable_operator_latch,
    )

    if not offline:
        pcfg = PerceptionConfig(
            stale_timeout_s=cfg.perception.stale_timeout_s,
            rate_hz=rate_hz,
            keypoint_conf_threshold=cfg.perception.keypoint_conf_threshold,
            yolo_model=cfg.perception.yolo_model,
            yolo_imgsz=cfg.perception.yolo_imgsz,
            yolo_imgsz_fisheye=cfg.perception.yolo_imgsz_fisheye,
            mjpeg_port=cfg.mjpeg.port,
            mjpeg_jpeg_quality=cfg.mjpeg.jpeg_quality,
            depth_vis_min_m=cfg.mjpeg.depth_vis_min_m,
            depth_vis_max_m=cfg.mjpeg.depth_vis_max_m,
            keypoints=cfg.perception.keypoints,
            use_fisheye=cfg.perception.use_fisheye,
            use_lidar=cfg.perception.use_lidar,
            fisheye_cameras=cfg.perception.fisheye_cameras,
            lidar_stale_timeout_s=cfg.perception.lidar_stale_timeout_s,
            lidar_search_radius_px=cfg.perception.lidar_search_radius_px,
            robot_ip=cfg.perception.robot_ip,
            stop_m=cfg.safety.stop_distance_m,
            slow_m=cfg.safety.slowdown_distance_m,
        )
        perception = PerceptionModule(pcfg, safety=safety)
        perception.start()
        print("waiting for perception worker to settle (~5 s)...", flush=True)
        time.sleep(5.0)
    else:
        print("OFFLINE mode — using scripted distance (no GDK/YOLO)", flush=True)

    start_t = time.monotonic()
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    last_state: Optional[SafetyState] = None
    tick = 0
    print(
        f"safety_backend_gdk running @ {rate_hz:.0f} Hz  "
        f"STOP≤{stop_m}m  SLOW≤{slow_m}m  "
        f"{'offline/scripted' if offline else 'GDK live'}",
        flush=True,
    )

    try:
        while _running:
            t = time.monotonic() - start_t

            if offline:
                distance_m: Optional[float] = scripted_distance(t)
                failsafe = False
            else:
                assert perception is not None
                try:
                    distance_m = perception.latest_distance()
                    failsafe = False
                except StaleSensorError as exc:
                    distance_m = None
                    failsafe = True
                    if last_state is not SafetyState.STOP:
                        print(f"[FAILSAFE] stale sensor → STOP: {exc}", flush=True)

            snap = safety.step(distance_m, failsafe=failsafe)
            decision = snap.effective

            latch = " [LATCHED]" if snap.operator_latched else ""
            marker = "*" if decision.state is not last_state else " "
            if decision.state is not last_state:
                print(
                    f"{marker} t={t:7.2f}s  "
                    f"d={'N/A' if distance_m is None else f'{distance_m:.2f}m':>7}  "
                    f"raw={snap.raw.state.value} filt={snap.filtered.state.value}  "
                    f"eff={decision.state.value:<5}  v={decision.velocity_factor:.2f}"
                    + ("  [FAILSAFE]" if failsafe else "")
                    + latch,
                    flush=True,
                )
            elif tick % int(rate_hz) == 0:
                print(
                    f"  t={t:7.2f}s  "
                    f"d={'N/A' if distance_m is None else f'{distance_m:.2f}m':>7}  "
                    f"eff={decision.state.value:<5}  v={decision.velocity_factor:.2f}"
                    + latch,
                    flush=True,
                )
            last_state = decision.state
            tick += 1

            next_tick += period
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        if perception is not None:
            perception.close()
        print("safety_backend_gdk stopped.", file=sys.stderr)

    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="GDK-first safety backend. All parameters in config.yaml."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: config.yaml next to this script). "
             "Can also be set via SAFETY_CONFIG env var.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)
    args = parse_args(argv)
    cfg = load_config(args.config)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
