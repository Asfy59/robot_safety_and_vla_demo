"""ROS2 safety node — publishes safety decisions at 30 Hz.

This is the default container entry point. It wires the same perception
pipeline as safety_backend_gdk.py into two ROS2 topics, and additionally
serves a live MJPEG stream on port 8080.

Published topics
----------------
/safety/decision  std_msgs/String  30 Hz
    JSON payload: {"state", "velocity_factor", "distance_m", "t", "failsafe"}

/safety/stop      std_msgs/Bool    30 Hz
    True while the backend wants the robot to STOP. Doubles as a 30 Hz
    heartbeat — the motion controller can detect this node going silent.

MJPEG live stream (host networking, same port on host as container)
-------------------------------------------------------------------
http://localhost:8080/                  landing page
http://localhost:8080/stream/annotated  YOLO bboxes + skeleton + safety banner
http://localhost:8080/stream/depth      colourised depth map
http://localhost:8080/healthz           JSON health probe

All parameters are controlled by config.yaml.
See safety_config.py for the full parameter reference.

Run
---
Default (uses config.yaml in the same directory):

    python3 safety_node.py

Custom config:

    python3 safety_node.py --config /workspace/my_config.yaml
    SAFETY_CONFIG=/workspace/my_config.yaml python3 safety_node.py

Verify topics (exec into the running container):

    docker compose exec safety_backend bash
    ros2 topic echo /safety/decision
    ros2 topic hz   /safety/decision   # expect ~30 Hz
    ros2 topic echo /safety/stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from safety_config import AppConfig, load_config
from safety_logic import SafetyController, SafetyState, scripted_distance
from safety_perception import PerceptionConfig, PerceptionModule, StaleSensorError


class SafetyNode(Node):
    """30 Hz safety decision publisher.

    Reads all parameters from an AppConfig (loaded from config.yaml).
    When dev.offline is true, uses a scripted sine-wave distance instead of
    the real GDK + YOLO perception pipeline, which is useful for development
    and testing without a robot.
    """

    def __init__(self, app_cfg: AppConfig) -> None:
        super().__init__("safety_backend")
        self._rate_hz = app_cfg.perception.rate_hz
        self._stop_m = app_cfg.safety.stop_distance_m
        self._slow_m = app_cfg.safety.slowdown_distance_m
        self._safety = SafetyController(
            stop_m=app_cfg.safety.stop_distance_m,
            slow_m=app_cfg.safety.slowdown_distance_m,
            confirm_frames=app_cfg.safety.confirm_frames,
            release_frames=app_cfg.safety.release_frames,
            enable_operator_latch=app_cfg.safety.enable_operator_latch,
        )
        # Thresholds are read from config and forwarded into decide() so that
        # config.yaml is the single source of truth — no code change required.
        self._start_monotonic = time.monotonic()
        self._last_state: Optional[SafetyState] = None
        self._perception: Optional[PerceptionModule] = None
        self._tick_count: int = 0

        # QoS depth=1 keep-last: consumers only want the latest decision, never a backlog.
        self._pub_decision = self.create_publisher(String, "/safety/decision", 1)
        self._pub_stop = self.create_publisher(Bool, "/safety/stop", 1)

        if not app_cfg.dev.offline:
            pcfg = PerceptionConfig(
                stale_timeout_s=app_cfg.perception.stale_timeout_s,
                rate_hz=self._rate_hz,
                keypoint_conf_threshold=app_cfg.perception.keypoint_conf_threshold,
                yolo_model=app_cfg.perception.yolo_model,
                yolo_imgsz=app_cfg.perception.yolo_imgsz,
                yolo_imgsz_fisheye=app_cfg.perception.yolo_imgsz_fisheye,
                mjpeg_port=app_cfg.mjpeg.port,
                mjpeg_jpeg_quality=app_cfg.mjpeg.jpeg_quality,
                depth_vis_min_m=app_cfg.mjpeg.depth_vis_min_m,
                depth_vis_max_m=app_cfg.mjpeg.depth_vis_max_m,
                keypoints=app_cfg.perception.keypoints,
                use_fisheye=app_cfg.perception.use_fisheye,
                use_lidar=app_cfg.perception.use_lidar,
                fisheye_cameras=app_cfg.perception.fisheye_cameras,
                lidar_stale_timeout_s=app_cfg.perception.lidar_stale_timeout_s,
                lidar_search_radius_px=app_cfg.perception.lidar_search_radius_px,
                robot_ip=app_cfg.perception.robot_ip,
                stop_m=app_cfg.safety.stop_distance_m,
                slow_m=app_cfg.safety.slowdown_distance_m,
            )
            self._perception = PerceptionModule(pcfg, safety=self._safety)
            self._perception.start()
            self.get_logger().info(
                f"perception started — GDK + YOLOv8 ({app_cfg.perception.yolo_model}), "
                f"keypoints={app_cfg.perception.keypoints}"
            )
        else:
            self.get_logger().info("offline mode — scripted sine-wave distance")

        self._timer = self.create_timer(1.0 / self._rate_hz, self._tick)
        self.get_logger().info(
            f"safety_backend up @ {self._rate_hz:.0f} Hz  "
            f"STOP≤{self._stop_m}m  SLOW≤{self._slow_m}m"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _current_distance(self) -> tuple[Optional[float], bool]:
        """Return ``(distance_m, is_failsafe)``.

        ``distance_m`` is None when no human is detected (→ CLEAR).
        ``is_failsafe`` is True when the sensor is stale (→ forced STOP).
        """
        if self._perception is None:
            return scripted_distance(time.monotonic() - self._start_monotonic), False
        try:
            return self._perception.latest_distance(), False
        except StaleSensorError as exc:
            self.get_logger().warn(f"stale sensor → FAILSAFE STOP: {exc}")
            return None, True

    # ── 30 Hz timer callback ──────────────────────────────────────────────────

    def _tick(self) -> None:
        distance_m, failsafe = self._current_distance()
        snap = self._safety.step(distance_m, failsafe=failsafe)
        decision = snap.effective

        t = round(time.monotonic() - self._start_monotonic, 3)
        payload = snap.as_dict()
        payload["t"] = t
        self._pub_decision.publish(String(data=json.dumps(payload)))
        self._pub_stop.publish(Bool(data=decision.state is SafetyState.STOP))

        d_str = "N/A" if distance_m is None else f"{distance_m:.2f}m"
        if decision.state is not self._last_state:
            self.get_logger().info(
                f"state → {decision.state.value}  d={d_str}  "
                f"raw={snap.raw.state.value} filt={snap.filtered.state.value}  "
                f"v={decision.velocity_factor:.2f}"
                + ("  [FAILSAFE]" if failsafe else "")
                + ("  [LATCHED]" if snap.operator_latched else "")
            )
            self._last_state = decision.state
        elif self._tick_count % int(self._rate_hz) == 0:
            self.get_logger().info(
                f"t={t:.1f}s  d={d_str}  "
                f"state={decision.state.value}  v={decision.velocity_factor:.2f}"
            )
        self._tick_count += 1

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self._perception is not None:
            self._perception.close()
        super().destroy_node()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="ROS2 safety node. All parameters in config.yaml."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: config.yaml next to this script). "
             "Can also be set via the SAFETY_CONFIG environment variable.",
    )
    args, ros_args = p.parse_known_args(argv)
    app_cfg = load_config(args.config)

    rclpy.init(args=ros_args or None)
    node = SafetyNode(app_cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
