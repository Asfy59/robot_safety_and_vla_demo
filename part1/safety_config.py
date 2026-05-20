"""Load and validate the safety backend configuration from a YAML file.

Priority (highest → lowest):
  1. Path passed explicitly to ``load()``
  2. ``SAFETY_CONFIG`` environment variable
  3. ``config.yaml`` in the same directory as this file

Usage::

    from safety_config import load_config
    cfg = load_config()               # uses default config.yaml
    cfg = load_config("custom.yaml")  # explicit path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


@dataclass
class SafetyGeometryConfig:
    stop_distance_m: float = 2.0
    slowdown_distance_m: float = 4.0
    confirm_frames: int = 2
    release_frames: int = 10
    enable_operator_latch: bool = True


@dataclass
class PerceptionCfg:
    rate_hz: float = 30.0
    stale_timeout_s: float = 1.0
    yolo_model: str = "yolov8n-pose.pt"
    yolo_imgsz: int = 320
    yolo_imgsz_fisheye: int = 640
    keypoint_conf_threshold: float = 0.3
    keypoints: Union[str, list] = "core"
    # Wide-FOV fisheye + LiDAR extension (off by default).
    use_fisheye: bool = False
    use_lidar: bool = False
    fisheye_cameras: list = field(default_factory=lambda: ["left", "right", "back"])
    lidar_stale_timeout_s: float = 0.5
    lidar_search_radius_px: int = 10
    robot_ip: str = "10.42.1.101"


@dataclass
class MjpegCfg:
    port: int = 8080
    jpeg_quality: int = 75
    depth_vis_min_m: float = 0.3
    depth_vis_max_m: float = 6.0


@dataclass
class DevCfg:
    offline: bool = False


@dataclass
class AppConfig:
    safety: SafetyGeometryConfig = field(default_factory=SafetyGeometryConfig)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    mjpeg: MjpegCfg = field(default_factory=MjpegCfg)
    dev: DevCfg = field(default_factory=DevCfg)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML, falling back to built-in defaults for missing keys.

    Parameters
    ----------
    path:
        Explicit config file path. If ``None``, checks the ``SAFETY_CONFIG``
        env var, then falls back to ``config.yaml`` next to this file.
    """
    if path is None:
        path = os.environ.get("SAFETY_CONFIG", str(_DEFAULT_CONFIG))
    path = Path(path)

    raw: dict = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        print(f"[config] loaded {path}", flush=True)
    else:
        print(f"[config] {path} not found — using built-in defaults", flush=True)

    def _get(section: str, key: str, default):
        return raw.get(section, {}).get(key, default)

    safety = SafetyGeometryConfig(
        stop_distance_m=_get("safety", "stop_distance_m", 2.0),
        slowdown_distance_m=_get("safety", "slowdown_distance_m", 4.0),
        confirm_frames=int(_get("safety", "confirm_frames", 2)),
        release_frames=int(_get("safety", "release_frames", 10)),
        enable_operator_latch=bool(_get("safety", "enable_operator_latch", True)),
    )

    perception = PerceptionCfg(
        rate_hz=_get("perception", "rate_hz", 30.0),
        stale_timeout_s=_get("perception", "stale_timeout_s", 1.0),
        yolo_model=_get("perception", "yolo_model", "yolov8n-pose.pt"),
        yolo_imgsz=int(_get("perception", "yolo_imgsz", 320)),
        yolo_imgsz_fisheye=int(_get("perception", "yolo_imgsz_fisheye", 640)),
        keypoint_conf_threshold=_get("perception", "keypoint_conf_threshold", 0.3),
        keypoints=_get("perception", "keypoints", "core"),
        use_fisheye=_get("perception", "use_fisheye", False),
        use_lidar=_get("perception", "use_lidar", False),
        fisheye_cameras=_get("perception", "fisheye_cameras", ["left", "right", "back"]),
        lidar_stale_timeout_s=_get("perception", "lidar_stale_timeout_s", 0.5),
        lidar_search_radius_px=_get("perception", "lidar_search_radius_px", 10),
        robot_ip=_get("perception", "robot_ip", "10.42.1.101"),
    )

    mjpeg = MjpegCfg(
        port=_get("mjpeg", "port", 8080),
        jpeg_quality=_get("mjpeg", "jpeg_quality", 75),
        depth_vis_min_m=_get("mjpeg", "depth_vis_min_m", 0.3),
        depth_vis_max_m=_get("mjpeg", "depth_vis_max_m", 6.0),
    )

    dev = DevCfg(
        offline=_get("dev", "offline", False),
    )

    _validate(safety, perception)
    return AppConfig(safety=safety, perception=perception, mjpeg=mjpeg, dev=dev)


def _validate(safety: SafetyGeometryConfig, perception: PerceptionCfg) -> None:
    if safety.stop_distance_m <= 0:
        raise ValueError(f"stop_distance_m must be > 0, got {safety.stop_distance_m}")
    if safety.slowdown_distance_m <= safety.stop_distance_m:
        raise ValueError(
            f"slowdown_distance_m ({safety.slowdown_distance_m}) must be "
            f"> stop_distance_m ({safety.stop_distance_m})"
        )
    if not (1.0 <= perception.rate_hz <= 200.0):
        raise ValueError(f"rate_hz must be in [1, 200], got {perception.rate_hz}")
    if perception.stale_timeout_s <= 0:
        raise ValueError(f"stale_timeout_s must be > 0, got {perception.stale_timeout_s}")
