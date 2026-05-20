"""Real-perception layer for the safety backend.

Sensor choice
-------------
Primary: **kHeadColor + kHeadDepth** (640×400 @ ~30 fps).

Justification:
* Both cameras are always-on in base mode — no activation required.
* The pair provides synchronized RGB + metric depth. Running YOLOv8n-pose
  on the color frame identifies where a person is in pixel space; sampling
  the registered depth image at those pixels gives a direct metric range
  without stereo matching.
* 30 fps gives ≥1.5 s of warning at normal walking speed (1.4 m/s) before
  a person reaches the 2 m hard-stop boundary.
* Covers the robot's front hemisphere — the primary interaction direction.

Secondary (optional): **kHeadLeft/Right/BackFisheye + kLidarFront** (15 fps / 10 Hz).

When ``use_fisheye: true`` and ``use_lidar: true`` in config.yaml, a second
perception path activates alongside the primary RGB-D path:
* Three fisheye cameras (≈180° FOV each) cover the full hemisphere including
  sides and back — closing the blind spots of the primary camera.
* The front LiDAR point cloud is projected into each fisheye image plane
  using the equidistant fisheye model (cv2.fisheye.projectPoints).  For each
  YOLO keypoint the nearest projected LiDAR point within ``lidar_search_radius_px``
  pixels provides a metric depth.
* The final reported distance is min(primary_distance, secondary_distance) so
  the secondary path can only make the system *more* conservative.
* If the secondary path goes stale the primary path continues unaffected —
  no additional FAILSAFE is triggered.

Fisheye activation
------------------
Fisheye cameras require ``develop`` mode on the robot (disabled by default).
When ``use_fisheye: true`` the backend automatically:
1. Writes the dev camera config to the robot via SSH (``robot_ip``).
2. Calls ``Camera.set_dev_camera_config`` via GDK.
3. SSH-switches the robot to ``develop`` mode.
On shutdown ``close()`` restores ``base`` mode so the robot is left clean.

Blind spots:
* Sides and back. Adding back LiDAR or fisheye cameras closes this gap.

Keypoint selection
------------------
Minimum depth over all YOLO-pose skeleton keypoints with confidence > 0.3.
Most conservative choice: STOP triggers as soon as *any* visible body part
enters the danger zone, not just the torso centroid.

Fail-safe
---------
If no new color+depth pair arrives within STALE_TIMEOUT_S (default 1 s),
`latest_distance()` raises `StaleSensorError`. The orchestrator maps this
to STOP so the robot halts rather than running blind.

3-D transform pipeline
-----------------------
1. YOLO keypoints are in color-image pixel space (u, v).
2. Depth d = uint16 value at (u, v) / 1000.0  [mm → m].
3. Project to 3-D in the RGBD camera frame using intrinsics (fx, fy, cx, cy):
       X_cam = (u - cx) * d / fx
       Y_cam = (v - cy) * d / fy
       Z_cam = d
4. Rotate + translate from RGBD camera frame to head_link3 via the
   kHeadRGBDToHeadLink3 extrinsic (quaternion + translation from GDK).
5. Rotate + translate from head_link3 to base_link via TF.
6. Return 3-D Euclidean distance from base_link origin.

If intrinsics are unavailable (empty list from GDK), fall back to
pinhole-model estimates based on image dimensions (fx=fy≈image_width).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from scipy.spatial.transform import Rotation


# ── Keypoint index map (COCO-17, used by YOLOv8-pose) ────────────────────────

KEYPOINT_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",   2: "right_eye",
    3: "left_ear",   4: "right_ear",
    5: "left_shoulder",  6: "right_shoulder",
    7: "left_elbow",     8: "right_elbow",
    9: "left_wrist",    10: "right_wrist",
    11: "left_hip",     12: "right_hip",
    13: "left_knee",    14: "right_knee",
    15: "left_ankle",   16: "right_ankle",
}

# Named presets — trade noise tolerance vs. coverage.
#
# "all"     All 17 keypoints. Most sensitive (catches an arm reaching in),
#           most noise from swinging extremities.
# "core"    Hips + shoulders + elbows + knees.  No wrists/ankles.
#           Good balance: ignores arm-swing and foot noise while still
#           detecting elbows (intermediate reach indicator).
# "torso"   Hips + shoulders only.  Most stable, least false-positives.
#           Use when the environment is busy or depth is noisy.
KEYPOINT_PRESETS: dict[str, list[int]] = {
    "all":    list(range(17)),
    "core":   [5, 6, 7, 8, 11, 12, 13, 14],   # shoulders, elbows, hips, knees
    "torso":  [5, 6, 11, 12],                   # shoulders + hips only
}


# ── Fisheye camera / extrinsic name maps ─────────────────────────────────────
# GDK enum attribute names, resolved at runtime to avoid import-time errors.

_FISHEYE_CAMERA_ATTR: dict[str, str] = {
    "left":  "kHeadLeftFisheye",
    "right": "kHeadRightFisheye",
    "back":  "kHeadBackFisheye",
}

_FISHEYE_EXTRINSIC_ATTR: dict[str, str] = {
    "left":  "kHeadLeftFisheyeToHeadLink3",
    "right": "kHeadRightFisheyeToHeadLink3",
    "back":  "kHeadBackFisheyeToHeadLink3",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PerceptionConfig:
    stale_timeout_s: float = 1.0
    rate_hz: float = 30.0
    stop_m: float = 2.0   # mirrored from SafetyGeometryConfig for the MJPEG UI
    slow_m: float = 4.0
    keypoint_conf_threshold: float = 0.3
    yolo_model: str = "yolov8n-pose.pt"
    yolo_imgsz: int = 320
    yolo_imgsz_fisheye: int = 640
    mjpeg_port: int = 8080
    mjpeg_jpeg_quality: int = 75
    # Depth visualisation range for MJPEG stream
    depth_vis_min_m: float = 0.3
    depth_vis_max_m: float = 6.0
    # Which keypoints to use for distance estimation.
    # Accepts a preset name ("all" | "core" | "torso") or a list of indices.
    # Default "core": hips, shoulders, elbows, knees — excludes wrists/ankles
    # to reduce noise from arm-swing and foot movement.
    keypoints: str | list[int] = "core"
    # ── Wide-FOV secondary path (fisheye + LiDAR) ─────────────────────────────
    use_fisheye: bool = False
    use_lidar: bool = False
    fisheye_cameras: list = field(default_factory=lambda: ["left", "right", "back"])
    lidar_stale_timeout_s: float = 0.5
    lidar_search_radius_px: int = 10
    robot_ip: str = "10.42.1.101"

    def active_keypoint_indices(self) -> list[int]:
        """Return the resolved list of keypoint indices."""
        if isinstance(self.keypoints, str):
            if self.keypoints not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset {self.keypoints!r}. "
                    f"Choose from {list(KEYPOINT_PRESETS)} or pass a list of ints."
                )
            return KEYPOINT_PRESETS[self.keypoints]
        return list(self.keypoints)


# ── Exceptions ────────────────────────────────────────────────────────────────

class StaleSensorError(RuntimeError):
    """Raised when the last camera frame is older than stale_timeout_s."""


# ── Internal frame buffer ─────────────────────────────────────────────────────

@dataclass
class _FrameBuffer:
    lock: threading.Lock = field(default_factory=threading.Lock)
    color_bgr: Optional[np.ndarray] = None      # (H, W, 3) uint8 BGR
    depth_m: Optional[np.ndarray] = None        # (H, W) float32 metres, NaN=invalid
    annotated_bgr: Optional[np.ndarray] = None  # color + YOLO overlay
    distance_m: Optional[float] = None          # nearest human, None=no human
    last_frame_t: float = 0.0                   # monotonic
    stop: bool = False
    # Fisheye annotated frames (populated when use_fisheye=True)
    fisheye_bgr: dict = field(default_factory=dict)  # name → (H,W,3) BGR


# ── GDK helpers ───────────────────────────────────────────────────────────────

def _decode_depth(raw: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert raw GDK depth bytes to float32 metres array (H×W).

    GDK 2.6.3 depth image: uint8 byte buffer, reinterpreted as uint16 mm values.
    Zero values mean "no return" and are set to NaN.
    """
    d16 = raw.view(np.uint16).reshape(height, width)
    out = d16.astype(np.float32) / 1000.0
    out[d16 == 0] = np.nan
    return out


def _color_to_bgr(img) -> np.ndarray:
    """Convert a GDK Image to BGR uint8 (H×W×3).

    Handles JPEG-encoded streams (wired connection) and raw formats
    (NV12 / YUYV / RGB / BGR) transparently.
    """
    arr = np.asarray(img.data)
    fmt = str(getattr(img, "color_format", "")).lower()
    enc = str(getattr(img, "encoding", "")).lower()

    # ── JPEG (wired/compressed stream) ────────────────────────────────────
    if "jpeg" in enc or (arr.ndim == 1 and arr.shape[0] < img.width * img.height // 2):
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # returns BGR
        if decoded is not None:
            # imdecode gives BGR; the color_format says RGB but JPEG already
            # carries color info — check if we need to swap.
            if "rgb" in fmt and "rgb" not in enc:
                return cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
            return decoded
        print("[perception] WARN: JPEG decode failed, emitting black frame",
              file=sys.stderr)
        return np.zeros((img.height, img.width, 3), dtype=np.uint8)

    # ── Raw 3-channel HWC ─────────────────────────────────────────────────
    if arr.ndim == 3 and arr.shape[2] == 3:
        if "rgb" in fmt or "rgb" in enc:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    # ── Planar / packed formats ───────────────────────────────────────────
    if arr.ndim == 2:
        h, w = arr.shape
        if "nv12" in fmt or "nv12" in enc:
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)
        if "yuyv" in fmt or "yuyv" in enc:
            return cv2.cvtColor(arr.reshape(img.height, img.width, 2),
                                cv2.COLOR_YUV2BGR_YUYV)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # ── 1-D raw RGB/BGR flat buffer ───────────────────────────────────────
    if arr.ndim == 1 and arr.shape[0] == img.height * img.width * 3:
        mat = arr.reshape(img.height, img.width, 3)
        if "rgb" in fmt:
            return cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        return mat.copy()

    print(f"[perception] WARN: unsupported color shape {arr.shape}; "
          f"enc={enc!r} fmt={fmt!r}", file=sys.stderr)
    return np.zeros((img.height, img.width, 3), dtype=np.uint8)


def _get_intrinsics(cam, cam_type) -> tuple[float, float, float, float, int, int]:
    """Return (fx, fy, cx, cy, w, h) for a camera type.

    Falls back to principal-point-at-centre and fx=fy=image_width heuristic
    if the GDK intrinsics API returns an empty list for this camera.
    """
    try:
        intr = cam.get_camera_intrinsic(cam_type)
        if intr is not None and len(intr.intrinsic) == 4:
            fx, fy, cx, cy = intr.intrinsic
            shape = cam.get_image_shape(cam_type)
            w, h = int(shape[0]), int(shape[1])
            return fx, fy, cx, cy, w, h
    except Exception as e:
        print(f"[perception] intrinsics unavailable ({e}), using defaults",
              file=sys.stderr)

    # Fallback: probe a frame for resolution, assume square pixels and 70° HFOV.
    try:
        import agibot_gdk as g  # type: ignore
        probe = cam.get_latest_image(cam_type, 2000.0)
        w, h = probe.width, probe.height
    except Exception:
        w, h = 640, 400
    # ~70° HFOV → fx = w / (2 * tan(35°)) ≈ w * 0.713
    fx = fy = w * 0.713
    cx, cy = w / 2.0, h / 2.0
    print(f"[perception] using fallback intrinsics: fx={fx:.1f} cx={cx:.1f} cy={cy:.1f}",
          flush=True)
    return fx, fy, cx, cy, w, h


def _get_extrinsic_quat_trans(tf_module, extrinsic_type):
    """Fetch a GDK sensor extrinsic as (scipy Rotation, translation np.array).

    The GDK Transform object exposes .rotation and .translation directly
    (no nested .transform wrapper).  Returns (None, None) if unavailable.
    """
    try:
        ext = tf_module.get_tf_from_sensor(extrinsic_type)
        if ext is None:
            return None, None
        # GDK Transform: direct .rotation (Quaternion) and .translation (Vector3)
        r = ext.rotation
        t = ext.translation
        rot = Rotation.from_quat([r.x, r.y, r.z, r.w])
        trans = np.array([t.x, t.y, t.z])
        return rot, trans
    except Exception as e:
        print(f"[perception] extrinsic unavailable ({e})", file=sys.stderr)
        return None, None


def _get_head_link3_tf(tf_module):
    """Return (Rotation, translation) for base_link → head_link3."""
    try:
        for ts in tf_module.get_all_tf_from_base_link():
            if ts.child_frame_id == "head_link3":
                t = ts.transform.translation
                r = ts.transform.rotation
                rot = Rotation.from_quat([r.x, r.y, r.z, r.w])
                trans = np.array([t.x, t.y, t.z])
                return rot, trans
    except Exception as e:
        print(f"[perception] head_link3 TF unavailable ({e})", file=sys.stderr)
    return None, None


# ── Fisheye activation / deactivation ────────────────────────────────────────

_DEV_CAM_CONFIG = """{
  "cam0":  {"fps": "30", "name": "head_stereo_right",  "publish": true},
  "cam3":  {"fps": "30", "name": "head_stereo_left",   "publish": true},
  "cam5":  {"fps": "30", "name": "hand_left_color",    "publish": true},
  "cam7":  {"fps": "30", "name": "hand_right_color",   "publish": true},
  "cam10": {"fps": "15", "name": "head_right_fisheye", "publish": true},
  "cam11": {"fps": "15", "name": "head_left_fisheye",  "publish": true},
  "cam12": {"fps": "15", "name": "head_back_fisheye",  "publish": true},
  "cam14": {"fps": "30", "name": "head_depth",         "publish": true},
  "cam15": {"fps": "30", "name": "head_color",         "publish": true}
}"""


def _ssh(robot_ip: str, cmd: str, timeout: int = 15) -> bool:
    """Run a command on the robot over SSH using subprocess.

    Sources /home/agi/app/env.sh first so GDK binaries (mode_switch etc.)
    find their shared libraries.  Returns True on success.
    """
    import subprocess as _sp
    full_cmd = f"source /home/agi/app/env.sh && {cmd}"
    try:
        res = _sp.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",        # fail fast if no key auth
             f"agi@{robot_ip}", full_cmd],
            capture_output=True, timeout=timeout,
        )
        if res.returncode != 0:
            print(
                f"[fisheye] SSH failed (rc={res.returncode}): "
                f"{res.stderr.decode(errors='replace').strip()}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(f"[fisheye] SSH error: {e}", file=sys.stderr)
        return False


def _activate_fisheye(cam, robot_ip: str) -> bool:
    """Full fisheye activation following the docs:

    1. Write cam_config.json to robot via SSH.
    2. Call GDK set_dev_camera_config from inside the container.
    3. Switch robot to develop mode via SSH.

    Falls back gracefully if SSH is unavailable (e.g. no key auth) — the
    backend can still read fisheye frames if the robot is already in develop
    mode from a prior manual activation.
    """
    print(f"[fisheye] activating on {robot_ip}...", flush=True)

    # Step 1 — write cam_config.json onto the robot.
    write_cmd = (
        "mkdir -p /data/gdk && "
        f"echo '{_DEV_CAM_CONFIG}' > /data/gdk/cam_config.json"
    )
    if _ssh(robot_ip, write_cmd):
        print("[fisheye] cam_config.json written.", flush=True)
    else:
        print("[fisheye] WARNING: could not write cam_config.json via SSH "
              "(run scripts/activate_fisheye.sh manually if needed)",
              file=sys.stderr)

    # Step 2 — apply config via GDK Camera API.
    try:
        cam.set_dev_camera_config("/data/gdk/cam_config.json")
        print("[fisheye] set_dev_camera_config ok", flush=True)
    except Exception as e:
        # Non-fatal: the mode_switch below will pick up the JSON from disk.
        print(f"[fisheye] set_dev_camera_config (non-fatal): {e}", file=sys.stderr)

    # Step 3 — switch to develop mode.
    if _ssh(robot_ip, "/home/agi/app/bin/mode_switch --mode develop"):
        print("[fisheye] develop mode active — waiting 5 s for DDS topics...",
              flush=True)
        time.sleep(5.0)
        return True
    else:
        print("[fisheye] WARNING: mode_switch failed — "
              "fisheye will work if robot is already in develop mode.",
              file=sys.stderr)
        time.sleep(2.0)
        return False


def _deactivate_fisheye(robot_ip: str) -> None:
    """Restore base mode on shutdown."""
    print(f"[fisheye] restoring base mode on {robot_ip}...", flush=True)
    if not _ssh(robot_ip, "/home/agi/app/bin/mode_switch --mode base"):
        print("[fisheye] WARNING: could not restore base mode via SSH — "
              "run scripts/deactivate_fisheye.sh manually.",
              file=sys.stderr)


# ── Fisheye intrinsics ─────────────────────────────────────────────────────────

def _get_fisheye_K_D(cam, cam_type) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (K 3×3, D 4×1, width, height) for a fisheye camera.

    K is the standard pinhole-style intrinsic matrix used by cv2.fisheye.
    D is the equidistant distortion vector [k1, k2, k3, k4].
    Falls back to heuristic values if GDK does not return valid intrinsics.
    """
    try:
        intr = cam.get_camera_intrinsic(cam_type)
        shape = cam.get_image_shape(cam_type)
        w, h = int(shape[0]), int(shape[1])
        if intr is not None and len(intr.intrinsic) >= 4:
            fx, fy, cx, cy = intr.intrinsic[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = list(intr.distortion) if intr.distortion else [0, 0, 0, 0]
            # cv2.fisheye expects exactly 4 distortion coefficients.
            D = np.array(dist[:4] + [0] * (4 - len(dist[:4])),
                         dtype=np.float64).reshape(4, 1)
            return K, D, w, h
    except Exception as e:
        print(f"[fisheye] intrinsics unavailable ({e}), using defaults", file=sys.stderr)

    # Fallback: fisheye cameras are 1920×1536; ~185° FOV → f ≈ W/π
    w, h = 1920, 1536
    f = w / np.pi
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    print(f"[fisheye] using fallback intrinsics: f={f:.1f} {w}×{h}", flush=True)
    return K, D, w, h


# ── LiDAR point cloud parsing ─────────────────────────────────────────────────

# ROS PointCloud2 / GDK PointField datatype constants.
_PC2_DTYPE: dict[int, type] = {
    1: np.int8,   2: np.uint8,
    3: np.int16,  4: np.uint16,
    5: np.int32,  6: np.uint32,
    7: np.float32, 8: np.float64,
}


def _parse_pointcloud(pc) -> Optional[np.ndarray]:
    """Parse a GDK PointCloud to a (N, 3) float32 XYZ array in sensor frame.

    GDK exposes the point cloud as a flat uint8 buffer with a ``point_step``
    stride and per-field ``offset`` / ``datatype`` metadata — identical to
    ROS sensor_msgs/PointCloud2.  The Livox front LiDAR layout is:
        x(float32) y(float32) z(float32) reflectivity(uint8) tag(uint8)
        timestamp(float64)  → point_step = 22 bytes.

    Returns None if parsing fails so the caller can fall back gracefully.
    """
    try:
        raw = np.asarray(pc.data)
        if raw.dtype != np.uint8:
            raw = raw.view(np.uint8)
        raw = raw.flatten()

        point_step = int(pc.point_step)
        n_pts = len(raw) // point_step
        if n_pts == 0:
            return None

        # Build a numpy structured dtype from the field metadata.
        dt_fields: list = []
        cursor = 0
        for f in sorted(pc.fields, key=lambda x: x.offset):
            if f.offset > cursor:
                dt_fields.append((f"_pad{cursor}", np.uint8, f.offset - cursor))
            elem_dt = _PC2_DTYPE.get(f.datatype, np.uint8)
            dt_fields.append((f.name, elem_dt))
            cursor = f.offset + np.dtype(elem_dt).itemsize
        if cursor < point_step:
            dt_fields.append((f"_tail", np.uint8, point_step - cursor))

        struct_dt = np.dtype(dt_fields)
        points = np.frombuffer(raw.tobytes(), dtype=struct_dt, count=n_pts)
        xyz = np.column_stack([
            points["x"].astype(np.float32),
            points["y"].astype(np.float32),
            points["z"].astype(np.float32),
        ])
        valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.01)
        return xyz[valid].astype(np.float32)
    except Exception as e:
        print(f"[perception] LiDAR parse failed: {e}", file=sys.stderr)
        return None


# ── LiDAR → fisheye projection ────────────────────────────────────────────────

def _project_lidar_to_fisheye_pixels(
    xyz_base: np.ndarray,                   # (N, 3) in base_link frame
    K: np.ndarray,                          # (3, 3) fisheye intrinsic
    D: np.ndarray,                          # (4, 1) equidistant distortion
    rot_cam_to_head: Rotation,
    trans_cam_to_head: np.ndarray,
    rot_head_to_base: Rotation,
    trans_head_to_base: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:                            # (M, 3) [u, v, Z_cam] in-image points
    """Project base_link XYZ points into a fisheye image using the equidistant model.

    Returns only points with positive depth that fall within image bounds.
    Z_cam is the depth along the optical axis (metres) — used for distance lookup.
    """
    if len(xyz_base) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # base_link → head_link3 (inverse of head_link3→base_link)
    xyz_head = rot_head_to_base.inv().apply(xyz_base - trans_head_to_base)

    # head_link3 → fisheye camera frame (inverse of cam→head_link3 extrinsic)
    xyz_cam = rot_cam_to_head.inv().apply(xyz_head - trans_cam_to_head)

    # Keep only points in front of the camera (Z > 0).
    front = xyz_cam[:, 2] > 0.1
    if not np.any(front):
        return np.zeros((0, 3), dtype=np.float32)
    pts = xyz_cam[front].astype(np.float64)
    Z_cam = pts[:, 2]

    # Project with cv2.fisheye.projectPoints (equidistant model).
    rvec = np.zeros((1, 1, 3), dtype=np.float64)
    tvec = np.zeros((1, 1, 3), dtype=np.float64)
    pts_in = pts.reshape(-1, 1, 3)
    try:
        proj, _ = cv2.fisheye.projectPoints(pts_in, rvec, tvec, K, D)
    except cv2.error:
        return np.zeros((0, 3), dtype=np.float32)

    uv = proj.reshape(-1, 2)
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    if not np.any(in_bounds):
        return np.zeros((0, 3), dtype=np.float32)

    return np.column_stack([uv[in_bounds], Z_cam[in_bounds]]).astype(np.float32)


# ── Fisheye keypoint depth lookup + base_link projection ──────────────────────

def _fisheye_keypoints_to_base_link(
    keypoints_uv: np.ndarray,      # (17, 3) [u, v, conf]
    lidar_uvz: np.ndarray,         # (M, 3) [u, v, Z_cam] projected LiDAR points
    K: np.ndarray,                 # (3, 3) fisheye intrinsic
    D: np.ndarray,                 # (4, 1) equidistant distortion
    rot_cam_to_head: Rotation,
    trans_cam_to_head: np.ndarray,
    rot_head_to_base: Rotation,
    trans_head_to_base: np.ndarray,
    conf_threshold: float,
    active_indices: list[int],
    search_radius_px: int,
) -> list[float]:
    """Return base_link distances for each active fisheye keypoint with LiDAR depth.

    For each keypoint:
    1. Find nearest projected LiDAR points within search_radius_px.
    2. Take minimum Z (closest surface — conservative safety choice).
    3. Unproject through fisheye model to get a 3-D ray.
    4. Scale the ray by the LiDAR depth to get 3-D position in camera frame.
    5. Transform camera → head_link3 → base_link.
    """
    distances = []
    if len(lidar_uvz) == 0:
        return distances

    for idx, (u, v, conf) in enumerate(keypoints_uv):
        if idx not in active_indices or conf < conf_threshold:
            continue

        # Find closest LiDAR points in pixel space.
        px_dists = np.sqrt(
            (lidar_uvz[:, 0] - u) ** 2 + (lidar_uvz[:, 1] - v) ** 2
        )
        nearby_mask = px_dists < search_radius_px
        if not np.any(nearby_mask):
            continue

        # Most conservative: minimum Z among nearby points.
        depth_m = float(np.min(lidar_uvz[nearby_mask, 2]))
        if depth_m <= 0:
            continue

        # Unproject pixel → normalised ray in camera frame.
        pt_px = np.array([[[u, v]]], dtype=np.float64)
        try:
            ray_norm = cv2.fisheye.undistortPoints(pt_px, K, D)
        except cv2.error:
            continue
        # ray_norm shape: (1,1,2) — gives (X/Z, Y/Z) in undistorted space.
        Xn, Yn = float(ray_norm[0, 0, 0]), float(ray_norm[0, 0, 1])
        # Scale by depth to get 3-D point in camera frame.
        p_cam = np.array([Xn * depth_m, Yn * depth_m, depth_m])

        # Camera frame → head_link3 → base_link.
        p_head = rot_cam_to_head.apply(p_cam) + trans_cam_to_head
        p_base = rot_head_to_base.apply(p_head) + trans_head_to_base

        distances.append(float(np.linalg.norm(p_base)))

    return distances


# ── YOLO + projection ─────────────────────────────────────────────────────────

def _project_keypoints_to_base_link(
    keypoints_uv: np.ndarray,       # (N, 3) [u, v, conf]  — all 17 keypoints
    depth_m: np.ndarray,            # (H, W) float32
    fx: float, fy: float, cx: float, cy: float,
    rot_cam_to_head: Optional[Rotation],
    trans_cam_to_head: Optional[np.ndarray],
    rot_head_to_base: Optional[Rotation],
    trans_head_to_base: Optional[np.ndarray],
    conf_threshold: float = 0.3,
    active_indices: Optional[list[int]] = None,  # None → use all
) -> list[float]:
    """Return list of 3-D distances (from base_link origin) for each active keypoint."""
    H, W = depth_m.shape
    distances = []

    for idx, (u, v, conf) in enumerate(keypoints_uv):
        if active_indices is not None and idx not in active_indices:
            continue
        if conf < conf_threshold:
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue

        # Sample a 7×7 patch to smooth over depth holes on people.
        R = 3
        patch = depth_m[
            max(0, vi - R):min(H, vi + R + 1),
            max(0, ui - R):min(W, ui + R + 1),
        ]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            continue
        d = float(np.median(valid))

        # Project to 3-D in camera frame.
        X_cam = (u - cx) * d / fx
        Y_cam = (v - cy) * d / fy
        Z_cam = d
        p_cam = np.array([X_cam, Y_cam, Z_cam])

        # Camera frame → head_link3.
        if rot_cam_to_head is not None and trans_cam_to_head is not None:
            p_head = rot_cam_to_head.apply(p_cam) + trans_cam_to_head
        else:
            p_head = p_cam  # assume identity

        # head_link3 → base_link.
        if rot_head_to_base is not None and trans_head_to_base is not None:
            p_base = rot_head_to_base.apply(p_head) + trans_head_to_base
        else:
            # Rough fallback: add typical head height.
            p_base = p_head + np.array([0.0, 0.0, 1.6])

        dist = float(np.linalg.norm(p_base))
        distances.append(dist)

    return distances


def _run_yolo(model, color_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640):
    """Run YOLOv8-pose and return list of (keypoints (N,3), bbox (x1,y1,x2,y2)).

    Each entry in the returned list is one detected person.
    imgsz controls the internal YOLO inference resolution — 320 is fast on CPU,
    640 is the standard trade-off, higher only helps with tiny distant people.
    """
    results = model(color_bgr, verbose=False, conf=conf, imgsz=imgsz)
    detections = []
    for res in results:
        if res.keypoints is None or res.boxes is None:
            continue
        kps = res.keypoints.data.cpu().numpy()   # (P, 17, 3) → [x, y, conf]
        boxes = res.boxes.xyxy.cpu().numpy()      # (P, 4)
        for i in range(len(boxes)):
            kp_person = kps[i]   # (17, 3)
            detections.append((kp_person, boxes[i]))
    return detections


# ── Background worker ─────────────────────────────────────────────────────────

def _perception_worker(buf: _FrameBuffer, cfg: PerceptionConfig) -> None:
    """Background thread: GDK init → YOLO loop → store results in buf."""
    import agibot_gdk as g  # type: ignore
    from ultralytics import YOLO  # type: ignore

    print("[perception] loading YOLO model...", flush=True)
    model = YOLO(cfg.yolo_model)
    print("[perception] YOLO loaded", flush=True)

    print("[perception] gdk_init...", flush=True)
    if g.gdk_init() != g.GDKRes.kSuccess:
        print("[perception] gdk_init failed", file=sys.stderr)
        buf.stop = True
        return

    cam = tf = lidar = None
    _fisheye_activated = False
    try:
        cam = g.Camera()
        tf = g.TF()
        if cfg.use_lidar:
            lidar = g.Lidar()
        time.sleep(4)  # wait for DDS settle (Camera=3s, TF=2s, Lidar=3s → max+buffer)

        # ── Calibration ──────────────────────────────────────────────────────
        color_t = g.CameraType.kHeadColor
        depth_t = g.CameraType.kHeadDepth

        active_kp = cfg.active_keypoint_indices()
        active_names = [KEYPOINT_NAMES[i] for i in active_kp]
        print(f"[perception] keypoints ({cfg.keypoints}): {active_names}", flush=True)

        fx, fy, cx, cy, W, H = _get_intrinsics(cam, depth_t)
        print(f"[perception] intrinsics: fx={fx:.1f} fy={fy:.1f} "
              f"cx={cx:.1f} cy={cy:.1f} {W}×{H}", flush=True)

        # kHeadRGBDToHeadLink3: camera frame → head_link3
        rot_cam_to_head, trans_cam_to_head = _get_extrinsic_quat_trans(
            tf, g.SensorExtrinsicType.kHeadRGBDToHeadLink3
        )
        print(f"[perception] cam→head_link3 extrinsic: "
              f"{'ok' if rot_cam_to_head is not None else 'fallback (identity)'}", flush=True)

        # ── Fisheye + LiDAR calibration (skipped when disabled) ──────────────
        fisheye_cams: dict = {}   # name → (cam_type, K, D, w, h, rot_c2h, trans_c2h)
        lidar_rot_to_base: Optional[Rotation] = None
        lidar_trans_to_base: Optional[np.ndarray] = None
        _last_lidar_uvz: dict = {}   # name → (M, 3) projected LiDAR pixels
        _last_lidar_scan_t: float = 0.0
        _fisheye_activated: bool = False

        if cfg.use_fisheye:
            _fisheye_activated = _activate_fisheye(cam, cfg.robot_ip)
            for name in cfg.fisheye_cameras:
                cam_attr = _FISHEYE_CAMERA_ATTR.get(name)
                ext_attr = _FISHEYE_EXTRINSIC_ATTR.get(name)
                if cam_attr is None:
                    print(f"[fisheye] unknown camera name {name!r}", file=sys.stderr)
                    continue
                try:
                    cam_type = getattr(g.CameraType, cam_attr)
                    K, D, w, h = _get_fisheye_K_D(cam, cam_type)
                except Exception as e:
                    print(f"[fisheye] {name}: cam_type unavailable ({e})",
                          file=sys.stderr)
                    continue
                rot_c2h = trans_c2h = None
                if ext_attr:
                    try:
                        ext_type = getattr(g.SensorExtrinsicType, ext_attr)
                        rot_c2h, trans_c2h = _get_extrinsic_quat_trans(tf, ext_type)
                    except Exception as e:
                        print(f"[fisheye] {name}: extrinsic unavailable ({e})",
                              file=sys.stderr)
                fisheye_cams[name] = (cam_type, K, D, w, h, rot_c2h, trans_c2h)
                print(f"[fisheye] {name} ready: {w}×{h}  extrinsic="
                      f"{'ok' if rot_c2h is not None else 'identity'}", flush=True)

        if cfg.use_lidar:
            # kChassisFrontLidarToBaseLink gives lidar_sensor→base_link directly.
            lidar_rot_to_base, lidar_trans_to_base = _get_extrinsic_quat_trans(
                tf, g.SensorExtrinsicType.kChassisFrontLidarToBaseLink
            )
            if lidar_rot_to_base is not None:
                print(f"[perception] LiDAR extrinsic ok  t={lidar_trans_to_base}", flush=True)
            else:
                print("[perception] LiDAR extrinsic unavailable — using identity",
                      file=sys.stderr)
                lidar_rot_to_base = Rotation.identity()
                lidar_trans_to_base = np.zeros(3)

        # Refresh TF at startup; re-read periodically so head pose changes
        # (pan/tilt) are picked up.
        _last_tf_refresh = 0.0
        rot_head_to_base = trans_head_to_base = None

        period = 1.0 / cfg.rate_hz
        # Fisheye cameras run at ~15 fps — only run YOLO on them every other tick.
        fisheye_period = max(1.0 / 15.0, period * 2)
        _last_fisheye_t = 0.0
        next_tick = time.monotonic()

        print("[perception] streaming", flush=True)
        while not buf.stop:
            now = time.monotonic()

            # Refresh head_link3 TF every 0.5 s.
            if now - _last_tf_refresh > 0.5:
                rot_head_to_base, trans_head_to_base = _get_head_link3_tf(tf)
                _last_tf_refresh = now

            color_img = cam.get_latest_image(color_t, 200.0)
            depth_img = cam.get_latest_image(depth_t, 200.0)

            if color_img is None or depth_img is None:
                # Don't update last_frame_t so the watchdog eventually fires.
                next_tick += period
                sl = next_tick - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
                else:
                    next_tick = time.monotonic()
                continue

            color_bgr = _color_to_bgr(color_img)
            depth_m = _decode_depth(
                np.asarray(depth_img.data), depth_img.height, depth_img.width
            )

            # ── YOLO inference ────────────────────────────────────────────────
            detections = _run_yolo(model, color_bgr,
                                   conf=cfg.keypoint_conf_threshold,
                                   imgsz=cfg.yolo_imgsz)

            min_dist: Optional[float] = None
            annotated = color_bgr.copy()

            for kp_person, bbox in detections:
                dists = _project_keypoints_to_base_link(
                    kp_person, depth_m, fx, fy, cx, cy,
                    rot_cam_to_head, trans_cam_to_head,
                    rot_head_to_base, trans_head_to_base,
                    cfg.keypoint_conf_threshold,
                    active_indices=active_kp,
                )
                if not dists:
                    continue
                person_min = min(dists)
                if min_dist is None or person_min < min_dist:
                    min_dist = person_min

                # Draw bbox + distance label.
                x1, y1, x2, y2 = (int(v) for v in bbox)
                label = f"{person_min:.2f}m"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw active keypoints (green) and ignored ones (grey).
                for ki, (ku, kv, kconf) in enumerate(kp_person):
                    if kconf < cfg.keypoint_conf_threshold:
                        continue
                    pt = (int(ku), int(kv))
                    if ki in active_kp:
                        cv2.circle(annotated, pt, 4, (0, 255, 0), -1)
                    else:
                        cv2.circle(annotated, pt, 3, (100, 100, 100), -1)

            # Overlay safety zone indicator.
            _draw_safety_overlay(annotated, min_dist)

            # Log distance every 30 ticks (~1 s) so you can see depth is working.
            if int(time.monotonic()) % 30 == 0 and min_dist is not None:
                print(f"[perception] nearest human: {min_dist:.2f} m", flush=True)

            # ── Secondary path: fisheye cameras + LiDAR ──────────────────────
            new_fisheye_frames: dict = {}
            _run_fisheye_now = (time.monotonic() - _last_fisheye_t) >= fisheye_period

            # Pull a fresh LiDAR scan and project into all fisheye frames.
            if lidar is not None and rot_head_to_base is not None:
                try:
                    pc = lidar.get_latest_pointcloud(g.LidarType.kLidarFront, 100.0)
                    xyz_lidar = _parse_pointcloud(pc) if pc is not None else None
                    if xyz_lidar is not None and len(xyz_lidar) > 0:
                        xyz_base = lidar_rot_to_base.apply(xyz_lidar) + lidar_trans_to_base
                        _last_lidar_scan_t = time.monotonic()
                        for name, (cam_type, K, D, w, h, rot_c2h, trans_c2h) in fisheye_cams.items():
                            if rot_c2h is not None:
                                _last_lidar_uvz[name] = _project_lidar_to_fisheye_pixels(
                                    xyz_base, K, D, rot_c2h, trans_c2h,
                                    rot_head_to_base, trans_head_to_base, w, h,
                                )
                except Exception as e:
                    print(f"[perception] LiDAR error: {e}", file=sys.stderr)

            # Run YOLO on each active fisheye camera and look up LiDAR depth.
            lidar_fresh = (
                _last_lidar_scan_t > 0
                and (time.monotonic() - _last_lidar_scan_t) < cfg.lidar_stale_timeout_s
            )
            if _run_fisheye_now:
                _last_fisheye_t = time.monotonic()
            for name, (cam_type, K, D, w, h, rot_c2h, trans_c2h) in fisheye_cams.items():
                if not _run_fisheye_now:
                    # Reuse last annotated frame to avoid stale-blank tiles.
                    if name in buf.fisheye_bgr:
                        new_fisheye_frames[name] = buf.fisheye_bgr[name]
                    continue
                try:
                    fe_img = cam.get_latest_image(cam_type, 200.0)
                    if fe_img is None:
                        continue
                    fe_bgr = _color_to_bgr(fe_img)
                    fe_annotated = fe_bgr.copy()

                    fe_detections = _run_yolo(model, fe_bgr,
                                             conf=cfg.keypoint_conf_threshold,
                                             imgsz=cfg.yolo_imgsz_fisheye)
                    lidar_uvz = _last_lidar_uvz.get(name, np.zeros((0, 3), np.float32))

                    for kp_person, bbox in fe_detections:
                        # Annotate bbox on fisheye frame.
                        x1, y1, x2, y2 = (int(v) for v in bbox)
                        cv2.rectangle(fe_annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)

                        # Draw keypoints.
                        for ki, (ku, kv, kconf) in enumerate(kp_person):
                            if kconf < cfg.keypoint_conf_threshold:
                                continue
                            pt = (int(ku), int(kv))
                            color = (255, 165, 0) if ki in active_kp else (80, 80, 80)
                            cv2.circle(fe_annotated, pt, 4, color, -1)

                        # Compute 3-D distances only when LiDAR data is fresh.
                        if cfg.use_lidar and lidar_fresh and rot_c2h is not None:
                            dists = _fisheye_keypoints_to_base_link(
                                kp_person, lidar_uvz, K, D,
                                rot_c2h, trans_c2h,
                                rot_head_to_base, trans_head_to_base,
                                cfg.keypoint_conf_threshold, active_kp,
                                cfg.lidar_search_radius_px,
                            )
                            if dists:
                                person_min = min(dists)
                                if min_dist is None or person_min < min_dist:
                                    min_dist = person_min
                                label = f"{person_min:.2f}m"
                                cv2.putText(fe_annotated, label,
                                            (x1, max(y1 - 6, 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 165, 0), 2)

                    source_tag = f"FISHEYE {name.upper()}"
                    if not (cfg.use_lidar and lidar_fresh):
                        source_tag += " (no LiDAR)"
                    _draw_safety_overlay(fe_annotated, min_dist, tag=source_tag)
                    new_fisheye_frames[name] = fe_annotated

                except Exception as e:
                    print(f"[perception] fisheye {name} error: {e}", file=sys.stderr)

            with buf.lock:
                buf.color_bgr = color_bgr
                buf.depth_m = depth_m
                buf.annotated_bgr = annotated
                buf.distance_m = min_dist
                buf.last_frame_t = time.monotonic()
                buf.fisheye_bgr.update(new_fisheye_frames)

            next_tick += period
            sl = next_tick - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_tick = time.monotonic()

    except Exception as e:
        print(f"[perception] worker crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            if cam is not None:
                cam.close_camera()
        except Exception:
            pass
        try:
            g.gdk_release()
        except Exception:
            pass
        if _fisheye_activated:
            _deactivate_fisheye(cfg.robot_ip)
        buf.stop = True
        print("[perception] gdk released", flush=True)


def _draw_safety_overlay(
    frame: np.ndarray,
    dist: Optional[float],
    tag: str = "",
) -> None:
    """Draw a coloured banner at the top of the frame indicating safety state."""
    from safety_logic import STOP_DISTANCE_M, SLOWDOWN_DISTANCE_M  # type: ignore

    if dist is None:
        color, text = (0, 200, 0), "CLEAR (no human)"
    elif dist <= STOP_DISTANCE_M:
        color, text = (0, 0, 255), f"STOP  {dist:.2f} m"
    elif dist <= SLOWDOWN_DISTANCE_M:
        color, text = (0, 165, 255), f"SLOW  {dist:.2f} m"
    else:
        color, text = (0, 200, 0), f"CLEAR {dist:.2f} m"

    if tag:
        text = f"[{tag}] {text}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), color, -1)
    cv2.putText(frame, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ── MJPEG helpers ─────────────────────────────────────────────────────────────

def _colorize_depth(depth_m: np.ndarray,
                    d_min: float = 0.3, d_max: float = 6.0) -> np.ndarray:
    d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, d_max)
    d = np.clip(d, d_min, d_max)
    norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    mask_inv = ~(np.isfinite(depth_m) & (depth_m > 0))
    color[mask_inv] = (0, 0, 0)
    return color


def _mjpeg_gen(buf: _FrameBuffer, kind: str,
               quality: int = 75, rate_hz: float = 15.0):
    """Yield multipart JPEG chunks for an MJPEG response."""
    period = 1.0 / rate_hz
    boundary = b"--frame"
    while not buf.stop:
        with buf.lock:
            if kind == "annotated":
                frame = buf.annotated_bgr
            elif kind == "depth":
                frame = buf.depth_m
            else:
                frame = buf.color_bgr
            frame = None if frame is None else (
                frame.copy() if kind != "depth" else frame.copy()
            )

        if frame is None:
            time.sleep(0.05)
            continue

        if kind == "depth":
            vis = _colorize_depth(frame)
        else:
            vis = frame
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            time.sleep(period)
            continue

        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpg)).encode()
            + b"\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )
        time.sleep(period)


# ── Flask app ─────────────────────────────────────────────────────────────────

def _make_flask_app(
    buf: _FrameBuffer,
    cfg: PerceptionConfig,
    safety: "SafetyController | None" = None,
) -> Flask:
    from safety_logic import SafetyController  # noqa: F401 — used in type hint above

    app = Flask(__name__)

    @app.route("/")
    def index():
        # Original per-camera tiles — unchanged stream URLs.
        fisheye_tiles = "".join(
            f'<div><h2>fisheye {name} (YOLO + LiDAR depth)</h2>'
            f'<img src="/stream/fisheye/{name}"></div>'
            for name in cfg.fisheye_cameras
        ) if cfg.use_fisheye else ""

        depth_range = (
            f"{cfg.depth_vis_min_m:.1f} m → {cfg.depth_vis_max_m:.1f} m"
        )

        return (
            "<html><head><title>safety_backend live</title>"
            "<style>"
            "body{font-family:sans-serif;background:#111;color:#eee;"
            "margin:0;padding:1em}"
            "h1{margin:.2em 0 .6em}"
            "h2{margin:.2em 0;font-size:.95em;color:#aaa}"
            "img{max-width:48vw;border:1px solid #444;display:block}"
            ".row{display:flex;gap:1em;flex-wrap:wrap}"
            "#status-bar{display:flex;align-items:center;gap:1.5em;"
            "background:#1a1a1a;border:1px solid #333;border-radius:10px;"
            "padding:.7em 1.2em;margin-bottom:1em}"
            "#traffic-light{display:flex;flex-direction:column;gap:6px;"
            "background:#111;border:2px solid #444;border-radius:20px;"
            "padding:8px;width:44px}"
            ".bulb{width:28px;height:28px;border-radius:50%;"
            "background:#2a2a2a;border:1px solid #444;transition:background .2s}"
            ".bulb.on-red{background:#f33;box-shadow:0 0 12px #f33}"
            ".bulb.on-yellow{background:#fc0;box-shadow:0 0 12px #fc0}"
            ".bulb.on-green{background:#3f6;box-shadow:0 0 12px #3f6}"
            "#state-label{font-size:2em;font-weight:bold;letter-spacing:.1em}"
            "#state-label.STOP{color:#f33}"
            "#state-label.SLOW{color:#fc0}"
            "#state-label.CLEAR{color:#3f6}"
            "#dist-label{font-size:.9em;color:#888}"
            "#pipeline-label{font-size:.8em;color:#666;margin-top:.25em}"
            "#latch-badge{display:none;font-size:.75em;color:#f88;"
            "border:1px solid #f55;border-radius:4px;padding:2px 6px;margin-top:.3em}"
            "#latch-badge.on{display:inline-block}"
            "#reset-btn{margin-left:auto;padding:.5em 1em;font-size:.9em;"
            "background:#333;color:#eee;border:1px solid #555;border-radius:6px;"
            "cursor:pointer}"
            "#reset-btn:hover{background:#444}"
            "#reset-btn:disabled{opacity:.4;cursor:not-allowed}"
            "#vfactor-bar-wrap{flex:1;min-width:120px}"
            "#vfactor-label{font-size:.75em;color:#666;margin-bottom:3px}"
            "#vfactor-bg{background:#222;border-radius:4px;height:10px;overflow:hidden}"
            "#vfactor-fill{height:100%;width:0%;transition:width .3s,background .3s}"
            "</style></head><body>"
            "<h1>safety_backend — live perception</h1>"
            # Traffic-light status (added above cameras — does not replace them).
            "<div id=status-bar>"
            "<div id=traffic-light>"
            '<div class=bulb id=bulb-red></div>'
            '<div class=bulb id=bulb-yellow></div>'
            '<div class=bulb id=bulb-green></div>'
            "</div>"
            '<div><div id=state-label>—</div>'
            '<div id=dist-label>distance: —</div>'
            '<div id=pipeline-label>raw — · filtered — · latched —</div>'
            '<span id=latch-badge>LATCHED — press Clear STOP</span></div>'
            '<div id=vfactor-bar-wrap>'
            '<div id=vfactor-label>velocity factor (effective)</div>'
            '<div id=vfactor-bg><div id=vfactor-fill></div></div>'
            "</div>"
            '<button type=button id=reset-btn disabled>Clear STOP</button>'
            "</div>"
            # Original camera grid — all MJPEG streams preserved.
            '<div class=row>'
            "<div><h2>head RGB-D — YOLO + safety state</h2>"
            '<img src="/stream/annotated"></div>'
            f"<div><h2>head RGB-D — depth ({depth_range})</h2>"
            '<img src="/stream/depth"></div>'
            + fisheye_tiles +
            "</div>"
            "<script>"
            "function applyState(d){"
            "var eff=d.effective_state||d.state;"
            "var l=document.getElementById('state-label');"
            "l.textContent=eff;l.className=eff;"
            "document.getElementById('bulb-red').className='bulb'+(eff==='STOP'?' on-red':'');"
            "document.getElementById('bulb-yellow').className='bulb'+(eff==='SLOW'?' on-yellow':'');"
            "document.getElementById('bulb-green').className='bulb'+(eff==='CLEAR'?' on-green':'');"
            "document.getElementById('dist-label').textContent="
            "d.distance_m!=null?'distance: '+d.distance_m.toFixed(2)+' m':'distance: no human';"
            "document.getElementById('pipeline-label').textContent="
            "'raw '+d.raw_state+' · filtered '+d.filtered_state"
            "+(d.operator_latched?' · latched YES':' · latched no');"
            "var badge=document.getElementById('latch-badge');"
            "badge.className=d.operator_latched?'on':'';"
            "var btn=document.getElementById('reset-btn');"
            "btn.disabled=!d.operator_latched;"
            "var f=document.getElementById('vfactor-fill');"
            "var v=d.velocity_factor!=null?d.velocity_factor:0;"
            "f.style.width=Math.round(v*100)+'%';"
            "f.style.background=eff==='STOP'?'#f33':eff==='SLOW'?'#fc0':'#3f6';}"
            "document.getElementById('reset-btn').onclick=function(){"
            "fetch('/api/reset',{method:'POST'}).then(r=>r.json())"
            ".then(function(){poll();});};"
            "function poll(){fetch('/healthz').then(r=>r.json())"
            ".then(applyState).catch(function(){applyState({state:'—',"
            "effective_state:'—',raw_state:'—',filtered_state:'—',"
            "operator_latched:false,distance_m:null,velocity_factor:0});});"
            "setTimeout(poll,200);}"
            "poll();</script>"
            "</body></html>"
        )

    @app.route("/stream/annotated")
    def stream_annotated():
        return Response(
            _mjpeg_gen(buf, "annotated", cfg.mjpeg_jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream/color")
    def stream_color():
        return Response(
            _mjpeg_gen(buf, "color", cfg.mjpeg_jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream/depth")
    def stream_depth():
        return Response(
            _mjpeg_gen(buf, "depth", cfg.mjpeg_jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream/fisheye/<name>")
    def stream_fisheye(name: str):
        def gen():
            boundary = b"--frame"
            period = 1.0 / 15.0   # fisheye cameras run at ~15 fps
            while not buf.stop:
                with buf.lock:
                    frame = buf.fisheye_bgr.get(name)
                    frame = None if frame is None else frame.copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, cfg.mjpeg_jpeg_quality])
                if ok:
                    yield (
                        boundary
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode()
                        + b"\r\n\r\n"
                        + jpg.tobytes()
                        + b"\r\n"
                    )
                time.sleep(period)

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/healthz")
    def healthz():
        from safety_logic import decide as _decide
        now = time.monotonic()
        with buf.lock:
            age = None if buf.last_frame_t == 0 else round(now - buf.last_frame_t, 3)
            dist = buf.distance_m
        snap = safety.latest if safety is not None else None
        if snap is not None:
            body = snap.as_dict()
            body["state"] = snap.effective.state.value  # backward compat
            body["ok"] = (
                not buf.stop and age is not None and age < cfg.stale_timeout_s
            )
            body["frame_age_s"] = age
            body["stopped"] = buf.stop
            return jsonify(body)
        decision = _decide(dist, cfg.stop_m, cfg.slow_m)
        return jsonify(
            ok=not buf.stop and age is not None and age < cfg.stale_timeout_s,
            frame_age_s=age,
            distance_m=dist,
            state=decision.state.value,
            effective_state=decision.state.value,
            raw_state=decision.state.value,
            filtered_state=decision.state.value,
            operator_latched=False,
            velocity_factor=round(decision.velocity_factor, 3),
            stopped=buf.stop,
        )

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        if safety is None:
            return jsonify(ok=False, error="safety controller not attached"), 503
        safety.reset_operator_latch()
        snap = safety.latest
        return jsonify(
            ok=True,
            operator_latched=safety.operator_latched,
            effective_state=(
                snap.effective.state.value if snap is not None else "CLEAR"
            ),
        )

    return app


# ── Public API ────────────────────────────────────────────────────────────────

class PerceptionModule:
    """Start GDK + YOLO in a background thread and serve MJPEG on a side port.

    Usage::

        cfg = PerceptionConfig()
        pm = PerceptionModule(cfg)
        pm.start()          # blocks ~5 s for GDK settle, then returns
        ...
        d = pm.latest_distance()   # None=no human, float=distance_m
        ...
        pm.close()
    """

    def __init__(
        self,
        cfg: PerceptionConfig | None = None,
        safety: "SafetyController | None" = None,
    ) -> None:
        self._cfg = cfg or PerceptionConfig()
        self._safety = safety
        self._buf = _FrameBuffer()
        self._worker_thread: threading.Thread | None = None
        self._flask_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background perception worker and MJPEG server."""
        self._worker_thread = threading.Thread(
            target=_perception_worker,
            args=(self._buf, self._cfg),
            daemon=True,
            name="perception-worker",
        )
        self._worker_thread.start()

        app = _make_flask_app(self._buf, self._cfg, safety=self._safety)
        self._flask_thread = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0",
                port=self._cfg.mjpeg_port,
                threaded=True,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="mjpeg-server",
        )
        self._flask_thread.start()
        print(f"[perception] MJPEG server → http://0.0.0.0:{self._cfg.mjpeg_port}",
              flush=True)

    def latest_distance(self) -> Optional[float]:
        """Return the nearest detected human distance in metres.

        Returns:
            ``None`` — no human detected (CLEAR).
            ``float`` — distance in metres.

        Raises:
            StaleSensorError — last frame older than stale_timeout_s.
        """
        if self._buf.stop:
            raise StaleSensorError("perception worker has stopped")

        with self._buf.lock:
            age = time.monotonic() - self._buf.last_frame_t
            if self._buf.last_frame_t > 0 and age > self._cfg.stale_timeout_s:
                raise StaleSensorError(
                    f"last frame was {age:.2f}s ago "
                    f"(threshold {self._cfg.stale_timeout_s}s)"
                )
            return self._buf.distance_m

    def close(self) -> None:
        """Signal the worker to stop and wait for it to release GDK."""
        self._buf.stop = True
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=8.0)
