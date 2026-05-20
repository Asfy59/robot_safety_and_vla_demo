# Part I — Safety Backend

Detects humans near the robot and publishes safety decisions at **30 Hz**:

| State | Condition | `velocity_factor` |
|---|---|---|
| `STOP` | human ≤ 2.0 m from `base_link` | 0.0 |
| `SLOW` | 2.0 m < human ≤ 4.0 m | linear ramp 0.0 → 1.0 |
| `CLEAR` | human > 4.0 m or no detection | 1.0 |

---

## Quickstart

```bash
cd part1/docker
docker compose up --build
```

Open the live annotated stream: **http://localhost:8080/**

Stop with `Ctrl-C`.

---

## Configuration

All parameters live in **`config.yaml`** — no code changes or CLI flags needed.

```yaml
safety:
  stop_distance_m: 2.0        # hard-stop threshold (metres from base_link)
  slowdown_distance_m: 4.0    # outer edge of slowdown zone

perception:
  rate_hz: 30.0               # decision + frame-grab rate
  stale_timeout_s: 1.0        # fail-safe timeout (→ STOP if exceeded)
  yolo_model: "yolov8n-pose.pt"
  keypoint_conf_threshold: 0.3
  keypoints: "core"           # see Keypoint Presets below

mjpeg:
  port: 8080
  jpeg_quality: 75

dev:
  offline: false              # true = scripted distance, no robot needed
```

Edit `config.yaml` on the host; the bind-mount means changes take effect on
the next `docker compose restart` without rebuilding.

### Keypoint Presets

| Preset | Indices | Body parts used | Use case |
|---|---|---|---|
| `"core"` *(default)* | 5,6,7,8,11,12,13,14 | shoulders, elbows, hips, knees | Best noise/sensitivity balance |
| `"torso"` | 5,6,11,12 | shoulders + hips only | Busy/noisy environments |
| `"all"` | 0–16 | all 17 COCO keypoints | Max sensitivity (lab/demo) |
| `[5,6,11,12]` | custom list | any combination | Fine-tuned deployments |

COCO-17 index reference:
```
Head:    0=nose  1=left_eye  2=right_eye  3=left_ear  4=right_ear
Torso:   5=left_shoulder   6=right_shoulder
        11=left_hip        12=right_hip
Arms:    7=left_elbow      8=right_elbow
         9=left_wrist*    10=right_wrist*    (* excluded from "core")
Legs:   13=left_knee      14=right_knee
        15=left_ankle*    16=right_ankle*    (* excluded from "core")
```

### Offline mode (no robot)

```bash
# Use the pre-built offline.yaml (sets dev.offline: true):
docker compose run safety_backend python3 safety_node.py --config offline.yaml
docker compose run safety_backend python3 safety_backend_gdk.py --config offline.yaml

# Or set dev.offline: true in config.yaml directly.
```

---

## Architecture

```
docker compose up
    └── entrypoint.sh                 source ROS2, auto-detect robot IP
        └── safety_node.py   ← DEFAULT entry point
              │
              ├── safety_config.py    load + validate config.yaml
              │
              ├── PerceptionModule    background thread
              │     ├── GDK Camera   kHeadColor + kHeadDepth  @ 30 fps
              │     ├── YOLOv8-pose  person detection + keypoints
              │     ├── TF pipeline  keypoint → camera → head_link3 → base_link
              │     ├── Watchdog     stale frame > 1s → StaleSensorError → STOP
              │     └── Flask :8080  MJPEG live stream
              │
              ├── safety_logic.py     decide(distance_m) → STOP/SLOW/CLEAR
              │
              └── rclpy 30 Hz timer
                    ├── /safety/decision  (std_msgs/String, JSON)
                    └── /safety/stop      (std_msgs/Bool)
```

### Alternative entry points

| Command | When to use |
|---|---|
| `python3 safety_node.py` | Default — ROS2 topics + MJPEG |
| `python3 safety_backend_gdk.py` | GDK-only, no ROS2 (stdout decisions) |
| `python3 safety_loop.py` | No Docker/GDK/ROS2 — pure logic smoke test |

---

## Sensor Choice & Justification

**Primary: `kHeadColor` + `kHeadDepth`** (640×400 @ ~30 fps, always-on)

- Synchronized RGB + metric depth in a single API call pair — no stereo
  matching or LiDAR post-processing required.
- At 30 fps and 1.4 m/s walking speed the system gets ≥1.5 s of warning
  before a person reaches the 2 m hard-stop boundary.
- Real intrinsics and `kHeadRGBDToHeadLink3` extrinsic from GDK give a
  correct 3-D transform without manual calibration.

**Blind spots:** the head camera covers ~70° in front of the robot. Sides and
back are unmonitored. Adding the back LiDAR or the three fisheye cameras
would close this gap.

### 3-D Transform Pipeline

```
YOLO keypoint (u, v, conf)
    │  depth[v, u] — uint16 mm → float32 m (3×3 median patch)
    ▼
Camera frame:  X = (u − cx) × d / fx
               Y = (v − cy) × d / fy
               Z = d
    │  kHeadRGBDToHeadLink3 extrinsic  (GDK, refreshed on startup)
    ▼
head_link3 frame
    │  base_link → head_link3 TF       (GDK, refreshed every 0.5 s)
    ▼
base_link frame  →  dist = ‖p‖  →  decide(dist)
```

---

## Fail-Safe Behaviour

| Condition | Response |
|---|---|
| Human ≤ 2.0 m | `STOP`, `velocity_factor = 0.0`, streamed at 30 Hz |
| Human 2.0–4.0 m | `SLOW`, `velocity_factor` ramps 0.0 → 1.0 |
| Human > 4.0 m or no detection | `CLEAR`, `velocity_factor = 1.0` |
| Human moves away | Auto-release — no manual reset needed |
| Camera frame older than 1.0 s | `STOP` (fail-safe) — logged as `[FAILSAFE]` |
| GDK worker thread crashes | `STOP` (fail-safe) |

---

## ROS2 Topics

| Topic | Type | Rate | Payload |
|---|---|---|---|
| `/safety/decision` | `std_msgs/String` | 30 Hz | `{"state", "velocity_factor", "distance_m", "t", "failsafe"}` |
| `/safety/stop` | `std_msgs/Bool` | 30 Hz | `true` = STOP. Doubles as heartbeat — silence means the node has died. |

```bash
# Inspect from inside the running container:
docker compose exec safety_backend bash
ros2 topic echo /safety/decision
ros2 topic hz   /safety/decision    # expect ~30.0 Hz
ros2 topic echo /safety/stop
```

---

## MJPEG Live Stream

| Endpoint | Description |
|---|---|
| `http://localhost:8080/` | Landing page — annotated + depth side-by-side |
| `http://localhost:8080/stream/annotated` | YOLO bboxes, skeleton (green = active keypoints, grey = ignored), safety state banner |
| `http://localhost:8080/stream/color` | Raw color feed |
| `http://localhost:8080/stream/depth` | Depth colourised 0.3 m (red) → 6 m (blue) |
| `http://localhost:8080/healthz` | `{"ok", "frame_age_s", "distance_m", "stopped"}` |

Accessible from any device on the same network:
`http://<workstation-ip>:8080/`

---

## Running the Tests

```bash
# Inside the container (no robot needed):
docker compose exec safety_backend bash
python3 -m pytest tests/ -v

# One-shot:
docker compose run --rm safety_backend python3 -m pytest tests/ -v
```

**55 tests** covering:

| Suite | What is tested |
|---|---|
| `TestDecideZones` | Zone boundaries at 0 m, 2.0 m, 3.0 m, 4.0 m, and beyond |
| `TestDecideContract` | Distance preserved, `velocity_factor` values |
| `TestAsDict` | JSON serialisation format |
| `TestAutoRelease` | Stateless — STOP lifts immediately when distance clears |
| `TestScriptedDistance` | Sine wave covers all three zones |
| `TestKeypointPresets` | Preset definitions and custom list validation |
| `TestFailSafe` | Stale sensor → STOP; recovery after stale |
| `TestStopSignal` | STOP fires at right distance, sustains, auto-releases |
| `TestVelocityFactor` | Ramp correctness at zone boundaries and midpoint |
| `TestLoopRate` | 30 Hz timing; jitter < 5 ms |
| `TestStopStreaming` | 60/60 consecutive ticks are STOP while in zone |

### Fail-safe simulation (no robot)

```bash
# Interactive scenario demos — print decisions to stdout at 30 Hz:
python3 tests/test_safety_loop.py --demo stale     # normal → stale → STOP
python3 tests/test_safety_loop.py --demo stop      # person walks in and out
python3 tests/test_safety_loop.py --demo slowdown  # person in slowdown zone
python3 tests/test_safety_loop.py --demo clear     # always clear
```

---

## Docker Quick Reference

```bash
# Build & start (ROS2 node, default):
docker compose up --build

# Start in background:
docker compose up -d

# Tail logs:
docker compose logs -f

# Open a shell in the running container:
docker compose exec safety_backend bash

# Rebuild after Dockerfile/requirements change:
docker compose build

# Stop and remove:
docker compose down
```

---

## File Layout

```
part1/
├── config.yaml               single source of truth for all parameters
├── offline.yaml              dev override: sets dev.offline: true
│
├── safety_config.py          YAML loader + validated AppConfig dataclasses
├── safety_logic.py           pure decision math: decide(distance_m) → STOP/SLOW/CLEAR
├── safety_perception.py      GDK RGB-D + YOLOv8-pose + TF + MJPEG server
├── safety_backend_gdk.py     GDK-first entry point (stdout only, no ROS2)
├── safety_node.py            ROS2 node (default container CMD)
├── safety_loop.py            minimal v1 loop: no Docker/GDK/ROS2 (pure logic test)
│
├── tests/
│   ├── test_safety_logic.py  unit tests: decision math + keypoint presets
│   └── test_safety_loop.py   unit tests: fail-safe, STOP rate, 30 Hz timing
│
├── tools/
│   ├── record.py             record head color+depth clips for offline replay
│   ├── preview.py            standalone MJPEG preview (no safety logic)
│   └── smoke_test.py         minimal GDK connectivity check
│
└── docker/
    ├── Dockerfile            FROM roboservice/ros2-humble-gdk:latest
    ├── docker-compose.yml    host networking, bind-mounts, port 8080
    ├── entrypoint.sh         source ROS2 + GDK; auto-detect robot IP
    └── requirements.txt      pyyaml, ultralytics, flask, scipy, opencv
```
