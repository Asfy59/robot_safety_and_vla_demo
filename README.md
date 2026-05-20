# Robot Safety Backend + VLA Demo

Robotics engineer challenge submission (RoboService).

| Folder | What it is |
|--------|------------|
| **part1/** | Standalone safety backend — human detection (RGB-D + fisheye/LiDAR), 30 Hz ROS2 `STOP`/`SLOW`/`CLEAR`, MJPEG dashboard, operator acknowledge |
| **part2/** | SmolVLA on LIBERO-Object (~71% success), eval scripts, episode viewer |

## Quick start — Part 1

```bash
cd part1
docker compose -f docker/docker-compose.yml up
# http://<host>:8080
```

See **part1/README.md** for fisheye activation, config, and tests.

## Quick start — Part 2

```bash
cd part2
# see part2/README.md for venv, eval, and viewer.py
```

## Note

Challenge docs (`docs/`) and eval video outputs live on the development machine only — not included in this repo.
