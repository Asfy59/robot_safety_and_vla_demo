#!/usr/bin/env bash
# Publish https://github.com/Asfy59/robot_safety_and_vla_demo (part1 + part2 only).
#
# One-time: deploy key ~/.ssh/id_ed25519_asfy59_demo.pub on the repo (write access).
# SSH host alias: github-asfy59 → git@github.com:Asfy59/...
#
# Usage:
#   /home/roboservice/asfand_challenge/part1/scripts/push_demo_repo.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

AUTHOR_NAME="${GIT_AUTHOR_NAME:-Asfy59}"
AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-Asfy59@users.noreply.github.com}"
CURRENT_BRANCH="$(git branch --show-current)"

cleanup() {
  git checkout "$CURRENT_BRANCH" 2>/dev/null || git checkout main
}
trap cleanup EXIT

git remote remove demo 2>/dev/null || true
git remote add demo git@github-asfy59:Asfy59/robot_safety_and_vla_demo.git

echo "Building fresh demo-release (part1 + part2, single Asfy59 commit)..."
git branch -D demo-release 2>/dev/null || true
git checkout --orphan demo-release
git reset

# Ensure eval dirs are never staged (even if .gitignore was not committed yet).
git add part1 part2
git rm -r --cached -f part1/outputs part2/outputs 2>/dev/null || true
git reset HEAD -- part1/outputs part2/outputs 2>/dev/null || true

cat > README.md << 'EOF'
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
EOF
git add README.md

if git ls-files | grep -qE '(^|/)outputs/|\.mp4$'; then
  echo "ERROR: refusing to push outputs/ or .mp4 files" >&2
  git ls-files | grep -E '(^|/)outputs/|\.mp4$' >&2
  exit 1
fi

git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" \
  commit -m "$(cat <<'EOF'
Add Part 1 safety backend and Part 2 SmolVLA LIBERO demo.

Part 1: GDK perception, YOLO pose, hysteresis + operator STOP latch, MJPEG UI.
Part 2: smolvla_libero eval on libero_object (71% success), run_demo + viewer.
EOF
)"

echo "Force-pushing demo-release → main..."
git push -f demo demo-release:main

echo "Done: https://github.com/Asfy59/robot_safety_and_vla_demo"
