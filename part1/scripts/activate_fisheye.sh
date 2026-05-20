#!/usr/bin/env bash
# Activate fisheye cameras on the robot.
#
# Run this ONCE from the host machine before starting the Docker container.
# The robot stays in develop mode until deactivate_fisheye.sh is called or
# the robot reboots.
#
# Usage:
#   ./scripts/activate_fisheye.sh [robot_ip]   (default: 10.42.1.101)
#
# Requires:
#   - SSH access to the robot (password: 1)
#   - sshpass   OR   pre-installed SSH key on robot

ROBOT_IP="${1:-10.42.1.101}"
ROBOT_USER="agi"
ROBOT_PASS="1"

_ssh() {
    if command -v sshpass &>/dev/null; then
        sshpass -p "$ROBOT_PASS" ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_IP" "$@"
    else
        ssh -o StrictHostKeyChecking=no "$ROBOT_USER@$ROBOT_IP" "$@"
    fi
}

CAM_CONFIG='{
  "cam0":  {"fps": "30", "name": "head_stereo_right",  "publish": true},
  "cam3":  {"fps": "30", "name": "head_stereo_left",   "publish": true},
  "cam5":  {"fps": "30", "name": "hand_left_color",    "publish": true},
  "cam7":  {"fps": "30", "name": "hand_right_color",   "publish": true},
  "cam10": {"fps": "15", "name": "head_right_fisheye", "publish": true},
  "cam11": {"fps": "15", "name": "head_left_fisheye",  "publish": true},
  "cam12": {"fps": "15", "name": "head_back_fisheye",  "publish": true},
  "cam14": {"fps": "30", "name": "head_depth",         "publish": true},
  "cam15": {"fps": "30", "name": "head_color",         "publish": true}
}'

echo "[fisheye] Writing cam_config.json to robot..."
echo "$CAM_CONFIG" | _ssh "mkdir -p /data/gdk && cat > /data/gdk/cam_config.json"
if [ $? -ne 0 ]; then
    echo "[fisheye] ERROR: Failed to write cam_config.json"
    exit 1
fi
echo "[fisheye] cam_config.json written."

echo "[fisheye] Switching robot to develop mode..."
_ssh 'source /home/agi/app/env.sh && /home/agi/app/bin/mode_switch --mode develop'
if [ $? -ne 0 ]; then
    echo "[fisheye] ERROR: mode_switch to develop failed"
    exit 1
fi

echo "[fisheye] Waiting 5s for fisheye DDS topics to come up..."
sleep 5
echo "[fisheye] Done. Robot is now in develop mode."
echo "[fisheye] Run ./scripts/deactivate_fisheye.sh when finished."
