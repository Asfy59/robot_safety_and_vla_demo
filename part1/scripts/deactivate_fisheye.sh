#!/usr/bin/env bash
# Restore the robot to base mode after a fisheye session.
# Always run this before handing the robot back to the next user.
#
# Usage:
#   ./scripts/deactivate_fisheye.sh [robot_ip]   (default: 10.42.1.101)

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

echo "[fisheye] Switching robot back to base mode..."
_ssh 'source /home/agi/app/env.sh && /home/agi/app/bin/mode_switch --mode base'
if [ $? -ne 0 ]; then
    echo "[fisheye] ERROR: mode_switch to base failed"
    exit 1
fi
echo "[fisheye] Robot is back in base mode."
