#!/bin/bash
# Source ROS2 + GDK env, then exec the container CMD.
#
# Auto-detects the local IP on the robot subnet (10.42.1.* wired or
# 192.168.10.*/WiFi) and sets LOCATOR_IP + AORTA_DISCOVERY_URI if they
# haven't already been set via docker-compose environment. The compose
# file sets safe defaults; this script overrides them only when a better
# interface is found.

set -e
source /opt/ros/humble/setup.bash

# ── GDK env.sh ────────────────────────────────────────────────────────────────
# Extends LD_LIBRARY_PATH / PYTHONPATH. Sourced silently — noise about
# missing IPs is harmless during offline runs.
if [ -f /root/.cache/agibot/app/env.sh ]; then
    set +e
    source /root/.cache/agibot/app/env.sh >/dev/null 2>&1 || true
    set -e
fi

# ── Robot IP auto-detection ───────────────────────────────────────────────────
# Priority 1: wired Ethernet (10.42.1.* subnet, lower latency / more reliable DDS)
wired_ip=$(ip -o -4 addr list 2>/dev/null | awk '/10\.42\.1\./{print $4}' | cut -d/ -f1 | head -1)
if [ -n "$wired_ip" ]; then
    export LOCATOR_IP="$wired_ip"
    export AORTA_DISCOVERY_URI="http://10.42.1.101:2379"
    echo "[entrypoint] wired connection: LOCATOR_IP=$LOCATOR_IP"
else
    # Priority 2: WiFi on the same 192.168.10.* subnet as the robot.
    wifi_ip=$(ip -o -4 addr list 2>/dev/null | awk '/192\.168\.10\./{print $4}' | cut -d/ -f1 | head -1)
    if [ -n "$wifi_ip" ]; then
        export LOCATOR_IP="$wifi_ip"
        export AORTA_DISCOVERY_URI="http://192.168.10.163:2379"
        echo "[entrypoint] WiFi connection: LOCATOR_IP=$LOCATOR_IP"
    else
        echo "[entrypoint] WARNING: no robot subnet found — GDK will run offline"
    fi
fi

export AORTA_DISPATCHER_THREAD_NUM="${AORTA_DISPATCHER_THREAD_NUM:-6}"

exec "$@"
