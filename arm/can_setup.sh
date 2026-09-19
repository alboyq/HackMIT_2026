#!/usr/bin/env bash
# REVIEW ONLY. Do not run while the arm owner is away or while any control process is active.
set -euo pipefail
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
ip -details link show can0

