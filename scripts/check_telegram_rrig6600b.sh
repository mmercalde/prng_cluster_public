#!/usr/bin/env bash
# check_telegram_rrig6600b.sh
# rrig6600b didn't send its boot notification after the S159G reboot.
# This script diagnoses the service and sends a manual test.
# Run from: Zeus

HOST="192.168.3.154"
SSH="ssh michael@$HOST"

echo "=== rrig6600b Telegram Boot Notify Check ==="

echo ""
echo "[1] Service status:"
$SSH "systemctl is-enabled cluster-boot-notify.service 2>&1; systemctl is-active cluster-boot-notify.service 2>&1"

echo ""
echo "[2] Last service journal (boot):"
$SSH "journalctl -u cluster-boot-notify.service -b 0 --no-pager 2>&1 | tail -20"

echo ""
echo "[3] Manual Telegram test (live):"
$SSH "bash /usr/local/bin/cluster_notify.sh '[TEST] rrig6600b Telegram manual verify S159G' && echo 'sent OK' || echo 'FAILED'"

echo ""
echo "[4] Boot notify script exists + executable:"
$SSH "ls -la /usr/local/bin/cluster_boot_notify.sh 2>&1; ls -la /usr/local/bin/cluster_notify.sh 2>&1"

echo ""
echo "[5] Config readable:"
$SSH "test -r /etc/cluster-boot-notify.conf && echo 'conf readable' || echo 'conf MISSING or unreadable'"
