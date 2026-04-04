#!/usr/bin/env bash
# diagnose_conf_rrig6600b.sh - Run from Zeus
HOST="192.168.3.154"

echo "=== Conf file forensics on rrig6600b ==="
echo ""
echo "[1] Does file exist at all (as root)?"
ssh michael@$HOST "sudo ls -la /etc/cluster-boot-notify.conf 2>&1"

echo ""
echo "[2] Full /etc permissions snapshot (anything touch /etc/cluster* recently)?"
ssh michael@$HOST "sudo find /etc -maxdepth 1 -name 'cluster*' -ls 2>&1"

echo ""
echo "[3] What changed in /etc around the reboot time (07:15-07:25)?"
ssh michael@$HOST "sudo find /etc -maxdepth 1 -newer /var/log/syslog -ls 2>&1 | head -20"

echo ""
echo "[4] journalctl around reboot for any file ops?"
ssh michael@$HOST "journalctl -b 0 --since '07:15' --until '07:25' --no-pager 2>&1 | grep -i 'conf\|notify\|environ\|tee\|write' | head -20"

echo ""
echo "[5] bash history on rrig6600b — what ran before reboot?"
ssh michael@$HOST "tail -30 ~/.bash_history 2>&1"
