#!/bin/bash
# monitor_all.sh — Launch all monitoring views in gnome-terminal tabs
#
# Runs ON SER8 (not Zeus). Opens 5 tabs + ensures dashboard up.
#
# Tab 1: Progress Monitor        — rich terminal (progress_monitor.py on Zeus)
# Tab 2: Health Snapshot         — 5s worker counts + recent chunks + MEM + errors
# Tab 3: Netconsole              — kernel messages from all 3 rigs
# Tab 4: Page Memory             — 2s rig RAM snapshots (pre-crash leak detection)
# Tab 5: Crash Monitor           — 3s UP/DOWN polling, logs state changes
#
# Also launches web_dashboard.py on Zeus if not running.
#
# Uses nohup + gnome-terminal — NEVER tmux.
#
# Usage:  bash monitor_all.sh

set -u

echo "═════════════════════════════════════════════════════════════"
echo " PRNG Cluster — monitor_all.sh (5 tabs + dashboard)"
echo "═════════════════════════════════════════════════════════════"

# ---- 0. Web dashboard (background on Zeus) ---------------------------------
echo ""
echo "[0/6] Web dashboard..."
DASH_RUNNING=$(ssh rzeus "pgrep -f web_dashboard.py | head -1" 2>/dev/null)
if [ -n "$DASH_RUNNING" ]; then
    echo "  ✅ already running on Zeus (PID $DASH_RUNNING)"
else
    echo "  ⚠  not running — launching via nohup on Zeus"
    ssh rzeus "cd ~/distributed_prng_analysis && \
        source ~/venvs/torch/bin/activate && \
        fuser -k 5000/tcp 2>/dev/null; sleep 1; \
        nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 &"
    sleep 3
    echo "  ✅ launched"
fi
echo "      → http://45.32.131.224:5002  (VPS proxy to Zeus:5000)"

# ---- 1. Netconsole listener check ------------------------------------------
echo ""
echo "[1/6] Netconsole listener (systemd on Zeus)..."
NETC=$(ssh rzeus "systemctl is-active netconsole-listener.service" 2>/dev/null)
if [ "$NETC" = "active" ]; then
    echo "  ✅ active"
else
    echo "  ⚠  not active — start manually:  ssh rzeus 'sudo systemctl start netconsole-listener.service'"
fi

# ---- 2-6. Launch 5 gnome-terminal tabs -------------------------------------
echo ""
echo "[2-6/6] Launching 5 monitoring tabs..."

gnome-terminal \
  --tab --title="Progress Monitor" \
    -- bash -c 'ssh -t rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && python3 progress_monitor.py"; exec bash' \
  --tab --title="Health Snapshot" \
    -- bash -c '
while true; do
  clear
  LOG=$(ssh rzeus "ls -t ~/distributed_prng_analysis/logs/s*.log 2>/dev/null | grep -vE \"dashboard|netconsole|soak\" | head -1")
  echo "════════════════════════════════════════════════════════════"
  echo " HEALTH SNAPSHOT — $(date)"
  echo " Log: $LOG"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  echo "── Rig Workers ──"
  for r in rrig6600 rrig6600b rrig6600c; do
    n=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "$r" \
          "ps aux | grep pwc_worker | grep -v grep | wc -l" 2>/dev/null || echo "?")
    printf "  %-12s %s / 8 workers\n" "$r" "$n"
  done
  echo ""
  echo "── Zeus Coordinator ──"
  ssh rzeus "ps aux | grep -E \"watcher_agent|window_optimizer|persistent_worker_coord\" | grep -v grep | awk \"{printf \\\"  PID=%-7s CPU=%-5s %s\\n\\\", \\\$2, \\\$3, \\\$11}\"" 2>/dev/null
  echo ""
  echo "── Recent Chunks ──"
  ssh rzeus "grep Chunk $LOG 2>/dev/null | tail -5" 2>/dev/null
  echo ""
  echo "── MEM DEBUG (last 3) ──"
  ssh rzeus "grep \"\\[MEM\" $LOG 2>/dev/null | tail -3" 2>/dev/null
  echo ""
  echo "── Errors (last 3) ──"
  ssh rzeus "grep -iE \"error|traceback|fault|crash\" $LOG 2>/dev/null | tail -3" 2>/dev/null
  echo ""
  echo "── Trial Progress ──"
  ssh rzeus "grep -E \"NEW BEST|Trial [0-9]|PASS|FAIL\" $LOG 2>/dev/null | tail -5" 2>/dev/null
  echo ""
  echo "(refreshing every 5s — Ctrl+C to exit)"
  sleep 5
done; exec bash' \
  --tab --title="Netconsole" \
    -- bash -c 'ssh rzeus "tail -f ~/distributed_prng_analysis/logs/netconsole_all_rigs.log"; exec bash' \
  --tab --title="Page Memory" \
    -- bash -c '
while true; do
  clear
  echo "========== $(date) — Rig Memory Snapshot =========="
  for r in rrig6600 rrig6600b rrig6600c; do
    echo "=== $r ==="
    ssh -o ConnectTimeout=3 -o BatchMode=yes "$r" \
        "grep -E \"PageTables|MemFree|MemAvailable|Slab\" /proc/meminfo" 2>/dev/null \
      || echo "UNREACHABLE"
    echo ""
  done
  sleep 2
done; exec bash' \
  --tab --title="Crash Monitor" \
    -- bash -c '
LOG=$HOME/rig_crash_monitor_persistent.log
declare -A LAST_GOOD DOWN_STATE
SAMPLE=0
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Crash monitor started" | tee -a $LOG
while true; do
  SAMPLE=$((SAMPLE+1))
  for ENTRY in rrig6600:192.168.3.120 rrig6600b:192.168.3.154 rrig6600c:192.168.3.162; do
    NAME="${ENTRY%%:*}"
    HOST="${ENTRY##*:}"
    RESULT=$(ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$HOST" "grep PageTables /proc/meminfo && pgrep -c pwc_worker || echo 0" 2>/dev/null)
    if [ -z "$RESULT" ]; then
      if [ "${DOWN_STATE[$NAME]:-0}" != "1" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] [DOWN] $NAME — last good sample ${LAST_GOOD[$NAME]:-0}" | tee -a $LOG
        DOWN_STATE[$NAME]=1
      else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] [DOWN] $NAME still down..." | tee -a $LOG
      fi
    else
      [ "${DOWN_STATE[$NAME]:-0}" = "1" ] && echo "[$(date +"%Y-%m-%d %H:%M:%S")] [UP] $NAME back online!" | tee -a $LOG
      DOWN_STATE[$NAME]=0
      PT=$(echo "$RESULT" | grep PageTables | awk "{print \$2, \$3}")
      WK=$(echo "$RESULT" | tail -1)
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] [OK#$SAMPLE] $NAME | PageTables: $PT | workers=$WK" | tee -a $LOG
      LAST_GOOD[$NAME]=$SAMPLE
    fi
  done
  sleep 3
done; exec bash'

echo ""
echo "✅ All monitors launched. Dashboard: http://45.32.131.224:5002"
echo "✅ Crash monitor log: ~/rig_crash_monitor_persistent.log"
