#!/bin/bash
# monitor_all.sh v4 — 7 separate gnome-terminal WINDOWS (not tabs)
#
# Runs ON SER8. Opens 7 independent windows you can arrange however:
#   Window 1: Progress Monitor   — rich terminal (progress_monitor.py)
#   Window 2: Live Log           — tail -f of active run log
#   Window 3: Health Snapshot    — workers + chunks + trials (5s)
#   Window 4: S163 MEM Debug     — pool/VmRSS/threshold breaches (10s)
#   Window 5: Netconsole         — kernel messages from all 3 rigs
#   Window 6: Page Memory        — rig RAM snapshots (2s)
#   Window 7: Crash Monitor      — 3s UP/DOWN polling, persistent log
#
# Also auto-launches web_dashboard.py on Zeus.
#
# Each window can be moved/resized/minimized independently.
# Close a window anytime — pipeline on Zeus keeps running (nohup).
#
# Uses nohup + gnome-terminal --window — NEVER tmux.
#
# Usage:  bash monitor_all.sh

set -u

echo "═════════════════════════════════════════════════════════════"
echo " PRNG Cluster — monitor_all.sh v4 (7 windows + dashboard)"
echo "═════════════════════════════════════════════════════════════"

# ---- 0. Web dashboard -----------------------------------------------------
echo ""
echo "[0/8] Web dashboard..."
DASH_RUNNING=$(ssh rzeus "pgrep -f web_dashboard.py | head -1" 2>/dev/null)
if [ -n "$DASH_RUNNING" ]; then
    echo "  ✅ already running on Zeus (PID $DASH_RUNNING)"
else
    echo "  ⚠  launching via nohup on Zeus"
    ssh rzeus "cd ~/distributed_prng_analysis && \
        source ~/venvs/torch/bin/activate && \
        fuser -k 5000/tcp 2>/dev/null; sleep 1; \
        nohup python3 web_dashboard.py > logs/dashboard.log 2>&1 &"
    sleep 3
    echo "  ✅ launched"
fi
echo "      → http://45.32.131.224:5002"

# ---- 1. Netconsole listener check ------------------------------------------
echo ""
echo "[1/8] Netconsole listener..."
NETC=$(ssh rzeus "systemctl is-active netconsole-listener.service" 2>/dev/null)
if [ "$NETC" = "active" ]; then
    echo "  ✅ active"
else
    echo "  ⚠  not active — start with:  ssh rzeus 'sudo systemctl start netconsole-listener.service'"
fi

# ---- 2. Resolve current active log (for Live Log window) -------------------
ACTIVE_LOG=$(ssh rzeus "ls -t ~/distributed_prng_analysis/logs/s*.log 2>/dev/null | grep -vE \"dashboard|netconsole|soak\" | head -1")
echo ""
echo "📋 Active log: ${ACTIVE_LOG:-none detected}"

# ---- 3-8. Launch 7 separate gnome-terminal WINDOWS -------------------------
echo ""
echo "[2-8/8] Launching 7 monitoring windows..."

# Window 1: Progress Monitor
gnome-terminal --window --title="1. Progress Monitor" --geometry=100x20+0+0 -- \
  bash -c 'ssh -t rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && python3 progress_monitor.py"; exec bash' &

# Window 2: Live Log tail
gnome-terminal --window --title="2. Live Log" --geometry=160x25+0+400 -- \
  bash -c "ssh rzeus \"tail -f ${ACTIVE_LOG:-/dev/null}\"; exec bash" &

# Window 3: Health Snapshot
gnome-terminal --window --title="3. Health Snapshot" --geometry=100x35+700+0 -- \
  bash -c '
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
  echo "── Errors (last 3) ──"
  ssh rzeus "grep -iE \"error|traceback|fault|crash\" $LOG 2>/dev/null | tail -3" 2>/dev/null
  echo ""
  echo "── Trial Progress ──"
  ssh rzeus "grep -E \"NEW BEST|Trial [0-9]|PASS|FAIL\" $LOG 2>/dev/null | tail -5" 2>/dev/null
  echo ""
  echo "(refreshing every 5s — Ctrl+C to exit)"
  sleep 5
done; exec bash' &

# Window 4: S163 MEM Debug
gnome-terminal --window --title="4. S163 MEM Debug" --geometry=100x35+700+500 -- \
  bash -c '
while true; do
  clear
  LOG=$(ssh rzeus "ls -t ~/distributed_prng_analysis/logs/s*.log 2>/dev/null | grep -vE \"dashboard|netconsole|soak\" | head -1")
  echo "════════════════════════════════════════════════════════════"
  echo " S163 MEM DEBUG — $(date)"
  echo " Log: $LOG"
  echo " free_all_blocks() removed (TB Option B) | Sample every 25 chunks"
  echo "════════════════════════════════════════════════════════════"
  echo ""
  MEM_DEBUG_VAL=$(ssh rzeus "ps aux | grep window_optimizer | grep -v grep | head -1 | grep -oP \"S163_MEM_DEBUG=\\K[0-9]\"" 2>/dev/null)
  if [ "$MEM_DEBUG_VAL" = "1" ]; then
    echo "  ✅ S163_MEM_DEBUG=1 (instrumentation ENABLED)"
  else
    echo "  ⚠  S163_MEM_DEBUG not set — threshold breaches only"
  fi
  echo ""
  echo "── MEM Samples (last 10, every 25 chunks) ──"
  ssh rzeus "grep \"\\[MEM chunk=\" $LOG 2>/dev/null | tail -10" 2>/dev/null
  echo ""
  echo "── Threshold Breaches (pool_used > 200MB) ──"
  ssh rzeus "grep \"\\[MEM WARNING\" $LOG 2>/dev/null | tail -10" 2>/dev/null
  echo ""
  echo "── Pool Growth Trend ──"
  ssh rzeus "grep \"\\[MEM chunk=\" $LOG 2>/dev/null | tail -10 | grep -oE \"pool_used=[0-9]+MB\"" 2>/dev/null
  echo ""
  echo "── Instrumentation Errors ──"
  ssh rzeus "grep \"MEM instrumentation error\" $LOG 2>/dev/null | tail -3" 2>/dev/null
  echo ""
  echo "(refreshing every 10s — Ctrl+C to exit)"
  sleep 10
done; exec bash' &

# Window 5: Netconsole
gnome-terminal --window --title="5. Netconsole" --geometry=140x25+1400+0 -- \
  bash -c 'ssh rzeus "tail -f ~/distributed_prng_analysis/logs/netconsole_all_rigs.log"; exec bash' &

# Window 6: Page Memory
gnome-terminal --window --title="6. Page Memory" --geometry=100x30+1400+500 -- \
  bash -c '
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
done; exec bash' &

# Window 7: Crash Monitor
gnome-terminal --window --title="7. Crash Monitor" --geometry=140x30+300+800 -- \
  bash -c '
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
done; exec bash' &

sleep 2
echo ""
echo "✅ 7 monitoring windows launched. Dashboard: http://45.32.131.224:5002"
echo "✅ Crash monitor log: ~/rig_crash_monitor_persistent.log"
