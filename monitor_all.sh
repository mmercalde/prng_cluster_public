#!/bin/bash
# monitor_all.sh v5 — 7 separate gnome-terminal WINDOWS (not tabs)
#
# Runs ON SER8. Opens 7 independent windows you can arrange however:
#   Window 1: Progress Monitor   — rich terminal (progress_monitor.py)
#   Window 2: Live Log           — tail -f of active run log
#   Window 3: Health Snapshot    — workers + chunks + trials (5s)
#   Window 4: Worker Heartbeats  — TB-spec heartbeat JSON pool stats (10s)
#   Window 5: Netconsole         — kernel messages from all 3 rigs
#   Window 6: Page Memory        — rig RAM snapshots (2s)
#   Window 7: Crash Monitor      — 3s UP/DOWN polling, persistent log
#
# Also auto-launches web_dashboard.py on Zeus (port 5002).
#
# Changes v5:
#   - Window 4: replaced log-based MEM_DEBUG with TB-spec heartbeat JSON
#     reads ~/worker_log_snapshots/worker_heartbeats/*.json on each rig
#   - Dashboard URL corrected from 5000 → 5002
#   - Crash monitor: fixed false DOWN at startup (now requires 2 consecutive
#     failures before declaring DOWN, skips DOWN on first poll)
#
# Each window can be moved/resized/minimized independently.
# Close a window anytime — pipeline on Zeus keeps running (nohup).
#
# Uses nohup + gnome-terminal --window — NEVER tmux.
#
# Usage:  bash monitor_all.sh

set -u

echo "═════════════════════════════════════════════════════════════"
echo " PRNG Cluster — monitor_all.sh v5 (7 windows + dashboard)"
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
        fuser -k 5002/tcp 2>/dev/null; sleep 1; \
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

# Window 1: Progress Monitor (v2 — continuous mode, trial context)
gnome-terminal --window --title="1. Progress Monitor" --geometry=100x22+0+0 -- \
  bash -c '
LOG_PATTERN=$(ssh rzeus "ls -t ~/distributed_prng_analysis/logs/s*.log 2>/dev/null | grep -vE \"dashboard|netconsole|soak\" | head -1 | xargs basename | sed \"s/_[0-9]*\\.log//\"" 2>/dev/null)
ssh -t rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  python3 progress_monitor.py --log-pattern \"${LOG_PATTERN:-s163}\" --trials 5"
exec bash' &

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

# Window 4: Worker Heartbeats (TB-spec — replaces log-based S163 MEM Debug)
gnome-terminal --window --title="4. Worker Heartbeats" --geometry=120x40+700+500 -- \
  bash -c '
HB_SCRIPT=$(cat <<'"'"'PYEOF'"'"'
import json, glob, os, sys
hb_dir = os.path.expanduser("~/worker_log_snapshots/worker_heartbeats")
files = sorted(glob.glob(os.path.join(hb_dir, "*.json")))
if not files:
    print("  (no heartbeat files yet — workers not yet started)")
    sys.exit(0)
for f in files:
    try:
        d = json.load(open(f))
        pool   = d.get("cupy_pool") or {}
        used   = pool.get("used_bytes")
        total  = pool.get("total_bytes")
        used_s  = f"{used//1024//1024}MB"  if used  is not None else "N/A"
        total_s = f"{total//1024//1024}MB" if total is not None else "N/A"
        state  = d.get("state", "?")
        job    = (d.get("job_id") or "idle")[:14]
        phase  = d.get("phase", "?")
        w      = d.get("worker_name", os.path.basename(f))[-24:]
        lkl    = d.get("last_kernel_launch_ts","")
        lkr    = d.get("last_kernel_return_ts","")
        err    = d.get("last_error","") or ""
        state_icon = {"pre_kernel":"⚡","post_kernel":"✅","idle":"💤",
                      "job_start":"🚀","result_sent":"📤","exception":"❌",
                      "connected":"🔗","init_start":"⏳","init_done":"🟢",
                      "shutdown":"🔴"}.get(state, "❓")
        print(f"  {state_icon} {w:<24} state={state:<12} pool={used_s:>6}/{total_s:<6}  job={job:<14}  phase={phase}")
        if err:
            print(f"    ⚠ ERROR: {err[:80]}")
    except Exception as e:
        print(f"  ⚠ {os.path.basename(f)}: {e}")
PYEOF
)
while true; do
  clear
  echo "════════════════════════════════════════════════════════════════════════"
  echo " WORKER HEARTBEATS (TB-spec) — $(date)"
  echo " Source: ~/worker_log_snapshots/worker_heartbeats/*.json on each rig"
  echo "════════════════════════════════════════════════════════════════════════"
  echo ""
  for rig in rrig6600 rrig6600b rrig6600c; do
    echo "── $rig ──"
    ssh -o ConnectTimeout=3 -o BatchMode=yes "$rig" \
      "python3 -c '$HB_SCRIPT'" 2>/dev/null || echo "  (unreachable)"
    echo ""
  done
  echo "── CuPy Pool Threshold Breaches (>200MB, from coordinator log) ──"
  LOG=$(ssh rzeus "ls -t ~/distributed_prng_analysis/logs/s*.log 2>/dev/null | grep -vE \"dashboard|netconsole|soak\" | head -1" 2>/dev/null)
  ssh rzeus "grep \"\\[MEM WARNING\" ${LOG:-/dev/null} 2>/dev/null | tail -5" 2>/dev/null || true
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
# v5 fix: requires 2 consecutive SSH failures before declaring DOWN
# to prevent false DOWN on startup or momentary SSH hiccup
gnome-terminal --window --title="7. Crash Monitor" --geometry=140x30+300+800 -- \
  bash -c '
LOG=$HOME/rig_crash_monitor_persistent.log
declare -A LAST_GOOD DOWN_STATE FAIL_COUNT
SAMPLE=0
echo "[$(date +"%Y-%m-%d %H:%M:%S")] [START] Crash monitor started (v5 — 2-strike DOWN)" | tee -a $LOG
while true; do
  SAMPLE=$((SAMPLE+1))
  for ENTRY in rrig6600:192.168.3.120 rrig6600b:192.168.3.154 rrig6600c:192.168.3.162; do
    NAME="${ENTRY%%:*}"
    HOST="${ENTRY##*:}"
    RESULT=$(ssh -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$HOST" "grep PageTables /proc/meminfo && pgrep -c pwc_worker || echo 0" 2>/dev/null)
    if [ -z "$RESULT" ]; then
      FAIL_COUNT[$NAME]=$(( ${FAIL_COUNT[$NAME]:-0} + 1 ))
      if [ "${FAIL_COUNT[$NAME]}" -ge 2 ]; then
        if [ "${DOWN_STATE[$NAME]:-0}" != "1" ]; then
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [DOWN] $NAME — last good sample ${LAST_GOOD[$NAME]:-never}" | tee -a $LOG
          DOWN_STATE[$NAME]=1
        else
          echo "[$(date +"%Y-%m-%d %H:%M:%S")] [DOWN] $NAME still down... (fail #${FAIL_COUNT[$NAME]})" | tee -a $LOG
        fi
      else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] [WARN] $NAME SSH fail #${FAIL_COUNT[$NAME]} — waiting for confirmation" | tee -a $LOG
      fi
    else
      FAIL_COUNT[$NAME]=0
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
echo "✅ 7 monitoring windows launched."
echo "✅ Dashboard:      http://45.32.131.224:5002"
echo "✅ Crash monitor:  ~/rig_crash_monitor_persistent.log"
