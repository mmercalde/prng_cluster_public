#!/bin/bash
# soak_monitor.sh — Resource monitoring for Phase 7 soak tests
# Usage: bash soak_monitor.sh [interval_seconds]
# Default interval: 30 seconds
#
# Run in a SEPARATE terminal during any soak test.
# Logs to logs/soak_monitor_YYYYMMDD_HHMMSS.log

set -euo pipefail

INTERVAL=${1:-30}
PROJECT=~/distributed_prng_analysis
LOGFILE="$PROJECT/logs/soak_monitor_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT/logs"

echo "========================================" | tee "$LOGFILE"
echo " SOAK MONITOR STARTED $(date)" | tee -a "$LOGFILE"
echo " Interval: ${INTERVAL}s" | tee -a "$LOGFILE"
echo " Log: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Record baseline
echo "=== BASELINE ===" | tee -a "$LOGFILE"
WATCHER_PID=$(pgrep -f "watcher_agent.py" | head -1 || echo "")
if [ -n "$WATCHER_PID" ]; then
    echo "WATCHER PID: $WATCHER_PID" | tee -a "$LOGFILE"
    BASELINE_RSS=$(ps -o rss= -p "$WATCHER_PID" 2>/dev/null || echo "0")
    BASELINE_RSS_MB=$((BASELINE_RSS / 1024))
    echo "Baseline RSS: ${BASELINE_RSS_MB} MB" | tee -a "$LOGFILE"
    BASELINE_FD=$(ls /proc/$WATCHER_PID/fd 2>/dev/null | wc -l || echo "0")
    echo "Baseline FDs: $BASELINE_FD" | tee -a "$LOGFILE"
else
    echo "WATCHER not yet running — will detect on first check" | tee -a "$LOGFILE"
    BASELINE_RSS_MB=0
    BASELINE_FD=0
fi

BASELINE_DECISIONS=$(wc -l < "$PROJECT/watcher_decisions.jsonl" 2>/dev/null || echo "0")
echo "Baseline decisions: $BASELINE_DECISIONS" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Alert thresholds
MEM_WARN_MB=500    # Warn if RSS exceeds this
FD_WARN=200        # Warn if file descriptors exceed this
DISK_WARN_GB=5     # Warn if available disk drops below this

CHECK_NUM=0

while true; do
    CHECK_NUM=$((CHECK_NUM + 1))
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    ALERTS=""

    echo "--- Check #$CHECK_NUM | $TS ---" | tee -a "$LOGFILE"

    # 1. WATCHER process
    WATCHER_PID=$(pgrep -f "watcher_agent.py" | head -1 || echo "")
    if [ -n "$WATCHER_PID" ]; then
        RSS_KB=$(ps -o rss= -p "$WATCHER_PID" 2>/dev/null || echo "0")
        RSS_MB=$((RSS_KB / 1024))
        DELTA_MB=$((RSS_MB - BASELINE_RSS_MB))
        echo "  WATCHER: PID=$WATCHER_PID RSS=${RSS_MB}MB (Δ${DELTA_MB}MB)" | tee -a "$LOGFILE"

        if [ "$RSS_MB" -gt "$MEM_WARN_MB" ]; then
            ALERTS="${ALERTS}⚠ MEMORY WARNING: ${RSS_MB}MB exceeds ${MEM_WARN_MB}MB threshold\n"
        fi

        # File descriptors
        FD_COUNT=$(ls /proc/$WATCHER_PID/fd 2>/dev/null | wc -l || echo "0")
        FD_DELTA=$((FD_COUNT - BASELINE_FD))
        echo "  File descriptors: $FD_COUNT (Δ$FD_DELTA)" | tee -a "$LOGFILE"

        if [ "$FD_COUNT" -gt "$FD_WARN" ]; then
            ALERTS="${ALERTS}⚠ FD WARNING: $FD_COUNT exceeds $FD_WARN threshold\n"
        fi
    else
        echo "  WATCHER: NOT RUNNING" | tee -a "$LOGFILE"
        ALERTS="${ALERTS}⚠ WATCHER process not found\n"
    fi

    # 2. GPU VRAM
    VRAM_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    echo "  GPU VRAM: $VRAM_INFO" | tee -a "$LOGFILE"

    # 3. Disk space
    DISK_AVAIL_KB=$(df /home/michael --output=avail | tail -1 | tr -d ' ')
    DISK_AVAIL_GB=$((DISK_AVAIL_KB / 1048576))
    echo "  Disk available: ${DISK_AVAIL_GB}GB" | tee -a "$LOGFILE"

    if [ "$DISK_AVAIL_GB" -lt "$DISK_WARN_GB" ]; then
        ALERTS="${ALERTS}⚠ DISK WARNING: Only ${DISK_AVAIL_GB}GB available\n"
    fi

    # 4. Request queue
    PENDING=$(ls "$PROJECT/watcher_requests/"*.json 2>/dev/null | grep -v archive | wc -l || echo "0")
    ARCHIVED=$(ls "$PROJECT/watcher_requests/archive/"*.json 2>/dev/null | wc -l || echo "0")
    echo "  Queue: pending=$PENDING archived=$ARCHIVED" | tee -a "$LOGFILE"

    # 5. Decision log
    DECISIONS=$(wc -l < "$PROJECT/watcher_decisions.jsonl" 2>/dev/null || echo "0")
    DECISION_DELTA=$((DECISIONS - BASELINE_DECISIONS))
    echo "  Decisions: $DECISIONS (Δ$DECISION_DELTA since start)" | tee -a "$LOGFILE"

    # 6. LLM server
    LLM_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:8080/health 2>/dev/null || echo "000")
    echo "  LLM health: HTTP $LLM_STATUS" | tee -a "$LOGFILE"

    if [ "$LLM_STATUS" = "000" ]; then
        ALERTS="${ALERTS}⚠ LLM server unreachable\n"
    fi

    # 7. Recent errors in watcher log
    if [ -f "$PROJECT/logs/watcher_agent.log" ]; then
        # Count errors in last 5 minutes of log
        RECENT_ERRORS=$(tail -100 "$PROJECT/logs/watcher_agent.log" | grep -ci "error\|exception\|traceback" || echo "0")
        echo "  Recent log errors: $RECENT_ERRORS" | tee -a "$LOGFILE"

        if [ "$RECENT_ERRORS" -gt "0" ]; then
            ALERTS="${ALERTS}⚠ $RECENT_ERRORS errors in recent log entries\n"
            # Show last error line
            LAST_ERR=$(tail -100 "$PROJECT/logs/watcher_agent.log" | grep -i "error\|exception" | tail -1)
            echo "    → $LAST_ERR" | tee -a "$LOGFILE"
        fi
    fi

    # 8. Halt flag
    if [ -f "$PROJECT/watcher_halt.flag" ]; then
        echo "  ⛔ HALT FLAG ACTIVE" | tee -a "$LOGFILE"
        ALERTS="${ALERTS}⛔ Halt flag is set\n"
    fi

    # Print alerts
    if [ -n "$ALERTS" ]; then
        echo "" | tee -a "$LOGFILE"
        echo -e "$ALERTS" | tee -a "$LOGFILE"
    fi

    echo "" | tee -a "$LOGFILE"
    sleep "$INTERVAL"
done
