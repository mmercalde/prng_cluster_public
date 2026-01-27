#!/bin/bash
#===============================================================================
# STEP 1 GPU STABILITY BENCHMARK - WITH MEMORY MONITORING
# Version: 2.0.0
# Date: 2026-01-26
# 
# Key addition: RAM monitoring on mining rigs to detect OOM before it crashes
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
RIG_6600="192.168.3.120"
RIG_6600B="192.168.3.154"
LOTTERY_FILE="synthetic_lottery.json"
OUTPUT_DIR="benchmark_step1_$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="$OUTPUT_DIR/results.csv"
LOG_FILE="$OUTPUT_DIR/benchmark.log"
MEMORY_LOG="$OUTPUT_DIR/memory_samples.csv"

# Defaults
TRIALS=${1:-10}
MAX_SEEDS=${2:-100000}
MEMORY_SAMPLE_INTERVAL=5  # seconds between memory samples

# Track issues
FIRST_UNKNOWN_RIG1=0
FIRST_UNKNOWN_RIG2=0
OOM_DETECTED=0

mkdir -p "$OUTPUT_DIR"

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

log() {
    echo -e "$(date '+%H:%M:%S') $1" | tee -a "$LOG_FILE"
}

get_gpu_count() {
    local HOST=$1
    local COUNT=0
    if [ "$HOST" == "localhost" ]; then
        COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
    else
        COUNT=$(ssh michael@$HOST "rocm-smi 2>&1 | grep -c '0x73ff'" 2>/dev/null | tr -d '[:space:]')
    fi
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        COUNT=0
    fi
    echo "$COUNT"
}

get_unknown_count() {
    local HOST=$1
    local COUNT=0
    if [ "$HOST" == "localhost" ]; then
        COUNT=0
    else
        COUNT=$(ssh michael@$HOST "rocm-smi 2>&1 | grep -E '^\s*[0-9]+' | grep -c 'unknown'" 2>/dev/null | tr -d '[:space:]')
    fi
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        COUNT=0
    fi
    echo "$COUNT"
}

#===============================================================================
# MEMORY MONITORING FUNCTIONS
#===============================================================================

get_memory_stats() {
    # Returns: total_mb,used_mb,available_mb,used_percent
    local HOST=$1
    if [ "$HOST" == "localhost" ]; then
        free -m | awk '/^Mem:/ {printf "%d,%d,%d,%.1f", $2, $3, $7, ($3/$2)*100}'
    else
        ssh michael@$HOST "free -m | awk '/^Mem:/ {printf \"%d,%d,%d,%.1f\", \$2, \$3, \$7, (\$3/\$2)*100}'" 2>/dev/null || echo "0,0,0,0"
    fi
}

check_oom_killer() {
    # Check if OOM killer fired recently (last 60 seconds)
    local HOST=$1
    local COUNT=0
    if [ "$HOST" == "localhost" ]; then
        COUNT=$(dmesg --time-format iso 2>/dev/null | grep -c "Out of memory\|oom-kill\|Killed process" | tail -1 || echo "0")
    else
        COUNT=$(ssh michael@$HOST "dmesg 2>/dev/null | tail -100 | grep -c 'Out of memory\|oom-kill\|Killed process'" 2>/dev/null || echo "0")
    fi
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        COUNT=0
    fi
    echo "$COUNT"
}

get_oom_victims() {
    # Get recent OOM kill victims
    local HOST=$1
    if [ "$HOST" == "localhost" ]; then
        dmesg 2>/dev/null | grep -E "Out of memory|Killed process" | tail -5 || true
    else
        ssh michael@$HOST "dmesg 2>/dev/null | grep -E 'Out of memory|Killed process' | tail -5" 2>/dev/null || true
    fi
}

capture_memory_snapshot() {
    local TRIAL=$1
    local PHASE=$2
    local TIMESTAMP=$(date '+%H:%M:%S')
    
    local ZEUS_MEM=$(get_memory_stats "localhost")
    local RIG1_MEM=$(get_memory_stats "$RIG_6600")
    local RIG2_MEM=$(get_memory_stats "$RIG_6600B")
    
    # Parse into components
    local ZEUS_USED=$(echo $ZEUS_MEM | cut -d',' -f2)
    local ZEUS_AVAIL=$(echo $ZEUS_MEM | cut -d',' -f3)
    local ZEUS_PCT=$(echo $ZEUS_MEM | cut -d',' -f4)
    
    local RIG1_USED=$(echo $RIG1_MEM | cut -d',' -f2)
    local RIG1_AVAIL=$(echo $RIG1_MEM | cut -d',' -f3)
    local RIG1_PCT=$(echo $RIG1_MEM | cut -d',' -f4)
    
    local RIG2_USED=$(echo $RIG2_MEM | cut -d',' -f2)
    local RIG2_AVAIL=$(echo $RIG2_MEM | cut -d',' -f3)
    local RIG2_PCT=$(echo $RIG2_MEM | cut -d',' -f4)
    
    echo "$TRIAL,$PHASE,$TIMESTAMP,$ZEUS_USED,$ZEUS_AVAIL,$ZEUS_PCT,$RIG1_USED,$RIG1_AVAIL,$RIG1_PCT,$RIG2_USED,$RIG2_AVAIL,$RIG2_PCT" >> "$MEMORY_LOG"
    
    # Return the line for display
    echo "Zeus: ${ZEUS_USED}MB/${ZEUS_PCT}% | rig-6600: ${RIG1_USED}MB/${RIG1_PCT}% | rig-6600b: ${RIG2_USED}MB/${RIG2_PCT}%"
}

# Background memory monitor - samples during trial execution
start_memory_monitor() {
    local TRIAL=$1
    local PID_FILE="$OUTPUT_DIR/.memory_monitor_pid"
    
    (
        while true; do
            capture_memory_snapshot "$TRIAL" "running" > /dev/null
            sleep $MEMORY_SAMPLE_INTERVAL
        done
    ) &
    
    echo $! > "$PID_FILE"
}

stop_memory_monitor() {
    local PID_FILE="$OUTPUT_DIR/.memory_monitor_pid"
    if [ -f "$PID_FILE" ]; then
        kill $(cat "$PID_FILE") 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
}

check_all_gpu_health() {
    local ZEUS_GPUS=$(get_gpu_count "localhost")
    local ZEUS_UNKNOWN=0
    
    local RIG1_VISIBLE=$(get_gpu_count "$RIG_6600")
    local RIG1_UNKNOWN=$(get_unknown_count "$RIG_6600")
    local RIG1_HEALTHY=$((RIG1_VISIBLE - RIG1_UNKNOWN))
    
    local RIG2_VISIBLE=$(get_gpu_count "$RIG_6600B")
    local RIG2_UNKNOWN=$(get_unknown_count "$RIG_6600B")
    local RIG2_HEALTHY=$((RIG2_VISIBLE - RIG2_UNKNOWN))
    
    local TOTAL_VISIBLE=$((ZEUS_GPUS + RIG1_VISIBLE + RIG2_VISIBLE))
    local TOTAL_UNKNOWN=$((ZEUS_UNKNOWN + RIG1_UNKNOWN + RIG2_UNKNOWN))
    local TOTAL_HEALTHY=$((ZEUS_GPUS + RIG1_HEALTHY + RIG2_HEALTHY))
    
    echo "$ZEUS_GPUS,$RIG1_VISIBLE,$RIG1_UNKNOWN,$RIG2_VISIBLE,$RIG2_UNKNOWN,$TOTAL_VISIBLE,$TOTAL_UNKNOWN,$TOTAL_HEALTHY"
}

capture_full_diagnostics() {
    local PHASE=$1
    local DIAG_FILE="$OUTPUT_DIR/diagnostics_${PHASE}.txt"
    
    log "${YELLOW}Capturing diagnostics: $PHASE${NC}"
    
    {
        echo "=== DIAGNOSTICS: $PHASE ==="
        echo "Timestamp: $(date)"
        echo ""
        
        echo "=== MEMORY STATUS ==="
        echo "--- Zeus ---"
        free -h
        echo ""
        echo "--- rig-6600 ---"
        ssh michael@$RIG_6600 "free -h" 2>&1 || echo "SSH failed"
        echo ""
        echo "--- rig-6600b ---"
        ssh michael@$RIG_6600B "free -h" 2>&1 || echo "SSH failed"
        echo ""
        
        echo "=== OOM KILLER HISTORY ==="
        echo "--- Zeus ---"
        dmesg 2>/dev/null | grep -E "Out of memory|oom-kill|Killed process" | tail -10 || echo "None"
        echo ""
        echo "--- rig-6600 ---"
        ssh michael@$RIG_6600 "dmesg 2>/dev/null | grep -E 'Out of memory|oom-kill|Killed process' | tail -10" 2>&1 || echo "None"
        echo ""
        echo "--- rig-6600b ---"
        ssh michael@$RIG_6600B "dmesg 2>/dev/null | grep -E 'Out of memory|oom-kill|Killed process' | tail -10" 2>&1 || echo "None"
        echo ""
        
        echo "=== GPU STATUS ==="
        echo "--- Zeus nvidia-smi ---"
        nvidia-smi 2>&1 || echo "nvidia-smi failed"
        echo ""
        echo "--- rig-6600 rocm-smi ---"
        ssh michael@$RIG_6600 "rocm-smi 2>&1" 2>&1 || echo "SSH failed"
        echo ""
        echo "--- rig-6600b rocm-smi ---"
        ssh michael@$RIG_6600B "rocm-smi 2>&1" 2>&1 || echo "SSH failed"
        echo ""
        
        echo "=== TOP MEMORY CONSUMERS ==="
        echo "--- rig-6600 ---"
        ssh michael@$RIG_6600 "ps aux --sort=-%mem | head -10" 2>&1 || true
        echo ""
        echo "--- rig-6600b ---"
        ssh michael@$RIG_6600B "ps aux --sort=-%mem | head -10" 2>&1 || true
        
    } > "$DIAG_FILE"
}

#===============================================================================
# MAIN BENCHMARK
#===============================================================================

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  STEP 1 GPU STABILITY BENCHMARK v2.0.0${NC}"
echo -e "${CYAN}  WITH MEMORY MONITORING${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
log "Trials: $TRIALS"
log "Max Seeds: $MAX_SEEDS"
log "Memory sample interval: ${MEMORY_SAMPLE_INTERVAL}s"
log "Output: $OUTPUT_DIR"
echo ""

# CSV Headers
echo "trial,start_time,end_time,duration_sec,zeus_gpus,rig1_visible,rig1_unknown,rig2_visible,rig2_unknown,total_visible,total_unknown,total_healthy,survivors,exit_code,oom_detected,errors" > "$RESULTS_FILE"
echo "trial,phase,timestamp,zeus_used_mb,zeus_avail_mb,zeus_pct,rig1_used_mb,rig1_avail_mb,rig1_pct,rig2_used_mb,rig2_avail_mb,rig2_pct" > "$MEMORY_LOG"

# Pre-flight check
log "${YELLOW}Pre-flight checks...${NC}"

# Memory baseline
log "Memory baseline:"
MEM_BASELINE=$(capture_memory_snapshot 0 "preflight")
log "  $MEM_BASELINE"

# Check for existing OOM events
PRE_OOM_ZEUS=$(check_oom_killer "localhost")
PRE_OOM_RIG1=$(check_oom_killer "$RIG_6600")
PRE_OOM_RIG2=$(check_oom_killer "$RIG_6600B")
log "Pre-existing OOM events: Zeus=$PRE_OOM_ZEUS, rig-6600=$PRE_OOM_RIG1, rig-6600b=$PRE_OOM_RIG2"

# GPU health
PRE_HEALTH=$(check_all_gpu_health)
PRE_ZEUS=$(echo $PRE_HEALTH | cut -d',' -f1)
PRE_RIG1_VIS=$(echo $PRE_HEALTH | cut -d',' -f2)
PRE_RIG1_UNK=$(echo $PRE_HEALTH | cut -d',' -f3)
PRE_RIG2_VIS=$(echo $PRE_HEALTH | cut -d',' -f4)
PRE_RIG2_UNK=$(echo $PRE_HEALTH | cut -d',' -f5)
PRE_TOTAL_HEALTHY=$(echo $PRE_HEALTH | cut -d',' -f8)

log "GPU status:"
log "  Zeus: $PRE_ZEUS GPUs"
log "  rig-6600: $PRE_RIG1_VIS visible, $PRE_RIG1_UNK unknown"
log "  rig-6600b: $PRE_RIG2_VIS visible, $PRE_RIG2_UNK unknown"
log "  Total healthy: $PRE_TOTAL_HEALTHY"

if [ "$PRE_TOTAL_HEALTHY" -lt 26 ]; then
    log "${RED}WARNING: Only $PRE_TOTAL_HEALTHY healthy GPUs (expected 26)${NC}"
fi

capture_full_diagnostics "preflight"

# Clean previous outputs
rm -f optimal_window_config.json bidirectional_survivors.json 2>/dev/null
rm -f forward_survivors.json reverse_survivors.json 2>/dev/null
rm -f bidirectional_survivors_binary.npz 2>/dev/null

# Run benchmark trials
log "${GREEN}Starting benchmark: $TRIALS trials × $MAX_SEEDS seeds${NC}"
echo ""

for i in $(seq 1 $TRIALS); do
    log "${CYAN}=== Trial $i/$TRIALS ===${NC}"
    
    # Pre-trial memory
    MEM_PRE=$(capture_memory_snapshot $i "pre")
    log "  Pre-trial memory: $MEM_PRE"
    
    START_TIME=$(date '+%H:%M:%S')
    START_EPOCH=$(date +%s)
    
    rm -f optimal_window_config.json bidirectional_survivors.json bidirectional_survivors_binary.npz 2>/dev/null
    
    TRIAL_LOG="$OUTPUT_DIR/trial_${i}.log"
    EXIT_CODE=0
    ERRORS=""
    TRIAL_OOM=0
    
    # Start background memory monitor
    start_memory_monitor $i
    
    # Run the trial
    timeout 600 python3 window_optimizer.py \
        --strategy bayesian \
        --lottery-file "$LOTTERY_FILE" \
        --trials 1 \
        --max-seeds "$MAX_SEEDS" \
        --prng-type java_lcg \
        --test-both-modes \
        > "$TRIAL_LOG" 2>&1 || EXIT_CODE=$?
    
    # Stop memory monitor
    stop_memory_monitor
    
    END_TIME=$(date '+%H:%M:%S')
    END_EPOCH=$(date +%s)
    DURATION=$((END_EPOCH - START_EPOCH))
    
    # Post-trial memory
    MEM_POST=$(capture_memory_snapshot $i "post")
    log "  Post-trial memory: $MEM_POST"
    
    # Check for OOM events
    POST_OOM_RIG1=$(check_oom_killer "$RIG_6600")
    POST_OOM_RIG2=$(check_oom_killer "$RIG_6600B")
    
    if [ "$POST_OOM_RIG1" -gt "$PRE_OOM_RIG1" ]; then
        log "${RED}⚠ OOM KILLER FIRED on rig-6600!${NC}"
        TRIAL_OOM=1
        OOM_DETECTED=1
        log "  Victims:"
        get_oom_victims "$RIG_6600" | while read line; do log "    $line"; done
        capture_full_diagnostics "trial_${i}_oom_rig1"
    fi
    
    if [ "$POST_OOM_RIG2" -gt "$PRE_OOM_RIG2" ]; then
        log "${RED}⚠ OOM KILLER FIRED on rig-6600b!${NC}"
        TRIAL_OOM=1
        OOM_DETECTED=1
        log "  Victims:"
        get_oom_victims "$RIG_6600B" | while read line; do log "    $line"; done
        capture_full_diagnostics "trial_${i}_oom_rig2"
    fi
    
    # Update OOM baseline
    PRE_OOM_RIG1=$POST_OOM_RIG1
    PRE_OOM_RIG2=$POST_OOM_RIG2
    
    # Check for errors in log
    if grep -qi "oom\|out of memory\|killed\|error\|exception\|traceback" "$TRIAL_LOG" 2>/dev/null; then
        ERRORS=$(grep -i "oom\|out of memory\|killed\|error\|exception" "$TRIAL_LOG" | head -3 | tr '\n' ';' | sed 's/;$//' | tr -d '"')
    fi
    
    # Survivor count
    SURVIVORS=0
    if [ -f "bidirectional_survivors.json" ]; then
        SURVIVORS=$(python3 -c "import json; print(len(json.load(open('bidirectional_survivors.json'))))" 2>/dev/null || echo "0")
    fi
    
    # Post-trial GPU health
    POST_HEALTH=$(check_all_gpu_health)
    POST_ZEUS=$(echo $POST_HEALTH | cut -d',' -f1)
    POST_RIG1_VIS=$(echo $POST_HEALTH | cut -d',' -f2)
    POST_RIG1_UNK=$(echo $POST_HEALTH | cut -d',' -f3)
    POST_RIG2_VIS=$(echo $POST_HEALTH | cut -d',' -f4)
    POST_RIG2_UNK=$(echo $POST_HEALTH | cut -d',' -f5)
    POST_TOTAL_VIS=$(echo $POST_HEALTH | cut -d',' -f6)
    POST_TOTAL_UNK=$(echo $POST_HEALTH | cut -d',' -f7)
    POST_TOTAL_HEALTHY=$(echo $POST_HEALTH | cut -d',' -f8)
    
    # Report
    if [ $EXIT_CODE -eq 0 ] && [ $TRIAL_OOM -eq 0 ]; then
        log "${GREEN}✓ Trial $i: ${DURATION}s, Healthy=$POST_TOTAL_HEALTHY, Survivors=$SURVIVORS${NC}"
    elif [ $TRIAL_OOM -eq 1 ]; then
        log "${RED}✗ Trial $i: OOM DETECTED (code=$EXIT_CODE), Healthy=$POST_TOTAL_HEALTHY${NC}"
    else
        log "${RED}✗ Trial $i: FAILED (code=$EXIT_CODE), Healthy=$POST_TOTAL_HEALTHY${NC}"
        capture_full_diagnostics "trial_${i}_failed"
    fi
    
    # Check for GPU degradation
    if [ "$POST_RIG1_UNK" -gt "$PRE_RIG1_UNK" ] && [ "$FIRST_UNKNOWN_RIG1" -eq 0 ]; then
        log "${RED}⚠ FIRST UNKNOWN on rig-6600 at trial $i!${NC}"
        capture_full_diagnostics "trial_${i}_first_unknown_rig1"
        FIRST_UNKNOWN_RIG1=1
    fi
    
    if [ "$POST_RIG2_UNK" -gt "$PRE_RIG2_UNK" ] && [ "$FIRST_UNKNOWN_RIG2" -eq 0 ]; then
        log "${RED}⚠ FIRST UNKNOWN on rig-6600b at trial $i!${NC}"
        capture_full_diagnostics "trial_${i}_first_unknown_rig2"
        FIRST_UNKNOWN_RIG2=1
    fi
    
    # CSV output
    echo "$i,$START_TIME,$END_TIME,$DURATION,$POST_ZEUS,$POST_RIG1_VIS,$POST_RIG1_UNK,$POST_RIG2_VIS,$POST_RIG2_UNK,$POST_TOTAL_VIS,$POST_TOTAL_UNK,$POST_TOTAL_HEALTHY,$SURVIVORS,$EXIT_CODE,$TRIAL_OOM,\"$ERRORS\"" >> "$RESULTS_FILE"
    
    # Update baselines
    PRE_RIG1_UNK=$POST_RIG1_UNK
    PRE_RIG2_UNK=$POST_RIG2_UNK
    PRE_TOTAL_VIS=$POST_TOTAL_VIS
    
    sleep 2
done

capture_full_diagnostics "postrun"

#===============================================================================
# SUMMARY
#===============================================================================

echo ""
log "${CYAN}============================================${NC}"
log "${CYAN}  BENCHMARK COMPLETE${NC}"
log "${CYAN}============================================${NC}"

FINAL_HEALTH=$(check_all_gpu_health)
FINAL_TOTAL_HEALTHY=$(echo $FINAL_HEALTH | cut -d',' -f8)
FINAL_TOTAL_UNK=$(echo $FINAL_HEALTH | cut -d',' -f7)

log "Final GPU status: $FINAL_TOTAL_HEALTHY healthy, $FINAL_TOTAL_UNK unknown"

SUCCESSES=$(awk -F',' '$14==0 && $15==0 {count++} END {print count+0}' "$RESULTS_FILE")
OOM_COUNT=$(awk -F',' '$15==1 {count++} END {print count+0}' "$RESULTS_FILE")
FAILURES=$(awk -F',' '$14!=0 {count++} END {print count+0}' "$RESULTS_FILE")

log ""
log "RESULTS:"
log "  Successful: $SUCCESSES/$TRIALS"
log "  OOM events: $OOM_COUNT"
log "  Other failures: $FAILURES"
log ""
log "Output files:"
log "  Results: $RESULTS_FILE"
log "  Memory log: $MEMORY_LOG"
log "  Diagnostics: $OUTPUT_DIR/diagnostics_*.txt"

if [ $OOM_DETECTED -eq 1 ]; then
    log ""
    log "${RED}⚠ OOM DETECTED - Check memory_samples.csv for peak usage${NC}"
    log "${RED}  Consider reducing max_seeds or adding cooldown between trials${NC}"
fi

echo ""
echo "=== RESULTS ==="
cat "$RESULTS_FILE" | column -t -s','

echo ""
echo "=== MEMORY PEAKS ==="
echo "Highest memory usage per node (from samples):"
awk -F',' 'NR>1 {
    if ($6 > max_zeus) max_zeus=$6;
    if ($9 > max_rig1) max_rig1=$9;
    if ($12 > max_rig2) max_rig2=$12;
}
END {
    printf "  Zeus: %.1f%%\n  rig-6600: %.1f%%\n  rig-6600b: %.1f%%\n", max_zeus, max_rig1, max_rig2
}' "$MEMORY_LOG"
