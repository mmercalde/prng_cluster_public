#!/bin/bash
#===============================================================================
# STEP 1 GPU STABILITY BENCHMARK
# Version: 1.1.1
# Date: 2026-01-25
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

# Defaults
TRIALS=${1:-10}
MAX_SEEDS=${2:-100000}

# Track first unknown detection
FIRST_UNKNOWN_RIG1=0
FIRST_UNKNOWN_RIG2=0

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
    # Ensure we return a number
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
    # Ensure we return a number
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        COUNT=0
    fi
    echo "$COUNT"
}

get_gpu_temps() {
    local HOST=$1
    if [ "$HOST" == "localhost" ]; then
        nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//'
    else
        ssh michael@$HOST "rocm-smi 2>&1 | grep -E '^\s*[0-9]+' | awk '{print \$5}' | tr -d '°C' | tr '\n' ',' | sed 's/,$//'" 2>/dev/null || echo "N/A"
    fi
}

check_all_gpu_health() {
    # Zeus
    local ZEUS_GPUS=$(get_gpu_count "localhost")
    local ZEUS_UNKNOWN=0
    
    # rig-6600
    local RIG1_VISIBLE=$(get_gpu_count "$RIG_6600")
    local RIG1_UNKNOWN=$(get_unknown_count "$RIG_6600")
    local RIG1_HEALTHY=$((RIG1_VISIBLE - RIG1_UNKNOWN))
    
    # rig-6600b
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
        echo "=== ZEUS nvidia-smi ==="
        nvidia-smi 2>&1 || echo "nvidia-smi failed"
        echo ""
        echo "=== RIG-6600 rocm-smi ==="
        ssh michael@$RIG_6600 "rocm-smi 2>&1" 2>&1 || echo "SSH failed"
        echo ""
        echo "=== RIG-6600B rocm-smi ==="
        ssh michael@$RIG_6600B "rocm-smi 2>&1" 2>&1 || echo "SSH failed"
        echo ""
        echo "=== RIG-6600 dmesg (GPU errors) ==="
        ssh michael@$RIG_6600 "dmesg 2>/dev/null | grep -i -E 'amdgpu|drm|error|fail|oom|killed|smu' | tail -20" 2>&1 || true
        echo ""
        echo "=== RIG-6600B dmesg (GPU errors) ==="
        ssh michael@$RIG_6600B "dmesg 2>/dev/null | grep -i -E 'amdgpu|drm|error|fail|oom|killed|smu' | tail -20" 2>&1 || true
    } > "$DIAG_FILE"
}

capture_rocm_snapshot() {
    local TRIAL=$1
    local PHASE=$2
    local SNAPSHOT_FILE="$OUTPUT_DIR/rocm_trial${TRIAL}_${PHASE}.txt"
    
    {
        echo "=== Trial $TRIAL - $PHASE ==="
        echo "Timestamp: $(date)"
        echo ""
        echo "=== RIG-6600 ==="
        ssh michael@$RIG_6600 "rocm-smi 2>&1" 2>&1 || true
        echo ""
        echo "=== RIG-6600B ==="
        ssh michael@$RIG_6600B "rocm-smi 2>&1" 2>&1 || true
    } > "$SNAPSHOT_FILE"
}

#===============================================================================
# MAIN BENCHMARK
#===============================================================================

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  STEP 1 GPU STABILITY BENCHMARK v1.1.1${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
log "Trials: $TRIALS"
log "Max Seeds: $MAX_SEEDS"
log "Output: $OUTPUT_DIR"
echo ""

# CSV Header
echo "trial,start_time,end_time,duration_sec,zeus_gpus,rig1_visible,rig1_unknown,rig2_visible,rig2_unknown,total_visible,total_unknown,total_healthy,zeus_temps,rig1_temps,survivors,exit_code,errors" > "$RESULTS_FILE"

# Pre-flight check
log "${YELLOW}Pre-flight GPU enumeration...${NC}"
PRE_HEALTH=$(check_all_gpu_health)
PRE_ZEUS=$(echo $PRE_HEALTH | cut -d',' -f1)
PRE_RIG1_VIS=$(echo $PRE_HEALTH | cut -d',' -f2)
PRE_RIG1_UNK=$(echo $PRE_HEALTH | cut -d',' -f3)
PRE_RIG2_VIS=$(echo $PRE_HEALTH | cut -d',' -f4)
PRE_RIG2_UNK=$(echo $PRE_HEALTH | cut -d',' -f5)
PRE_TOTAL_VIS=$(echo $PRE_HEALTH | cut -d',' -f6)
PRE_TOTAL_UNK=$(echo $PRE_HEALTH | cut -d',' -f7)
PRE_TOTAL_HEALTHY=$(echo $PRE_HEALTH | cut -d',' -f8)

log "PRE-RUN STATUS:"
log "  Zeus: $PRE_ZEUS GPUs"
log "  rig-6600: $PRE_RIG1_VIS visible, $PRE_RIG1_UNK unknown"
log "  rig-6600b: $PRE_RIG2_VIS visible, $PRE_RIG2_UNK unknown"
log "  Total: $PRE_TOTAL_VIS visible, $PRE_TOTAL_UNK unknown, $PRE_TOTAL_HEALTHY healthy"

if [ "$PRE_TOTAL_HEALTHY" -lt 26 ]; then
    log "${RED}WARNING: Only $PRE_TOTAL_HEALTHY healthy GPUs (expected 26)${NC}"
    capture_full_diagnostics "preflight_degraded"
fi

capture_full_diagnostics "preflight"
capture_rocm_snapshot 0 "preflight"

# Clean previous outputs
rm -f optimal_window_config.json bidirectional_survivors.json 2>/dev/null
rm -f forward_survivors.json reverse_survivors.json 2>/dev/null
rm -f bidirectional_survivors_binary.npz 2>/dev/null

# Run benchmark trials
log "${GREEN}Starting benchmark: $TRIALS trials × $MAX_SEEDS seeds${NC}"
echo ""

for i in $(seq 1 $TRIALS); do
    log "${CYAN}=== Trial $i/$TRIALS ===${NC}"
    
    capture_rocm_snapshot $i "pre"
    
    START_TIME=$(date '+%H:%M:%S')
    START_EPOCH=$(date +%s)
    
    rm -f optimal_window_config.json bidirectional_survivors.json bidirectional_survivors_binary.npz 2>/dev/null
    
    TRIAL_LOG="$OUTPUT_DIR/trial_${i}.log"
    EXIT_CODE=0
    ERRORS=""
    
    timeout 600 python3 window_optimizer.py \
        --strategy bayesian \
        --lottery-file "$LOTTERY_FILE" \
        --trials 1 \
        --max-seeds "$MAX_SEEDS" \
        --prng-type java_lcg \
        --test-both-modes \
        > "$TRIAL_LOG" 2>&1 || EXIT_CODE=$?
    
    END_TIME=$(date '+%H:%M:%S')
    END_EPOCH=$(date +%s)
    DURATION=$((END_EPOCH - START_EPOCH))
    
    capture_rocm_snapshot $i "post"
    
    if grep -qi "oom\|out of memory\|killed\|error\|exception\|traceback" "$TRIAL_LOG" 2>/dev/null; then
        ERRORS=$(grep -i "oom\|out of memory\|killed\|error\|exception" "$TRIAL_LOG" | head -3 | tr '\n' ';' | sed 's/;$//' | tr -d '"')
    fi
    
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
    
    ZEUS_TEMPS=$(get_gpu_temps "localhost")
    RIG1_TEMPS=$(get_gpu_temps "$RIG_6600")
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "${GREEN}✓ Trial $i: ${DURATION}s, Healthy=$POST_TOTAL_HEALTHY, Unknown=$POST_TOTAL_UNK, Survivors=$SURVIVORS${NC}"
    else
        log "${RED}✗ Trial $i: FAILED (code=$EXIT_CODE), Healthy=$POST_TOTAL_HEALTHY${NC}"
        capture_full_diagnostics "trial_${i}_failed"
    fi
    
    # Check for NEW unknown state
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
    
    # Check for GPU dropout
    if [ "$POST_TOTAL_VIS" -lt "$PRE_TOTAL_VIS" ]; then
        log "${RED}⚠ GPU DROPOUT: $PRE_TOTAL_VIS → $POST_TOTAL_VIS${NC}"
        capture_full_diagnostics "trial_${i}_dropout"
    fi
    
    echo "$i,$START_TIME,$END_TIME,$DURATION,$POST_ZEUS,$POST_RIG1_VIS,$POST_RIG1_UNK,$POST_RIG2_VIS,$POST_RIG2_UNK,$POST_TOTAL_VIS,$POST_TOTAL_UNK,$POST_TOTAL_HEALTHY,$ZEUS_TEMPS,$RIG1_TEMPS,$SURVIVORS,$EXIT_CODE,\"$ERRORS\"" >> "$RESULTS_FILE"
    
    # Update baseline
    PRE_RIG1_UNK=$POST_RIG1_UNK
    PRE_RIG2_UNK=$POST_RIG2_UNK
    PRE_TOTAL_VIS=$POST_TOTAL_VIS
    
    sleep 2
done

capture_full_diagnostics "postrun"

echo ""
log "${CYAN}============================================${NC}"
log "${CYAN}  BENCHMARK COMPLETE${NC}"
log "${CYAN}============================================${NC}"

FINAL_HEALTH=$(check_all_gpu_health)
FINAL_TOTAL_VIS=$(echo $FINAL_HEALTH | cut -d',' -f6)
FINAL_TOTAL_UNK=$(echo $FINAL_HEALTH | cut -d',' -f7)
FINAL_TOTAL_HEALTHY=$(echo $FINAL_HEALTH | cut -d',' -f8)

log "FINAL: $FINAL_TOTAL_VIS visible, $FINAL_TOTAL_UNK unknown, $FINAL_TOTAL_HEALTHY healthy"

SUCCESSES=$(grep -c ",0,\"" "$RESULTS_FILE" 2>/dev/null || echo "0")
log "Successful trials: $SUCCESSES/$TRIALS"
log "Results: $RESULTS_FILE"

echo ""
cat "$RESULTS_FILE" | column -t -s','

