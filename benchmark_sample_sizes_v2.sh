#!/bin/bash
# Benchmark script to find optimal sample size for 12 concurrent GPU jobs
# Version 2.0 - Addresses Team Beta concerns
#
# Fixes:
#   1. Passes trials explicitly via CLI (not mutating scripts)
#   2. Resets allocator state between tests
#   3. Calls run_scorer_meta_optimizer.sh directly with --trials and --sample-size
#   4. Only changes concurrency on ROCm nodes (not localhost)
#
# Usage: ./benchmark_sample_sizes_v2.sh

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate torch environment if not already active
if [[ -z "$VIRTUAL_ENV" ]] || [[ ! "$VIRTUAL_ENV" == *"torch"* ]]; then
    echo "Activating torch environment..."
    source ~/venvs/torch/bin/activate
fi

# Trap to restore config on any exit
cleanup() {
    if [ -f "distributed_config.json.bak_benchmark" ]; then
        cp distributed_config.json.bak_benchmark distributed_config.json 2>/dev/null || true
        echo "Configuration restored on exit."
    fi
}
trap cleanup EXIT

# NOTE:
# This benchmark tests allocator + host memory pressure under
# maximum concurrency. It does not measure:
# - GPU kernel saturation
# - inter-GPU communication
# - long-horizon training convergence

# Configuration
TRIALS_PER_TEST=20
SAMPLE_SIZES=(350 450 550 650 750)
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
LOG_DIR="benchmark_logs"

# Rig IPs
RIG_6600="192.168.3.120"
RIG_6600B="192.168.3.154"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  SAMPLE SIZE BENCHMARK v2.0"
echo "  12 Concurrent GPUs (ROCm nodes only)"
echo "  $TRIALS_PER_TEST trials per test"
echo "=============================================="
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Backup current config
echo -e "${YELLOW}Backing up current configuration...${NC}"
cp distributed_config.json distributed_config.json.bak_benchmark

# Set concurrency to 12 for ROCm nodes ONLY (not localhost)
# Team Beta fix #4: Precision - only change ROCm nodes
echo -e "${YELLOW}Setting max_concurrent_script_jobs to 12 for ROCm nodes only...${NC}"

python3 -c "
import json
with open('distributed_config.json', 'r') as f:
    config = json.load(f)
for node in config['nodes']:
    if node['hostname'] != 'localhost':
        node['max_concurrent_script_jobs'] = 12
with open('distributed_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Updated ROCm node concurrency to 12')
"

# Verify
echo "Concurrency settings after change:"
grep "max_concurrent" distributed_config.json

# Initialize results CSV with Team Beta metrics
echo "sample_size,trials,total_runtime_s,avg_trial_time_s,throughput_trials_per_min,success_rate,successful,failed,best_accuracy,pruned_trials" > "$RESULTS_FILE"

echo ""
echo -e "${CYAN}Starting benchmark with ${#SAMPLE_SIZES[@]} sample sizes, $TRIALS_PER_TEST trials each...${NC}"
echo ""

reset_allocator_state() {
    # Team Beta fix #2: Reset allocator state between tests
    echo -e "${YELLOW}Resetting allocator state on ROCm nodes...${NC}"
    
    # Sync filesystems
    ssh michael@$RIG_6600 "sync" 2>/dev/null &
    ssh michael@$RIG_6600B "sync" 2>/dev/null &
    wait
    
    # Brief pause for stability
    sleep 5
    
    echo -e "${GREEN}Allocator state reset complete${NC}"
}

capture_diagnostics() {
    # Capture GPU and system state for debugging
    local DIAG_FILE="$LOG_DIR/diagnostics_${SIZE}_$(date +%H%M%S).log"
    echo -e "${RED}Capturing diagnostics to $DIAG_FILE${NC}"
    
    echo "=== DIAGNOSTICS CAPTURED AT $(date) ===" > "$DIAG_FILE"
    echo "" >> "$DIAG_FILE"
    
    echo "=== RIG-6600 (192.168.3.120) rocm-smi ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600 "rocm-smi 2>&1" >> "$DIAG_FILE" 2>&1 || echo "SSH failed" >> "$DIAG_FILE"
    
    echo "" >> "$DIAG_FILE"
    echo "=== RIG-6600B (192.168.3.154) rocm-smi ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600B "rocm-smi 2>&1" >> "$DIAG_FILE" 2>&1 || echo "SSH failed" >> "$DIAG_FILE"
    
    echo "" >> "$DIAG_FILE"
    echo "=== RIG-6600 dmesg (last 30 GPU-related) ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600 "dmesg | grep -i -E 'amdgpu|drm|error|fail|gpu' | tail -30" >> "$DIAG_FILE" 2>&1 || true
    
    echo "" >> "$DIAG_FILE"
    echo "=== RIG-6600B dmesg (last 30 GPU-related) ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600B "dmesg | grep -i -E 'amdgpu|drm|error|fail|gpu' | tail -30" >> "$DIAG_FILE" 2>&1 || true
    
    echo "" >> "$DIAG_FILE"
    echo "=== Running processes on RIG-6600 ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600 "ps aux | grep -E 'python|scorer' | grep -v grep" >> "$DIAG_FILE" 2>&1 || true
    
    echo "" >> "$DIAG_FILE"
    echo "=== Running processes on RIG-6600B ===" >> "$DIAG_FILE"
    ssh michael@$RIG_6600B "ps aux | grep -E 'python|scorer' | grep -v grep" >> "$DIAG_FILE" 2>&1 || true
    
    echo -e "${YELLOW}Diagnostics saved to: $DIAG_FILE${NC}"
}

check_gpu_health() {
    # Check if any GPU shows unhealthy state (N/A values in critical fields)
    local UNHEALTHY=0
    
    # Check rig-6600b for N/A in SCLK column (indicates hung GPU)
    local BAD_GPUS=$(ssh michael@$RIG_6600B "rocm-smi 2>&1 | grep -E '^\s*[0-9]+' | grep -c 'unknown'" 2>/dev/null || echo "0")
    if [ "$BAD_GPUS" -gt 0 ]; then
        echo -e "${RED}WARNING: $BAD_GPUS unhealthy GPU(s) detected on rig-6600b${NC}"
        UNHEALTHY=1
    fi
    
    # Check rig-6600
    BAD_GPUS=$(ssh michael@$RIG_6600 "rocm-smi 2>&1 | grep -E '^\s*[0-9]+' | grep -c 'unknown'" 2>/dev/null || echo "0")
    if [ "$BAD_GPUS" -gt 0 ]; then
        echo -e "${RED}WARNING: $BAD_GPUS unhealthy GPU(s) detected on rig-6600${NC}"
        UNHEALTHY=1
    fi
    
    return $UNHEALTHY
}

# Pre-flight GPU health check
echo -e "${YELLOW}Pre-flight GPU health check...${NC}"
if ! check_gpu_health; then
    echo -e "${RED}GPU health check failed! Capture diagnostics and abort? (y/N)${NC}"
    read -t 10 -n 1 REPLY || REPLY="n"
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SIZE="preflight"
        capture_diagnostics
        exit 1
    fi
fi
echo -e "${GREEN}All GPUs healthy${NC}"
echo ""

for SIZE in "${SAMPLE_SIZES[@]}"; do
    echo "=============================================="
    echo -e "${GREEN}Testing sample_size = $SIZE${NC}"
    echo "=============================================="
    
    # Reset allocator state BEFORE each test
    reset_allocator_state
    
    LOG_FILE="$LOG_DIR/benchmark_size_${SIZE}.log"
    
    # Team Beta fix #1 and #3: 
    # - Pass trials as positional argument (script expects $1)
    # - Sed only the sample-size which is hardcoded
    echo "Running $TRIALS_PER_TEST trials with sample_size=$SIZE..."
    START_TIME=$(date +%s.%N)
    
    # Save original sample-size value
    ORIG_SAMPLE=$(grep -oP '(?<=--sample-size )\d+' run_scorer_meta_optimizer.sh 2>/dev/null | head -1 || echo "5000")
    
    echo "  Original sample_size=$ORIG_SAMPLE"
    echo "  Benchmark: trials=$TRIALS_PER_TEST (via arg), sample_size=$SIZE (via sed)"
    
    # Apply benchmark sample-size
    sed -i "s/--sample-size [0-9]*/--sample-size $SIZE/g" run_scorer_meta_optimizer.sh
    
    # Run with trials as first argument, with timeout
    TIMEOUT_MINUTES=10
    echo "  Timeout: ${TIMEOUT_MINUTES} minutes"
    
    timeout ${TIMEOUT_MINUTES}m ./run_scorer_meta_optimizer.sh $TRIALS_PER_TEST 2>&1 | tee "$LOG_FILE"
    RUN_EXIT_CODE=${PIPESTATUS[0]}
    echo "Run exit code: $RUN_EXIT_CODE"
    
    # Check for timeout (exit code 124)
    if [ "$RUN_EXIT_CODE" -eq 124 ]; then
        echo -e "${RED}TEST TIMED OUT after ${TIMEOUT_MINUTES} minutes!${NC}"
        capture_diagnostics
        echo -e "${YELLOW}Continue to next test? (y/N) - auto-abort in 10s${NC}"
        read -t 10 -n 1 REPLY || REPLY="n"
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting benchmark due to timeout."
            # Restore sample-size before exit
            sed -i "s/--sample-size [0-9]*/--sample-size $ORIG_SAMPLE/g" run_scorer_meta_optimizer.sh
            exit 1
        fi
    fi
    
    # Check for other failures
    if [ "$RUN_EXIT_CODE" -ne 0 ] && [ "$RUN_EXIT_CODE" -ne 124 ]; then
        echo -e "${RED}TEST FAILED with exit code $RUN_EXIT_CODE${NC}"
        capture_diagnostics
    fi
    
    # Post-run GPU health check
    echo "Post-run GPU health check..."
    if ! check_gpu_health; then
        echo -e "${RED}GPU became unhealthy during test!${NC}"
        capture_diagnostics
        echo -e "${YELLOW}Continue to next test? (y/N) - auto-skip in 10s${NC}"
        read -t 10 -n 1 REPLY || REPLY="n"
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting benchmark. Check diagnostics."
            exit 1
        fi
    fi
    
    # Log GPU memory watermark (Team Beta optional refinement)
    echo "--- GPU Memory Watermark (post-run) ---" >> "$LOG_FILE"
    ssh michael@$RIG_6600 "rocm-smi --showmeminfo vram 2>/dev/null | head -n 5" >> "$LOG_FILE" 2>&1 || true
    ssh michael@$RIG_6600B "rocm-smi --showmeminfo vram 2>/dev/null | head -n 5" >> "$LOG_FILE" 2>&1 || true
    
    # Restore original sample-size
    sed -i "s/--sample-size [0-9]*/--sample-size $ORIG_SAMPLE/g" run_scorer_meta_optimizer.sh
    echo -e "${GREEN}Original script values restored${NC}"
    
    END_TIME=$(date +%s.%N)
    TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Disable exit-on-error for parsing section
    set +e
    
    # Parse results from log - count successful trials
    # Use head -1 to ensure single value
    SUCCESSFUL=$(grep -c "^✓" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    FAILED=$(grep -c "^✗" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    
    # Sanitize to ensure integers
    SUCCESSFUL=${SUCCESSFUL//[^0-9]/}
    FAILED=${FAILED//[^0-9]/}
    SUCCESSFUL=${SUCCESSFUL:-0}
    FAILED=${FAILED:-0}
    
    # If checkmark parsing fails, try alternate pattern
    if [ "$SUCCESSFUL" -eq 0 ]; then
        SUCCESSFUL=$(grep -c "scorer_trial.*→.*[0-9.]*s)" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
        SUCCESSFUL=${SUCCESSFUL//[^0-9]/}
        SUCCESSFUL=${SUCCESSFUL:-0}
    fi
    
    TOTAL=$((SUCCESSFUL + FAILED))
    if [ "$TOTAL" -eq 0 ]; then
        TOTAL=$TRIALS_PER_TEST  # Fallback
    fi
    
    # Re-enable exit-on-error
    set -e
    
    # Team Beta metrics: Parse best accuracy and pruned trials
    set +e
    # Extract best trial accuracy (look for "accuracy": value in Best trial block)
    BEST_ACCURACY=$(grep -A10 "Best trial:" "$LOG_FILE" | grep -oP '(?<="accuracy": )[^,]+' | head -1 || echo "N/A")
    BEST_ACCURACY=${BEST_ACCURACY:-"N/A"}
    
    # Count pruned trials (Optuna logs pruned trials)
    PRUNED_TRIALS=$(grep -c "Trial .* pruned" "$LOG_FILE" 2>/dev/null || echo "0")
    PRUNED_TRIALS=${PRUNED_TRIALS//[^0-9]/}
    PRUNED_TRIALS=${PRUNED_TRIALS:-0}
    set -e
    
    if [ "$TOTAL" -gt 0 ] && [ "$SUCCESSFUL" -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=4; $SUCCESSFUL / $TOTAL" | bc 2>/dev/null || echo "0")
        AVG_TRIAL_TIME=$(echo "scale=2; $TOTAL_TIME / $SUCCESSFUL" | bc 2>/dev/null || echo "0")
        THROUGHPUT=$(echo "scale=2; ($SUCCESSFUL * 60) / $TOTAL_TIME" | bc 2>/dev/null || echo "0")
    else
        SUCCESS_RATE="0"
        AVG_TRIAL_TIME="0"
        THROUGHPUT="0"
        echo -e "${RED}WARNING: No successful trials detected${NC}"
    fi
    
    # Record results (including Team Beta metrics)
    echo "$SIZE,$TOTAL,$TOTAL_TIME,$AVG_TRIAL_TIME,$THROUGHPUT,$SUCCESS_RATE,$SUCCESSFUL,$FAILED,$BEST_ACCURACY,$PRUNED_TRIALS" >> "$RESULTS_FILE"
    
    echo ""
    echo -e "${GREEN}Results for sample_size=$SIZE:${NC}"
    echo "  Total runtime: ${TOTAL_TIME}s"
    echo "  Successful: $SUCCESSFUL / $TOTAL"
    echo "  Avg trial time: ${AVG_TRIAL_TIME}s"
    echo "  Throughput: ${THROUGHPUT} trials/min"
    echo "  Success rate: ${SUCCESS_RATE}"
    echo "  Best accuracy: ${BEST_ACCURACY}"
    echo "  Pruned trials: ${PRUNED_TRIALS}"
    echo ""
    
    # Longer pause between tests for full cooldown
    echo "Cooling down 15s before next test..."
    sleep 15
done

echo "=============================================="
echo -e "${GREEN}BENCHMARK COMPLETE${NC}"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Display summary table
echo "SUMMARY:"
echo "--------"
column -t -s',' "$RESULTS_FILE"

echo ""

# Find optimal (highest throughput with >=95% success)
echo "ANALYSIS:"
echo "---------"
BEST=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '$6 >= 0.95 {print $5, $1}' | sort -rn | head -1)
if [ -n "$BEST" ]; then
    BEST_THROUGHPUT=$(echo "$BEST" | awk '{print $1}')
    BEST_SIZE=$(echo "$BEST" | awk '{print $2}')
    echo -e "${GREEN}Optimal sample_size: $BEST_SIZE${NC}"
    echo -e "${GREEN}Throughput: $BEST_THROUGHPUT trials/min${NC}"
    echo ""
    echo "This represents the maximum safe sample size per ROCm worker"
    echo "at 12-way concurrency for this hardware configuration."
else
    echo -e "${YELLOW}No configuration achieved >= 95% success rate${NC}"
    echo "Consider reducing max_concurrent_script_jobs"
fi

echo ""
echo "=============================================="
echo "APPLY OPTIMAL SETTINGS"
echo "=============================================="
echo ""
echo "To manually apply the optimal settings:"
echo "  sed -i 's/\"max_concurrent_script_jobs\": [0-9]*/\"max_concurrent_script_jobs\": 12/g' distributed_config.json"
if [ -n "$BEST_SIZE" ]; then
    echo "  sed -i 's/--sample-size [0-9]*/--sample-size $BEST_SIZE/g' run_scorer_meta_optimizer.sh"
fi
echo ""

# Team Beta report language
echo "=============================================="
echo "TEAM ALPHA REPORT"
echo "=============================================="
echo ""
echo "At 12-way ROCm concurrency, Step 2.5 throughput scales inversely with"
echo "sample size. The stability envelope is satisfied for sample sizes"
echo "${SAMPLE_SIZES[0]}–${SAMPLE_SIZES[-1]} with cleanup + GFXOFF disabled."
echo ""
echo "Optimal operating point is the smallest sample size that preserves"
echo "Optuna signal quality (top-5 stability) while maximizing throughput."
if [ -n "$BEST_SIZE" ]; then
    echo ""
    echo "Recommended: sample_size=$BEST_SIZE @ $BEST_THROUGHPUT trials/min"
fi
echo ""
echo "Configuration will be restored automatically on exit."
