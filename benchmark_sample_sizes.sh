#!/bin/bash
# Benchmark script to find optimal sample size for 12 concurrent GPU jobs
# Usage: ./benchmark_sample_sizes.sh
#
# This script tests different sample sizes and measures throughput
# to find the optimal configuration for the cluster.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
TRIALS_PER_TEST=20
SAMPLE_SIZES=(500 1000 2000 3000 4000)
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
LOG_DIR="benchmark_logs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  SAMPLE SIZE BENCHMARK - 12 Concurrent GPUs"
echo "=============================================="
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Backup current config
echo -e "${YELLOW}Backing up current configuration...${NC}"
cp distributed_config.json distributed_config.json.bak_benchmark
cp run_scorer_meta_optimizer.sh run_scorer_meta_optimizer.sh.bak_benchmark

# Set concurrency to 12 for both rigs
echo -e "${YELLOW}Setting max_concurrent_script_jobs to 12...${NC}"
sed -i 's/"max_concurrent_script_jobs": [0-9]*/"max_concurrent_script_jobs": 12/g' distributed_config.json

# Verify
echo "Concurrency settings:"
grep "max_concurrent_script_jobs" distributed_config.json

# Initialize results CSV
echo "sample_size,trials,total_runtime_s,avg_trial_time_s,throughput_trials_per_min,success_rate" > "$RESULTS_FILE"

echo ""
echo -e "${CYAN}Starting benchmark with ${#SAMPLE_SIZES[@]} sample sizes...${NC}"
echo ""

for SIZE in "${SAMPLE_SIZES[@]}"; do
    echo "=============================================="
    echo -e "${GREEN}Testing sample_size = $SIZE${NC}"
    echo "=============================================="
    
    # Update sample size in script
    sed -i "s/--sample-size [0-9]*/--sample-size $SIZE/g" run_scorer_meta_optimizer.sh
    
    # Verify change
    echo "Config: $(grep 'sample-size' run_scorer_meta_optimizer.sh | head -1)"
    
    LOG_FILE="$LOG_DIR/benchmark_size_${SIZE}.log"
    
    # Run the benchmark
    echo "Running $TRIALS_PER_TEST trials..."
    START_TIME=$(date +%s.%N)
    
    # Run via WATCHER (captures full output)
    PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 2>&1 | tee "$LOG_FILE"
    
    END_TIME=$(date +%s.%N)
    TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Parse results from log
    SUCCESSFUL=$(grep -c "^✓" "$LOG_FILE" || echo "0")
    FAILED=$(grep -c "^✗" "$LOG_FILE" || echo "0")
    TOTAL=$((SUCCESSFUL + FAILED))
    
    if [ "$TOTAL" -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=2; $SUCCESSFUL / $TOTAL" | bc)
        AVG_TRIAL_TIME=$(echo "scale=2; $TOTAL_TIME / $TOTAL" | bc)
        THROUGHPUT=$(echo "scale=2; ($SUCCESSFUL * 60) / $TOTAL_TIME" | bc)
    else
        SUCCESS_RATE=0
        AVG_TRIAL_TIME=0
        THROUGHPUT=0
    fi
    
    # Record results
    echo "$SIZE,$TOTAL,$TOTAL_TIME,$AVG_TRIAL_TIME,$THROUGHPUT,$SUCCESS_RATE" >> "$RESULTS_FILE"
    
    echo ""
    echo -e "${GREEN}Results for sample_size=$SIZE:${NC}"
    echo "  Total runtime: ${TOTAL_TIME}s"
    echo "  Successful: $SUCCESSFUL / $TOTAL"
    echo "  Avg trial time: ${AVG_TRIAL_TIME}s"
    echo "  Throughput: ${THROUGHPUT} trials/min"
    echo ""
    
    # Brief pause between tests
    echo "Pausing 10s before next test..."
    sleep 10
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

# Find optimal (highest throughput with 100% success)
echo "ANALYSIS:"
echo "---------"
BEST=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '$6 >= 0.95 {print $5, $1}' | sort -rn | head -1)
if [ -n "$BEST" ]; then
    BEST_THROUGHPUT=$(echo "$BEST" | awk '{print $1}')
    BEST_SIZE=$(echo "$BEST" | awk '{print $2}')
    echo -e "${GREEN}Optimal sample_size: $BEST_SIZE (throughput: $BEST_THROUGHPUT trials/min)${NC}"
else
    echo -e "${YELLOW}No configuration achieved >= 95% success rate${NC}"
fi

echo ""
echo "Restoring original configuration..."
echo "(Run the commands below manually if you want to apply optimal settings)"
echo ""
echo "  # To restore original:"
echo "  cp distributed_config.json.bak_benchmark distributed_config.json"
echo "  cp run_scorer_meta_optimizer.sh.bak_benchmark run_scorer_meta_optimizer.sh"
echo ""
echo "  # To apply optimal (if found):"
echo "  sed -i 's/--sample-size [0-9]*/--sample-size $BEST_SIZE/g' run_scorer_meta_optimizer.sh"
echo ""
