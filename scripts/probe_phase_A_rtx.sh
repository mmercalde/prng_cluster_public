#!/usr/bin/env bash
# =============================================================================
# PHASE A — RTX 3080 Ti ISOLATED SINGLE-CARD CEILING PROBE
# S128 GPU Throughput Investigation
# Run on: Zeus
# Usage: bash probe_phase_A_rtx.sh 2>&1 | tee /tmp/probe_A_rtx.log
# =============================================================================

set -eo pipefail

PRNG_DIR="$HOME/distributed_prng_analysis"
VENV="$HOME/venvs/torch/bin/activate"
RESULTS_FILE="/tmp/probe_A_rtx_results.csv"
GPU_ID=0  # Isolate to gpu0 only (one RTX 3080 Ti)

cd "$PRNG_DIR"
source "$VENV"

echo "================================================"
echo " Phase A — RTX 3080 Ti Isolated Ceiling Probe"
echo " GPU: $GPU_ID | Date: $(date)"
echo "================================================"
echo ""
echo "seed_count,seeds_per_sec,peak_vram_mb,duration_sec,status" > "$RESULTS_FILE"

run_probe() {
    local SEEDS=$1
    local STEP_LABEL=$2

    echo "--- Step $STEP_LABEL: $SEEDS seeds ---"

    # Build minimal probe job file
    cat > /tmp/probe_rtx_job.json << EOF
{
  "job_id": "probe_rtx_${STEP_LABEL}",
  "gpu_id": $GPU_ID,
  "seed_start": 0,
  "seed_count": $SEEDS,
  "dataset_path": "$PRNG_DIR/daily3.json",
  "output_file": "/tmp/probe_rtx_${STEP_LABEL}_out.json",
  "window_size": 8,
  "offset": 43,
  "prng_type": "java_lcg",
  "forward_threshold": 0.25,
  "reverse_threshold": 0.25,
  "test_both_modes": false
}
EOF

    # Start VRAM monitor in background
    nvidia-smi --query-gpu=memory.used --format=csv,noheader \
        --id=$GPU_ID -l 1 > /tmp/probe_vram_rtx.log 2>&1 &
    MONITOR_PID=$!

    START_TS=$(date +%s%3N)
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 sieve_filter.py \
        --job-file /tmp/probe_rtx_job.json --gpu-id 0 2>&1
    EXIT_CODE=$?
    END_TS=$(date +%s%3N)

    kill $MONITOR_PID 2>/dev/null || true

    DURATION_MS=$(( END_TS - START_TS ))
    DURATION_SEC=$(echo "scale=2; $DURATION_MS / 1000" | bc)
    PEAK_VRAM=$(sort -n /tmp/probe_vram_rtx.log | tail -1 | tr -d ' MiB')

    if [ $EXIT_CODE -eq 0 ]; then
        SPS=$(echo "scale=0; $SEEDS * 1000 / $DURATION_MS" | bc)
        STATUS="OK"
        echo "  ✅ $SEEDS seeds | ${SPS} seeds/sec | ${PEAK_VRAM} MB VRAM | ${DURATION_SEC}s"
        echo "$SEEDS,$SPS,$PEAK_VRAM,$DURATION_SEC,OK" >> "$RESULTS_FILE"
    else
        STATUS="FAIL"
        echo "  ❌ FAILED at $SEEDS seeds (exit=$EXIT_CODE) — STOPPING LADDER"
        echo "$SEEDS,0,$PEAK_VRAM,$DURATION_SEC,FAIL" >> "$RESULTS_FILE"
        echo ""
        echo "Results saved to $RESULTS_FILE"
        exit 0
    fi

    # Check for >20% throughput regression vs prior step
    local LINES=$(wc -l < "$RESULTS_FILE")
    if [ "$LINES" -gt 2 ]; then
        local PREV_SPS=$(tail -2 "$RESULTS_FILE" | head -1 | cut -d',' -f2)
        local RATIO=$(echo "scale=2; $SPS * 100 / $PREV_SPS" | bc)
        if (( $(echo "$RATIO < 80" | bc -l) )); then
            echo "  ⚠️  THROUGHPUT REGRESSION: ${RATIO}% of prior step — STOPPING LADDER"
            echo "  Ceiling is the prior step's seed count."
            break
        fi
    fi

    echo ""
}

# Step ladder: 100k → 500k → 1M → 2M → 5M
run_probe 100000  "A1"
run_probe 500000  "A2"
run_probe 1000000 "A3"
run_probe 2000000 "A4"
run_probe 5000000 "A5"

echo "================================================"
echo " Phase A RTX COMPLETE"
echo " Results: $RESULTS_FILE"
echo "================================================"
cat "$RESULTS_FILE"
