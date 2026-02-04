#!/bin/bash
# Benchmark 4: Single-Sieve Memory Profile
# Runs SINGLE forward sieves at different seed counts while capturing
# detailed memory snapshots before, during, and after.
# Run from ~/distributed_prng_analysis on Zeus.
#
# Purpose: Deep-dive into what happens to VRAM and host RAM during
# sieve operations. The "microscope" test — fewer trials, more granular data.
#
# RUN THIS FIRST — it's the safest benchmark and gives the memory baseline.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench4_memory"
mkdir -p "$OUTDIR"

SEED_COUNTS=(50000 500000 5000000 10000000)
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 4: SINGLE-SIEVE MEMORY PROFILE"
echo "Max concurrent: $MAX_CONCURRENT"
echo "PRNG: $PRNG"
echo "Seed counts: ${SEED_COUNTS[*]}"
echo ""
echo "This runs individual forward sieves (NOT window optimizer)."
echo "Lowest risk benchmark — run this first."
echo "=========================================="
echo ""

capture_memory() {
    local LABEL=$1
    local OUTFILE=$2
    echo "--- Memory snapshot: $LABEL ---" >> "$OUTFILE"
    echo "Timestamp: $(date +%H:%M:%S)" >> "$OUTFILE"
    
    # Zeus
    echo "Zeus host RAM:" >> "$OUTFILE"
    free -m | head -2 >> "$OUTFILE" 2>&1
    echo "Zeus VRAM:" >> "$OUTFILE"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$OUTFILE" 2>&1
    
    # ROCm rigs
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME host RAM:" >> "$OUTFILE"
        ssh $node "free -m | head -2" >> "$OUTFILE" 2>&1
        echo "$HOSTNAME GPU states:" >> "$OUTFILE"
        # Get compact GPU info (temp, power, memory used)
        ssh $node "rocm-smi --showtemp --showpower --showmemuse 2>/dev/null || rocm-smi 2>&1" >> "$OUTFILE" 2>&1
    done
    echo "" >> "$OUTFILE"
}

./gpu_health_check.sh "bench4_pre"

for SEEDS in "${SEED_COUNTS[@]}"; do
    LABEL="mem_seeds_${SEEDS}"
    MEMLOG="$OUTDIR/${LABEL}_memory.log"
    
    echo ""
    echo "=========================================="
    echo "Memory profiling: seed_count=$SEEDS (single forward sieve)"
    echo "=========================================="
    
    # Capture BEFORE
    capture_memory "BEFORE" "$MEMLOG"
    
    # Run single forward sieve
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --resume-policy restart \
        --max-concurrent "$MAX_CONCURRENT" \
        --method residue_sieve \
        --prng-type "$PRNG" \
        --window-size 512 \
        --skip-max 20 \
        --seeds "$SEEDS" \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Capture AFTER (immediately — before natural cleanup)
    capture_memory "AFTER_IMMEDIATE" "$MEMLOG"
    
    # Wait 10 seconds, capture RECOVERED state
    sleep 10
    capture_memory "RECOVERED_10s" "$MEMLOG"
    
    # Wait 30 more seconds, capture fully-recovered state
    sleep 30
    capture_memory "RECOVERED_40s" "$MEMLOG"
    
    # Check for anomalies
    GPU_ERRORS=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
        GPU_ERRORS=$((GPU_ERRORS + ERRORS))
    done
    KNOWN_BAD=1
    NEW_ERRORS=$((GPU_ERRORS - KNOWN_BAD))
    if [ "$NEW_ERRORS" -lt 0 ]; then NEW_ERRORS=0; fi
    
    SURVIVORS=$(grep -oP "survivors.*?(\d+)" "$OUTDIR/${LABEL}_output.log" | tail -1 || echo "check log")
    
    RESULT="seed_count=$SEEDS | time=${ELAPSED}s | exit=$EXIT_CODE | new_gpu_errors=$NEW_ERRORS | $SURVIVORS"
    echo "$RESULT"
    echo "$RESULT" >> "$OUTDIR/summary.txt"
    
    echo ""
    echo "Memory log: $MEMLOG"
    echo "Compare BEFORE vs AFTER_IMMEDIATE vs RECOVERED in the log."
    echo ""
    
    # 30-second cooldown between tests
    sleep 30
done

./gpu_health_check.sh "bench4_post"

echo ""
echo "=========================================="
echo "BENCHMARK 4 COMPLETE"
echo "Results:"
cat "$OUTDIR/summary.txt"
echo ""
echo "ANALYSIS: Check memory logs in $OUTDIR/"
echo "Compare BEFORE → AFTER → RECOVERED for each seed count."
echo "Look for: host RAM that doesn't recover, VRAM leaks, temp spikes."
echo "=========================================="
