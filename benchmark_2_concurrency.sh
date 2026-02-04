#!/bin/bash
# Benchmark 2: Concurrency Scaling
# Tests different --max-concurrent values with fixed seed count.
# Run from ~/distributed_prng_analysis on Zeus.
#
# Purpose: Verify whether full 26-GPU concurrency is safe for seed-range
# jobs, or if we need node-level limits like Steps 2-3.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench2_concurrency"
mkdir -p "$OUTDIR"

CONCURRENCY_LEVELS=(8 16 20 26)
SEEDS=500000
TRIALS=5
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 2: CONCURRENCY SCALING"
echo "Seed count: $SEEDS"
echo "Trials per test: $TRIALS"
echo "PRNG: $PRNG"
echo "Concurrency levels: ${CONCURRENCY_LEVELS[*]}"
echo "=========================================="
echo ""

./gpu_health_check.sh "bench2_pre"

for CONC in "${CONCURRENCY_LEVELS[@]}"; do
    LABEL="conc_${CONC}"
    LOGFILE="$OUTDIR/${LABEL}_details.log"
    
    echo ""
    echo "=========================================="
    echo "Testing max-concurrent=$CONC"
    echo "=========================================="
    
    ./gpu_health_check.sh "bench2_pre_${LABEL}"
    
    # Record host memory before
    echo "--- Host Memory Before ---" >> "$LOGFILE"
    for node in localhost 192.168.3.120 192.168.3.154 192.168.3.162; do
        if [ "$node" = "localhost" ]; then
            echo "Zeus:" >> "$LOGFILE"
            free -m | head -2 >> "$LOGFILE" 2>&1
        else
            HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
            echo "$HOSTNAME:" >> "$LOGFILE"
            ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
        fi
    done
    
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --optimize-window \
        --prng-type "$PRNG" \
        --opt-strategy bayesian \
        --opt-iterations "$TRIALS" \
        --opt-seed-count "$SEEDS" \
        --max-concurrent "$CONC" \
        --resume-policy restart \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Record host memory after
    echo "" >> "$LOGFILE"
    echo "--- Host Memory After ---" >> "$LOGFILE"
    for node in localhost 192.168.3.120 192.168.3.154 192.168.3.162; do
        if [ "$node" = "localhost" ]; then
            echo "Zeus:" >> "$LOGFILE"
            free -m | head -2 >> "$LOGFILE" 2>&1
        else
            HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
            echo "$HOSTNAME:" >> "$LOGFILE"
            ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
        fi
    done
    
    ./gpu_health_check.sh "bench2_post_${LABEL}"
    
    # Check for NEW GPU anomalies
    GPU_ERRORS=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
        GPU_ERRORS=$((GPU_ERRORS + ERRORS))
    done
    KNOWN_BAD=1
    NEW_ERRORS=$((GPU_ERRORS - KNOWN_BAD))
    if [ "$NEW_ERRORS" -lt 0 ]; then NEW_ERRORS=0; fi
    
    STATUS="PASS"
    if [ "$NEW_ERRORS" -gt 0 ]; then
        STATUS="⚠️  FAIL ($NEW_ERRORS new anomalies)"
    fi
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="❌ CRASH (exit=$EXIT_CODE)"
    fi
    
    THROUGHPUT=$(echo "scale=0; $SEEDS * $TRIALS * 2 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
    
    RESULT="max-concurrent=$CONC | time=${ELAPSED}s | status=$STATUS | throughput=${THROUGHPUT} seeds/sec"
    echo "$RESULT"
    echo "$RESULT" >> "$OUTDIR/summary.txt"
    
    # 30-second cooldown
    echo "Cooling down 30s..."
    sleep 30
done

./gpu_health_check.sh "bench2_post"

echo ""
echo "=========================================="
echo "BENCHMARK 2 COMPLETE"
echo "Results:"
cat "$OUTDIR/summary.txt"
echo ""
echo "KEY: If throughput doesn't scale linearly with concurrency,"
echo "that suggests host memory contention on mining rigs."
echo "=========================================="
