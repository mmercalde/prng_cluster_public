#!/bin/bash
# Benchmark 1: Seed Count Scaling
# Tests different seed_count values with fixed 3 trials to find throughput sweet spot.
# Run from ~/distributed_prng_analysis on Zeus.
#
# Purpose: Find optimal seed_count per sieve operation — analogous to
# finding sample_size=450 for Step 2.5.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench1_seed_scaling"
mkdir -p "$OUTDIR"

SEED_COUNTS=(10000 50000 100000 500000 1000000 5000000 10000000)
TRIALS=3
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 1: SEED COUNT SCALING"
echo "Trials per test: $TRIALS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "PRNG: $PRNG"
echo "Test points: ${SEED_COUNTS[*]}"
echo "=========================================="
echo ""

# Pre-benchmark health check
./gpu_health_check.sh "bench1_pre"

for SEEDS in "${SEED_COUNTS[@]}"; do
    LABEL="seeds_${SEEDS}"
    LOGFILE="$OUTDIR/${LABEL}_details.log"
    
    echo ""
    echo "=========================================="
    echo "Testing seed_count=$SEEDS"
    echo "=========================================="
    
    # Pre-test health check
    ./gpu_health_check.sh "bench1_pre_${LABEL}"
    
    # Record host memory before
    echo "--- Host Memory Before ---" >> "$LOGFILE"
    echo "Zeus:" >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME:" >> "$LOGFILE"
        ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
    done
    
    # Run the test
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --optimize-window \
        --prng-type "$PRNG" \
        --opt-strategy bayesian \
        --opt-iterations "$TRIALS" \
        --opt-seed-count "$SEEDS" \
        --max-concurrent "$MAX_CONCURRENT" \
        --resume-policy restart \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Record host memory after
    echo "" >> "$LOGFILE"
    echo "--- Host Memory After ---" >> "$LOGFILE"
    echo "Zeus:" >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME:" >> "$LOGFILE"
        ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
    done
    
    # Post-test health check
    ./gpu_health_check.sh "bench1_post_${LABEL}"
    
    # Check for GPU anomalies (exclude rig-6600b GPU[4] known issue)
    GPU_ERRORS=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
        GPU_ERRORS=$((GPU_ERRORS + ERRORS))
    done
    
    # rig-6600b GPU[4] always shows 1 anomaly — subtract it
    # Adjust this if your known-broken GPU count changes
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
    
    # Throughput: total seeds processed = seed_count * trials * 2 (fwd + rev)
    THROUGHPUT=$(echo "scale=0; $SEEDS * $TRIALS * 2 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
    
    RESULT="seed_count=$SEEDS | time=${ELAPSED}s | status=$STATUS | throughput=${THROUGHPUT} seeds/sec"
    echo "$RESULT"
    echo "$RESULT" >> "$OUTDIR/summary.txt"
    
    # 30-second cooldown between tests
    echo "Cooling down 30s..."
    sleep 30
done

# Post-benchmark health check
./gpu_health_check.sh "bench1_post"

echo ""
echo "=========================================="
echo "BENCHMARK 1 COMPLETE"
echo "Results in: $OUTDIR/"
echo "Summary:"
cat "$OUTDIR/summary.txt"
echo "=========================================="
