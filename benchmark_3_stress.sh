#!/bin/bash
# Benchmark 3: Multi-Trial Stress Test
# Runs increasing numbers of consecutive Window Optimizer trials to find
# the stability ceiling. STOP if any test shows GPU sensor anomalies.
# Run from ~/distributed_prng_analysis on Zeus.
#
# Purpose: Find how many back-to-back trials the cluster can handle
# without GPU degradation.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench3_stress"
mkdir -p "$OUTDIR"

TRIAL_COUNTS=(5 10 20 30 50)
SEEDS=500000
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 3: MULTI-TRIAL STRESS TEST"
echo "Seed count: $SEEDS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "PRNG: $PRNG"
echo "Trial counts to test: ${TRIAL_COUNTS[*]}"
echo ""
echo "⚠️  This is the stress test. Watch GPU monitors closely."
echo "⚠️  Each test runs from scratch (not cumulative)."
echo "=========================================="
echo ""

./gpu_health_check.sh "bench3_pre"

for TRIALS in "${TRIAL_COUNTS[@]}"; do
    LABEL="trials_${TRIALS}"
    LOGFILE="$OUTDIR/${LABEL}_details.log"
    
    echo ""
    echo "=========================================="
    echo "🔥 Testing $TRIALS consecutive trials"
    echo "=========================================="
    echo "⚠️  Watch GPU monitors!"
    echo "⚠️  If you see NEW N/A sensors, Ctrl+C and note the trial count."
    echo ""
    
    ./gpu_health_check.sh "bench3_pre_${LABEL}"
    
    # Record initial state
    echo "--- Initial State ---" >> "$LOGFILE"
    date >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME RAM:" >> "$LOGFILE"
        ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
    done
    
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
    
    # Record final state
    echo "" >> "$LOGFILE"
    echo "--- Final State ---" >> "$LOGFILE"
    date >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME RAM:" >> "$LOGFILE"
        ssh $node "free -m | head -2" >> "$LOGFILE" 2>&1
    done
    
    ./gpu_health_check.sh "bench3_post_${LABEL}"
    
    # Check for GPU anomalies
    GPU_ERRORS=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
        GPU_ERRORS=$((GPU_ERRORS + ERRORS))
    done
    
    # Subtract known-bad GPU[4] on rig-6600b
    KNOWN_BAD=1
    NEW_ERRORS=$((GPU_ERRORS - KNOWN_BAD))
    if [ "$NEW_ERRORS" -lt 0 ]; then NEW_ERRORS=0; fi
    
    STATUS="✅ PASS"
    if [ "$NEW_ERRORS" -gt 0 ]; then
        STATUS="⚠️  FAIL ($NEW_ERRORS new anomalies)"
    fi
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="❌ CRASH (exit=$EXIT_CODE)"
    fi
    
    RATE=$(echo "scale=2; $TRIALS * 60 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
    
    RESULT="trials=$TRIALS | time=${ELAPSED}s | rate=${RATE} trials/min | status=$STATUS"
    echo "$RESULT"
    echo "$RESULT" >> "$OUTDIR/summary.txt"
    
    # If we got errors, STOP — we've found the ceiling
    if [ "$NEW_ERRORS" -gt 0 ] || [ "$EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "🛑 ==========================================="
        echo "🛑 STOPPING: GPU anomalies detected at $TRIALS trials."
        echo "🛑 Stability ceiling: < $TRIALS consecutive trials"
        echo "🛑 ==========================================="
        echo ""
        echo "Recommended next steps:"
        echo "  1. Reboot affected rigs"
        echo "  2. The fix: add inter-trial cooldown + cleanup to window_optimizer.py"
        echo "  3. Re-test at the failed trial count with protections enabled"
        echo ""
        echo "STABILITY_CEILING=less_than_$TRIALS" >> "$OUTDIR/summary.txt"
        break
    fi
    
    # 60-second cooldown between stress levels
    echo "Cooling down 60s between stress levels..."
    sleep 60
done

./gpu_health_check.sh "bench3_post"

echo ""
echo "=========================================="
echo "BENCHMARK 3 COMPLETE"
echo "Results:"
cat "$OUTDIR/summary.txt"
echo "=========================================="
