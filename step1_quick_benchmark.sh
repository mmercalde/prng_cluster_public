#!/bin/bash
# step1_quick_benchmark.sh — Fast targeted memory/stability tests
# Total runtime: ~15 minutes
# Run from ~/distributed_prng_analysis on Zeus
#
# v2: Fixed GPU error detection (excludes Partitions column N/A)
#     Fixed output parsing (coordinator output separated from timing)

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs"
mkdir -p "$OUTDIR"
REPORT="$OUTDIR/STEP1_BENCHMARK_REPORT.md"
PRNG="java_lcg"
LOTTERY="daily3.json"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================
# Helper: capture memory snapshot (compact)
# ============================================================
snap_memory() {
    local LABEL=$1
    echo "  [$LABEL]"
    echo "--- $LABEL $(date +%H:%M:%S) ---" >> "$MEMLOG"
    
    echo "Zeus RAM:" >> "$MEMLOG"
    free -m | grep Mem >> "$MEMLOG" 2>&1
    echo "Zeus VRAM:" >> "$MEMLOG"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$MEMLOG" 2>&1
    
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        {
            HN=$(ssh -o ConnectTimeout=3 $node hostname 2>/dev/null || echo "$node")
            echo "$HN RAM:" >> "$MEMLOG"
            ssh -o ConnectTimeout=3 $node "free -m | grep Mem" >> "$MEMLOG" 2>&1
        } &
    done
    wait
    echo "" >> "$MEMLOG"
}

# ============================================================
# Helper: run single sieve, sets SIEVE_ELAPSED and SIEVE_EXIT
# ============================================================
run_sieve() {
    local SEEDS=$1
    local CONC=$2
    local TMPLOG=$(mktemp /tmp/sieve_XXXXXX.log)
    local START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --resume-policy restart \
        --max-concurrent "$CONC" \
        --method residue_sieve \
        --prng-type "$PRNG" \
        --window-size 512 \
        --skip-max 20 \
        --seeds "$SEEDS" \
        > "$TMPLOG" 2>&1
    
    SIEVE_EXIT=$?
    local END=$(date +%s)
    SIEVE_ELAPSED=$((END - START))
    
    # Show last 3 lines for context
    tail -3 "$TMPLOG"
    rm -f "$TMPLOG"
}

# ============================================================
# Helper: count REAL gpu errors
# Only counts GPUs showing "unknown" in Perf column
# (NOT the normal "N/A, N/A, 0" in Partitions column)
# ============================================================
count_real_errors() {
    local TOTAL=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        local E=$(ssh -o ConnectTimeout=3 $node "rocm-smi 2>&1 | grep '0x73ff' | grep -c 'unknown'" 2>/dev/null | tr -cd '0-9')
        [ -z "$E" ] && E=0
        TOTAL=$((TOTAL + E))
    done
    echo "$TOTAL"
}

# ============================================================
# Start
# ============================================================
echo "=========================================="
echo "STEP 1 QUICK BENCHMARK v2"
echo "Started: $TIMESTAMP"
echo "=========================================="

# Verify GPU count first
echo ""
echo "Cluster GPU check:"
ZEUS_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
RIG1_GPUS=$(ssh -o ConnectTimeout=3 192.168.3.120 "rocm-smi 2>&1 | grep -c '0x73ff'" 2>/dev/null || echo "0")
RIG2_GPUS=$(ssh -o ConnectTimeout=3 192.168.3.154 "rocm-smi 2>&1 | grep -c '0x73ff'" 2>/dev/null || echo "0")
RIG3_GPUS=$(ssh -o ConnectTimeout=3 192.168.3.162 "rocm-smi 2>&1 | grep -c '0x73ff'" 2>/dev/null || echo "0")
TOTAL_GPUS=$((ZEUS_GPUS + RIG1_GPUS + RIG2_GPUS + RIG3_GPUS))
echo "  Zeus: $ZEUS_GPUS | rig-6600: $RIG1_GPUS | rig-6600b: $RIG2_GPUS | rig-6600c: $RIG3_GPUS | Total: $TOTAL_GPUS"

# Start report
cat > "$REPORT" << EOF
# Step 1 Quick Benchmark Report
**Generated:** $TIMESTAMP
**Cluster:** Zeus ($ZEUS_GPUS GPU) + rig-6600 ($RIG1_GPUS GPU) + rig-6600b ($RIG2_GPUS GPU) + rig-6600c ($RIG3_GPUS GPU) = $TOTAL_GPUS GPUs

---

EOF

# ============================================================
# TEST 1: Seed Count Memory Impact (single sieve each)
# ============================================================
echo ""
echo "=========================================="
echo "TEST 1: Seed Count Memory Impact"
echo "=========================================="

MEMLOG="$OUTDIR/test1_memory.log"
> "$MEMLOG"

SEED_COUNTS=(10000 50000 100000 500000 1000000 5000000)

echo "" >> "$REPORT"
echo "## Test 1: Seed Count — Single Sieve Each" >> "$REPORT"
echo "" >> "$REPORT"
echo "| seed_count | Time (s) | Exit | GPU Errors | Status |" >> "$REPORT"
echo "|-----------|---------|------|------------|--------|" >> "$REPORT"

for SEEDS in "${SEED_COUNTS[@]}"; do
    echo ""
    echo "--- seed_count=$SEEDS ---"
    
    snap_memory "BEFORE_${SEEDS}"
    
    run_sieve $SEEDS 26
    
    snap_memory "AFTER_${SEEDS}"
    
    ERRORS=$(count_real_errors)
    
    if [ "$SIEVE_EXIT" = "0" ] && [ "$ERRORS" = "0" ]; then
        STATUS="✅ PASS"
    elif [ "$ERRORS" != "0" ]; then
        STATUS="⚠️ $ERRORS GPUs"
    else
        STATUS="❌ exit=$SIEVE_EXIT"
    fi
    
    echo "  → seed_count=$SEEDS | ${SIEVE_ELAPSED}s | exit=$SIEVE_EXIT | gpu_errors=$ERRORS | $STATUS"
    printf "| %s | %s | %s | %s | %s |\n" "$SEEDS" "$SIEVE_ELAPSED" "$SIEVE_EXIT" "$ERRORS" "$STATUS" >> "$REPORT"
    
    sleep 10
done

echo "" >> "$REPORT"

# ============================================================
# TEST 2: Concurrency Impact (single sieve each)
# ============================================================
echo ""
echo "=========================================="
echo "TEST 2: Concurrency Impact"
echo "=========================================="

MEMLOG="$OUTDIR/test2_memory.log"
> "$MEMLOG"

CONC_LEVELS=(8 16 26)

echo "## Test 2: Concurrency — 500K Seeds Each" >> "$REPORT"
echo "" >> "$REPORT"
echo "| max-concurrent | Time (s) | Exit | GPU Errors | Status |" >> "$REPORT"
echo "|---------------|---------|------|------------|--------|" >> "$REPORT"

for CONC in "${CONC_LEVELS[@]}"; do
    echo ""
    echo "--- max-concurrent=$CONC ---"
    
    snap_memory "BEFORE_CONC${CONC}"
    
    run_sieve 500000 $CONC
    
    snap_memory "AFTER_CONC${CONC}"
    
    ERRORS=$(count_real_errors)
    
    if [ "$SIEVE_EXIT" = "0" ] && [ "$ERRORS" = "0" ]; then
        STATUS="✅ PASS"
    elif [ "$ERRORS" != "0" ]; then
        STATUS="⚠️ $ERRORS GPUs"
    else
        STATUS="❌ exit=$SIEVE_EXIT"
    fi
    
    echo "  → conc=$CONC | ${SIEVE_ELAPSED}s | exit=$SIEVE_EXIT | gpu_errors=$ERRORS | $STATUS"
    printf "| %s | %s | %s | %s | %s |\n" "$CONC" "$SIEVE_ELAPSED" "$SIEVE_EXIT" "$ERRORS" "$STATUS" >> "$REPORT"
    
    sleep 10
done

echo "" >> "$REPORT"

# ============================================================
# TEST 3: Back-to-Back Stress (rapid fire, no cooldown)
# ============================================================
echo ""
echo "=========================================="
echo "TEST 3: Back-to-Back Stress"
echo "10 sieves, 500K seeds, NO cooldown"
echo "=========================================="

MEMLOG="$OUTDIR/test3_memory.log"
> "$MEMLOG"

RAPID_COUNT=10
RAPID_SEEDS=500000

echo "## Test 3: Rapid Fire — $RAPID_COUNT Sieves, No Cooldown" >> "$REPORT"
echo "" >> "$REPORT"
echo "| Sieve # | Time (s) | Exit | GPU Errors | Status |" >> "$REPORT"
echo "|---------|---------|------|------------|--------|" >> "$REPORT"

snap_memory "BEFORE_STRESS"

STRESS_FAILED=false
for i in $(seq 1 $RAPID_COUNT); do
    echo "  Sieve $i/$RAPID_COUNT..."
    
    run_sieve $RAPID_SEEDS 26
    
    ERRORS=$(count_real_errors)
    
    if [ "$SIEVE_EXIT" = "0" ] && [ "$ERRORS" = "0" ]; then
        STATUS="✅"
    elif [ "$ERRORS" != "0" ]; then
        STATUS="⚠️ $ERRORS GPUs"
        STRESS_FAILED=true
    else
        STATUS="❌ exit=$SIEVE_EXIT"
        STRESS_FAILED=true
    fi
    
    echo "  → Sieve $i | ${SIEVE_ELAPSED}s | gpu_errors=$ERRORS | $STATUS"
    printf "| %s | %s | %s | %s | %s |\n" "$i" "$SIEVE_ELAPSED" "$SIEVE_EXIT" "$ERRORS" "$STATUS" >> "$REPORT"
    
    if [ "$STRESS_FAILED" = true ]; then
        echo "  🛑 Stopping — GPU anomaly at sieve $i"
        echo "" >> "$REPORT"
        echo "**🛑 Stopped at sieve $i — GPU anomaly detected.**" >> "$REPORT"
        break
    fi
done

snap_memory "AFTER_STRESS"

echo "" >> "$REPORT"
if [ "$STRESS_FAILED" = false ]; then
    echo "**✅ All $RAPID_COUNT consecutive sieves passed.**" >> "$REPORT"
fi
echo "" >> "$REPORT"

# ============================================================
# TEST 4: Recovery Check
# ============================================================
echo ""
echo "=========================================="
echo "TEST 4: Recovery (30s wait)"
echo "=========================================="

MEMLOG="$OUTDIR/test4_memory.log"
> "$MEMLOG"

echo "## Test 4: Recovery Check" >> "$REPORT"
echo "" >> "$REPORT"

snap_memory "IMMEDIATE"
echo "  Waiting 30s..."
sleep 30
snap_memory "AFTER_30s"

FINAL_ERRORS=$(count_real_errors)
if [ "$FINAL_ERRORS" = "0" ]; then
    echo "  ✅ Cluster clean"
    echo "**✅ No residual GPU anomalies after 30s recovery.**" >> "$REPORT"
else
    echo "  ⚠️ $FINAL_ERRORS residual errors"
    echo "**⚠️ $FINAL_ERRORS residual GPU anomalies.**" >> "$REPORT"
fi
echo "" >> "$REPORT"

# ============================================================
# Summary
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Configuration Recommendation" >> "$REPORT"
echo "" >> "$REPORT"
echo '```bash' >> "$REPORT"
echo "# Fill in optimal values from results above:" >> "$REPORT"
echo "opt_seed_count=           # Highest PASS from Test 1" >> "$REPORT"
echo "max_concurrent=           # Highest PASS from Test 2" >> "$REPORT"
echo "inter_trial_cooldown=     # If Test 3 failed: required; if passed: optional" >> "$REPORT"
echo '```' >> "$REPORT"
echo "" >> "$REPORT"

echo "## Raw Logs" >> "$REPORT"
echo "" >> "$REPORT"
echo "- \`$OUTDIR/test1_memory.log\` — Seed count memory snapshots" >> "$REPORT"
echo "- \`$OUTDIR/test2_memory.log\` — Concurrency memory snapshots" >> "$REPORT"
echo "- \`$OUTDIR/test3_memory.log\` — Stress test memory snapshots" >> "$REPORT"
echo "- \`$OUTDIR/test4_memory.log\` — Recovery snapshots" >> "$REPORT"
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "*Generated: $(date '+%Y-%m-%d %H:%M:%S')*" >> "$REPORT"

# ============================================================
# Print report
# ============================================================
echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
echo ""
cat "$REPORT"
echo ""
echo "Report saved: $REPORT"
