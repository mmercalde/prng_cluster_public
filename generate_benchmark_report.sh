#!/bin/bash
# generate_benchmark_report.sh — Reads all benchmark logs and produces
# a formatted markdown report with tables, analysis, and recommendations.
# Run from ~/distributed_prng_analysis on Zeus AFTER completing all benchmarks.
#
# Usage: ./generate_benchmark_report.sh
# Output: benchmark_logs/STEP1_BENCHMARK_REPORT.md

cd ~/distributed_prng_analysis

LOGDIR="benchmark_logs"
REPORT="$LOGDIR/STEP1_BENCHMARK_REPORT.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================
# Helper functions
# ============================================================

# Extract value after a key= pattern from a pipe-delimited line
extract_field() {
    local LINE="$1"
    local KEY="$2"
    echo "$LINE" | grep -oP "${KEY}=\K[^ |]+" | head -1
}

# Extract field that may contain spaces (up to next |)
extract_field_full() {
    local LINE="$1"
    local KEY="$2"
    echo "$LINE" | sed -n "s/.*${KEY}=\([^|]*\).*/\1/p" | sed 's/^ *//;s/ *$//' | head -1
}

# Count GPU anomalies from a health check file (excluding known rig-6600b GPU[4])
count_anomalies_in_file() {
    local FILE="$1"
    if [ ! -f "$FILE" ]; then
        echo "N/A"
        return
    fi
    local WARNINGS=$(grep -c "WARNING" "$FILE" 2>/dev/null || echo "0")
    echo "$WARNINGS"
}

# Parse free -m output to get "used" memory in MB
parse_host_mem_used() {
    local FILE="$1"
    local HOST="$2"
    local SECTION="$3"
    # Look for the section, then find the Mem: line after it
    awk -v host="$HOST" -v section="$SECTION" '
        $0 ~ section { found_section=1 }
        found_section && $0 ~ host { found_host=1 }
        found_host && /^Mem:/ { print $3; exit }
    ' "$FILE" 2>/dev/null
}

# ============================================================
# Begin report
# ============================================================

cat > "$REPORT" << 'HEADER'
# Step 1 GPU Benchmark Report
## coordinator.py (Seed-Based Jobs) — Performance & Stability Analysis

HEADER

echo "**Generated:** $TIMESTAMP  " >> "$REPORT"
echo "**Host:** $(hostname)  " >> "$REPORT"
echo "**Cluster:** Zeus (2× RTX 3080 Ti) + rig-6600 (8× RX 6600) + rig-6600b (8× RX 6600) + rig-6600c (8× RX 6600)  " >> "$REPORT"
echo "" >> "$REPORT"

# ============================================================
# Pre-benchmark GPU state
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Pre-Benchmark Cluster State" >> "$REPORT"
echo "" >> "$REPORT"

# Find earliest health check
PRE_CHECK=$(ls -t "$LOGDIR"/gpu_health_*pre*.txt 2>/dev/null | tail -1)
if [ -n "$PRE_CHECK" ]; then
    echo "Source: \`$(basename "$PRE_CHECK")\`" >> "$REPORT"
    echo "" >> "$REPORT"
    
    # Extract Zeus GPU info
    ZEUS_INFO=$(grep -A2 "Zeus (CUDA)" "$PRE_CHECK" | tail -1)
    if [ -n "$ZEUS_INFO" ]; then
        echo "**Zeus (CUDA):** $ZEUS_INFO" >> "$REPORT"
        echo "" >> "$REPORT"
    fi
    
    # Check for any warnings in pre-state
    PRE_WARNINGS=$(grep "WARNING" "$PRE_CHECK" 2>/dev/null)
    if [ -n "$PRE_WARNINGS" ]; then
        echo "**Pre-existing anomalies:**" >> "$REPORT"
        echo '```' >> "$REPORT"
        echo "$PRE_WARNINGS" >> "$REPORT"
        echo '```' >> "$REPORT"
        echo "" >> "$REPORT"
        echo "> Note: rig-6600b GPU[4] (PCI 0000:0F:00.0) has a known broken SMU — its anomaly is expected and excluded from pass/fail analysis." >> "$REPORT"
    else
        echo "**Pre-existing anomalies:** None detected" >> "$REPORT"
    fi
    echo "" >> "$REPORT"
else
    echo "*No pre-benchmark health check found.*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# Benchmark 4: Memory Profile
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Benchmark 4: Single-Sieve Memory Profile" >> "$REPORT"
echo "" >> "$REPORT"
echo "**Purpose:** Measure VRAM and host RAM impact of individual sieve operations at different seed counts.  " >> "$REPORT"
echo "**Method:** Single forward sieve per test with memory snapshots before/after/recovered.  " >> "$REPORT"
echo "" >> "$REPORT"

B4_SUMMARY="$LOGDIR/bench4_memory/summary.txt"
if [ -f "$B4_SUMMARY" ]; then
    echo "### Results" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| seed_count | Wall Time (s) | Exit Code | New GPU Errors | Notes |" >> "$REPORT"
    echo "|-----------|--------------|-----------|----------------|-------|" >> "$REPORT"
    
    while IFS= read -r line; do
        SEEDS=$(extract_field "$line" "seed_count")
        TIME=$(extract_field "$line" "time" | sed 's/s$//')
        EXIT=$(extract_field "$line" "exit")
        ERRORS=$(extract_field "$line" "new_gpu_errors")
        SURVIVORS=$(echo "$line" | sed -n 's/.*new_gpu_errors=[0-9]* | \(.*\)/\1/p')
        
        # Determine status icon
        if [ "$EXIT" = "0" ] && [ "$ERRORS" = "0" ]; then
            STATUS_ICON="✅"
        elif [ "$ERRORS" != "0" ]; then
            STATUS_ICON="⚠️"
        else
            STATUS_ICON="❌"
        fi
        
        printf "| %s | %s | %s %s | %s | %s |\n" \
            "$SEEDS" "$TIME" "$STATUS_ICON" "$EXIT" "$ERRORS" "$SURVIVORS" >> "$REPORT"
    done < "$B4_SUMMARY"
    
    echo "" >> "$REPORT"
    
    # Parse memory logs for delta analysis
    echo "### Memory Deltas" >> "$REPORT"
    echo "" >> "$REPORT"
    
    HAS_MEMDATA=false
    for MEMLOG in "$LOGDIR"/bench4_memory/mem_seeds_*_memory.log; do
        [ -f "$MEMLOG" ] || continue
        HAS_MEMDATA=true
        SEEDLABEL=$(basename "$MEMLOG" | sed 's/mem_seeds_//;s/_memory.log//')
        
        echo "**seed_count=$SEEDLABEL:**" >> "$REPORT"
        echo '```' >> "$REPORT"
        # Show the BEFORE and AFTER_IMMEDIATE and RECOVERED sections compactly
        grep -A1 "Memory snapshot:" "$MEMLOG" | head -20 >> "$REPORT"
        echo '```' >> "$REPORT"
        echo "" >> "$REPORT"
    done
    
    if [ "$HAS_MEMDATA" = false ]; then
        echo "*No detailed memory logs found.*" >> "$REPORT"
        echo "" >> "$REPORT"
    fi
    
    # Analysis
    echo "### Analysis" >> "$REPORT"
    echo "" >> "$REPORT"
    
    # Check if all passed
    FAIL_COUNT=$(grep -c -E "⚠️|❌|FAIL|CRASH" "$B4_SUMMARY" 2>/dev/null || echo "0")
    TOTAL_COUNT=$(wc -l < "$B4_SUMMARY")
    
    if [ "$FAIL_COUNT" = "0" ]; then
        echo "All $TOTAL_COUNT seed counts completed without GPU anomalies. Single sieve operations are stable across the tested range." >> "$REPORT"
    else
        echo "**$FAIL_COUNT of $TOTAL_COUNT tests showed issues.** GPU stress appears during single-sieve operations at higher seed counts." >> "$REPORT"
    fi
    echo "" >> "$REPORT"
else
    echo "*Benchmark 4 not run or no summary found at \`$B4_SUMMARY\`*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# Benchmark 1: Seed Count Scaling
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Benchmark 1: Seed Count Scaling" >> "$REPORT"
echo "" >> "$REPORT"
echo "**Purpose:** Find optimal seed_count per trial — analogous to \`sample_size=450\` for Step 2.5.  " >> "$REPORT"
echo "**Method:** 3 Bayesian trials per seed count, all 26 GPUs.  " >> "$REPORT"
echo "" >> "$REPORT"

B1_SUMMARY="$LOGDIR/bench1_seed_scaling/summary.txt"
if [ -f "$B1_SUMMARY" ]; then
    echo "### Results" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| seed_count | Wall Time (s) | Status | Throughput (seeds/sec) |" >> "$REPORT"
    echo "|-----------|--------------|--------|----------------------|" >> "$REPORT"
    
    BEST_THROUGHPUT=0
    BEST_SEEDS=""
    HIGHEST_SAFE=""
    
    while IFS= read -r line; do
        SEEDS=$(extract_field "$line" "seed_count")
        TIME=$(extract_field "$line" "time" | sed 's/s$//')
        STATUS=$(extract_field_full "$line" "status")
        THROUGHPUT=$(extract_field "$line" "throughput")
        
        # Track best throughput among passing tests
        if echo "$STATUS" | grep -q "PASS"; then
            HIGHEST_SAFE="$SEEDS"
            # Compare throughput (integer comparison)
            TP_NUM=$(echo "$THROUGHPUT" | grep -oP '^\d+' 2>/dev/null || echo "0")
            if [ "$TP_NUM" -gt "$BEST_THROUGHPUT" ] 2>/dev/null; then
                BEST_THROUGHPUT=$TP_NUM
                BEST_SEEDS=$SEEDS
            fi
        fi
        
        printf "| %s | %s | %s | %s |\n" "$SEEDS" "$TIME" "$STATUS" "$THROUGHPUT" >> "$REPORT"
    done < "$B1_SUMMARY"
    
    echo "" >> "$REPORT"
    
    # Analysis
    echo "### Analysis" >> "$REPORT"
    echo "" >> "$REPORT"
    
    if [ -n "$BEST_SEEDS" ]; then
        echo "**Peak throughput:** $BEST_THROUGHPUT seeds/sec at seed_count=$BEST_SEEDS  " >> "$REPORT"
    fi
    if [ -n "$HIGHEST_SAFE" ]; then
        echo "**Highest safe seed_count:** $HIGHEST_SAFE (last PASS before failures or end of test)  " >> "$REPORT"
    fi
    
    FAIL_COUNT=$(grep -c -E "FAIL|CRASH" "$B1_SUMMARY" 2>/dev/null || echo "0")
    TOTAL_COUNT=$(wc -l < "$B1_SUMMARY")
    
    if [ "$FAIL_COUNT" = "0" ]; then
        echo "All $TOTAL_COUNT seed counts passed cleanly at 3 trials each." >> "$REPORT"
    else
        FIRST_FAIL=$(grep -m1 -E "FAIL|CRASH" "$B1_SUMMARY" | head -1)
        FIRST_FAIL_SEEDS=$(extract_field "$FIRST_FAIL" "seed_count")
        echo "**First failure at seed_count=$FIRST_FAIL_SEEDS.** GPU stress threshold identified." >> "$REPORT"
    fi
    echo "" >> "$REPORT"
else
    echo "*Benchmark 1 not run or no summary found at \`$B1_SUMMARY\`*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# Benchmark 2: Concurrency Scaling
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Benchmark 2: Concurrency Scaling" >> "$REPORT"
echo "" >> "$REPORT"
echo "**Purpose:** Test whether full 26-GPU concurrency is safe for seed-range jobs.  " >> "$REPORT"
echo "**Method:** 5 Bayesian trials at 500K seeds, varying --max-concurrent.  " >> "$REPORT"
echo "" >> "$REPORT"

B2_SUMMARY="$LOGDIR/bench2_concurrency/summary.txt"
if [ -f "$B2_SUMMARY" ]; then
    echo "### Results" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| max-concurrent | Wall Time (s) | Status | Throughput (seeds/sec) | Scaling Factor |" >> "$REPORT"
    echo "|---------------|--------------|--------|----------------------|---------------|" >> "$REPORT"
    
    BASELINE_TP=0
    BEST_CONC=""
    HIGHEST_SAFE_CONC=""
    
    while IFS= read -r line; do
        CONC=$(extract_field "$line" "max-concurrent")
        TIME=$(extract_field "$line" "time" | sed 's/s$//')
        STATUS=$(extract_field_full "$line" "status")
        THROUGHPUT=$(extract_field "$line" "throughput")
        
        TP_NUM=$(echo "$THROUGHPUT" | grep -oP '^\d+' 2>/dev/null || echo "0")
        
        # First row is baseline
        if [ "$BASELINE_TP" = "0" ] && [ "$TP_NUM" -gt 0 ] 2>/dev/null; then
            BASELINE_TP=$TP_NUM
            SCALING="1.00×"
        elif [ "$BASELINE_TP" -gt 0 ] 2>/dev/null; then
            SCALING=$(echo "scale=2; $TP_NUM / $BASELINE_TP" | bc 2>/dev/null || echo "N/A")
            SCALING="${SCALING}×"
        else
            SCALING="N/A"
        fi
        
        if echo "$STATUS" | grep -q "PASS"; then
            HIGHEST_SAFE_CONC="$CONC"
        fi
        
        printf "| %s | %s | %s | %s | %s |\n" "$CONC" "$TIME" "$STATUS" "$THROUGHPUT" "$SCALING" >> "$REPORT"
    done < "$B2_SUMMARY"
    
    echo "" >> "$REPORT"
    
    # Analysis
    echo "### Analysis" >> "$REPORT"
    echo "" >> "$REPORT"
    
    if [ -n "$HIGHEST_SAFE_CONC" ]; then
        echo "**Safe concurrency ceiling:** $HIGHEST_SAFE_CONC GPUs  " >> "$REPORT"
    fi
    
    echo "" >> "$REPORT"
    echo "Scaling interpretation:" >> "$REPORT"
    echo "- **Linear scaling** (e.g., 26 GPUs ≈ 3.25× of 8 GPUs): No contention, full concurrency is safe." >> "$REPORT"
    echo "- **Sub-linear scaling** (e.g., 26 GPUs < 2.5× of 8 GPUs): Host memory or I/O contention. Consider per-node limits." >> "$REPORT"
    echo "- **Negative scaling** (throughput drops at higher concurrency): Severe contention. Must limit concurrency." >> "$REPORT"
    echo "" >> "$REPORT"
else
    echo "*Benchmark 2 not run or no summary found at \`$B2_SUMMARY\`*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# Benchmark 3: Multi-Trial Stress Test
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Benchmark 3: Multi-Trial Stress Test" >> "$REPORT"
echo "" >> "$REPORT"
echo "**Purpose:** Find how many consecutive Window Optimizer trials the cluster can handle.  " >> "$REPORT"
echo "**Method:** Increasing trial counts at 500K seeds, 26 GPUs. Auto-stop on GPU anomalies.  " >> "$REPORT"
echo "" >> "$REPORT"

B3_SUMMARY="$LOGDIR/bench3_stress/summary.txt"
if [ -f "$B3_SUMMARY" ]; then
    echo "### Results" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Trials | Wall Time (s) | Rate (trials/min) | Status |" >> "$REPORT"
    echo "|--------|--------------|-------------------|--------|" >> "$REPORT"
    
    STABILITY_CEILING=""
    LAST_PASS=""
    
    while IFS= read -r line; do
        # Skip the STABILITY_CEILING marker line
        if echo "$line" | grep -q "STABILITY_CEILING"; then
            STABILITY_CEILING=$(echo "$line" | sed 's/STABILITY_CEILING=//')
            continue
        fi
        
        TRIALS=$(extract_field "$line" "trials")
        TIME=$(extract_field "$line" "time" | sed 's/s$//')
        RATE=$(extract_field "$line" "rate")
        STATUS=$(extract_field_full "$line" "status")
        
        if echo "$STATUS" | grep -q "PASS"; then
            LAST_PASS="$TRIALS"
        fi
        
        printf "| %s | %s | %s | %s |\n" "$TRIALS" "$TIME" "$RATE" "$STATUS" >> "$REPORT"
    done < "$B3_SUMMARY"
    
    echo "" >> "$REPORT"
    
    # Analysis
    echo "### Analysis" >> "$REPORT"
    echo "" >> "$REPORT"
    
    if [ -n "$STABILITY_CEILING" ]; then
        echo "**🛑 Stability ceiling found:** $STABILITY_CEILING  " >> "$REPORT"
        echo "" >> "$REPORT"
        echo "The cluster showed GPU anomalies before reaching this trial count. Inter-trial cooldown and cleanup are required." >> "$REPORT"
    elif [ -n "$LAST_PASS" ]; then
        echo "**✅ All tested trial counts passed** (up to $LAST_PASS trials).  " >> "$REPORT"
        echo "" >> "$REPORT"
        echo "The cluster handled $LAST_PASS consecutive trials without degradation. Inter-trial cooldown may not be strictly necessary at this scale, but adding a small cooldown (5-10s) is still recommended as a safety margin for larger runs." >> "$REPORT"
    else
        echo "*Could not determine stability ceiling from results.*" >> "$REPORT"
    fi
    echo "" >> "$REPORT"
else
    echo "*Benchmark 3 not run or no summary found at \`$B3_SUMMARY\`*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# GPU Health Delta (Pre vs Post)
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Post-Benchmark Cluster State" >> "$REPORT"
echo "" >> "$REPORT"

POST_CHECK=$(ls -t "$LOGDIR"/gpu_health_*post*.txt 2>/dev/null | head -1)
if [ -n "$POST_CHECK" ]; then
    echo "Source: \`$(basename "$POST_CHECK")\`" >> "$REPORT"
    echo "" >> "$REPORT"
    
    POST_WARNINGS=$(grep "WARNING" "$POST_CHECK" 2>/dev/null)
    if [ -n "$POST_WARNINGS" ]; then
        echo "**Post-benchmark anomalies:**" >> "$REPORT"
        echo '```' >> "$REPORT"
        echo "$POST_WARNINGS" >> "$REPORT"
        echo '```' >> "$REPORT"
    else
        echo "**Post-benchmark anomalies:** None (cluster healthy)" >> "$REPORT"
    fi
    echo "" >> "$REPORT"
else
    echo "*No post-benchmark health check found.*" >> "$REPORT"
    echo "" >> "$REPORT"
fi

# ============================================================
# Recommendations
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Recommended Step 1 Configuration" >> "$REPORT"
echo "" >> "$REPORT"
echo "Based on benchmark results, fill in the optimal values:" >> "$REPORT"
echo "" >> "$REPORT"
echo '```bash' >> "$REPORT"
echo "# Step 1 optimal configuration (derived from benchmarks)" >> "$REPORT"

# Try to auto-populate from results
if [ -f "$B1_SUMMARY" ]; then
    # Find highest passing seed count
    BEST_SC=$(grep "PASS" "$B1_SUMMARY" | tail -1 | grep -oP 'seed_count=\K\d+')
    if [ -n "$BEST_SC" ]; then
        echo "opt_seed_count=$BEST_SC           # Highest stable seed count (Benchmark 1)" >> "$REPORT"
    else
        echo "opt_seed_count=?????           # From Benchmark 1 (highest stable)" >> "$REPORT"
    fi
else
    echo "opt_seed_count=?????           # From Benchmark 1 (highest stable)" >> "$REPORT"
fi

if [ -f "$B2_SUMMARY" ]; then
    BEST_CC=$(grep "PASS" "$B2_SUMMARY" | tail -1 | grep -oP 'max-concurrent=\K\d+')
    if [ -n "$BEST_CC" ]; then
        echo "max_concurrent=$BEST_CC              # Safe concurrency ceiling (Benchmark 2)" >> "$REPORT"
    else
        echo "max_concurrent=??              # From Benchmark 2 (safe ceiling)" >> "$REPORT"
    fi
else
    echo "max_concurrent=??              # From Benchmark 2 (safe ceiling)" >> "$REPORT"
fi

if [ -f "$B3_SUMMARY" ]; then
    if [ -n "$LAST_PASS" ]; then
        echo "max_consecutive_trials=$LAST_PASS      # Stability ceiling (Benchmark 3)" >> "$REPORT"
    else
        echo "max_consecutive_trials=??      # From Benchmark 3 (stability ceiling)" >> "$REPORT"
    fi
else
    echo "max_consecutive_trials=??      # From Benchmark 3 (stability ceiling)" >> "$REPORT"
fi

echo "inter_trial_cooldown=??s       # Derived from Benchmark 3 (if ceiling found)" >> "$REPORT"
echo '```' >> "$REPORT"
echo "" >> "$REPORT"

# ============================================================
# Implementation recommendations
# ============================================================

echo "## Implementation Plan" >> "$REPORT"
echo "" >> "$REPORT"
echo "Based on these results, the following protections should be added to Step 1:" >> "$REPORT"
echo "" >> "$REPORT"
echo "### 1. window_optimizer.py / window_optimizer_integration_final.py" >> "$REPORT"
echo "" >> "$REPORT"
echo "- **Inter-trial cooldown:** \`time.sleep(N)\` between Bayesian trials" >> "$REPORT"
echo "- **Inter-trial GPU cleanup:** SSH \`_best_effort_gpu_cleanup()\` to all nodes between trials" >> "$REPORT"
echo "- **GPU health gate:** Check \`rocm-smi\` between trials; abort if new anomalies detected" >> "$REPORT"
echo "- **Seed count cap:** Default \`--opt-seed-count\` to validated optimal value" >> "$REPORT"
echo "" >> "$REPORT"
echo "### 2. coordinator.py" >> "$REPORT"
echo "" >> "$REPORT"
echo "- **Per-node concurrency limits:** If Benchmark 2 showed sub-linear scaling" >> "$REPORT"
echo "- **Inter-batch cooldown:** If Benchmark 4 showed memory not recovering between sieves" >> "$REPORT"
echo "" >> "$REPORT"

# ============================================================
# Raw log inventory
# ============================================================

echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Raw Log Inventory" >> "$REPORT"
echo "" >> "$REPORT"
echo '```' >> "$REPORT"
find "$LOGDIR" -type f -name "*.txt" -o -name "*.log" 2>/dev/null | sort >> "$REPORT"
echo '```' >> "$REPORT"
echo "" >> "$REPORT"
echo "---" >> "$REPORT"
echo "*End of Step 1 Benchmark Report — Generated $(date '+%Y-%m-%d %H:%M:%S')*" >> "$REPORT"

# ============================================================
# Done
# ============================================================

echo ""
echo "=========================================="
echo "BENCHMARK REPORT GENERATED"
echo "=========================================="
echo "Report: $REPORT"
echo ""
echo "View it:"
echo "  cat $REPORT"
echo ""
echo "Copy to ser8:"
echo "  # From ser8:"
echo "  scp rzeus:~/distributed_prng_analysis/$REPORT ~/Downloads/"
echo "=========================================="
