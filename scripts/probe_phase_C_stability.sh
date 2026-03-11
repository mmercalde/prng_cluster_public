#!/usr/bin/env bash
# =============================================================================
# PHASE C — THROUGHPUT STABILITY TEST
# S128 GPU Throughput Investigation
# Run on: Zeus (launches jobs across all 26 GPUs via WATCHER pipeline)
#
# BEFORE RUNNING:
#   Phase B must be complete. Replace RTX_CAP and AMD_CAP with Phase B × 0.85
#   (already calculated for you in the apply_caps.py output).
#   This script runs 50 consecutive WATCHER jobs at the confirmed caps.
#
# Usage: bash probe_phase_C_stability.sh 2>&1 | tee /tmp/probe_C_stability.log
# =============================================================================

set -eo pipefail

PRNG_DIR="$HOME/distributed_prng_analysis"
VENV="$HOME/venvs/torch/bin/activate"
source "$VENV"

cd "$PRNG_DIR"

# ============================================================
# ⚠️  FILL THESE IN FROM PHASE B × 0.85 BEFORE RUNNING ⚠️
# ============================================================
RTX_CAP=500000   # Phase B RTX ceiling × 0.85
AMD_CAP=200000   # Phase B AMD ceiling × 0.85
# ============================================================

TOTAL_JOBS=50
PASS=0
FAIL=0
RESULTS_FILE="/tmp/probe_C_stability.csv"
echo "job_num,duration_sec,total_seeds,agg_sps,status" > "$RESULTS_FILE"

echo "================================================"
echo " Phase C — Stability Test"
echo " 50 consecutive jobs | RTX_CAP=$RTX_CAP AMD_CAP=$AMD_CAP"
echo " Date: $(date)"
echo "================================================"
echo ""

# Patch coordinator caps for this test (temp override via CLI args)
for JOB_NUM in $(seq 1 $TOTAL_JOBS); do
    echo -n "Job $JOB_NUM/$TOTAL_JOBS ... "
    START_TS=$(date +%s%3N)

    # Launch one WATCHER pipeline run at Step 1 only (sieve, not full pipeline)
    # Uses --start-step 1 --end-step 1 with seed cap overrides
    PYTHONPATH=. python3 agents/watcher_agent.py \
        --run-pipeline \
        --start-step 1 --end-step 1 \
        --params "{
            \"seed_count\": $RTX_CAP,
            \"seed_cap_nvidia\": $RTX_CAP,
            \"seed_cap_amd\": $AMD_CAP,
            \"window_trials\": 1,
            \"resume_study\": false,
            \"study_name\": \"stability_test_run\"
        }" > /tmp/probe_C_job_${JOB_NUM}.log 2>&1
    EXIT_CODE=$?

    END_TS=$(date +%s%3N)
    DURATION_MS=$(( END_TS - START_TS ))
    DURATION_SEC=$(echo "scale=2; $DURATION_MS / 1000" | bc)

    if [ $EXIT_CODE -eq 0 ]; then
        (( PASS++ )) || true
        echo "✅ ${DURATION_SEC}s"
        echo "$JOB_NUM,$DURATION_SEC,$RTX_CAP,$(echo "scale=0; $RTX_CAP * 1000 / $DURATION_MS" | bc),OK" >> "$RESULTS_FILE"
    else
        (( FAIL++ )) || true
        echo "❌ FAILED (exit=$EXIT_CODE) — check /tmp/probe_C_job_${JOB_NUM}.log"
        echo "$JOB_NUM,$DURATION_SEC,$RTX_CAP,0,FAIL" >> "$RESULTS_FILE"
        if [ $FAIL -ge 3 ]; then
            echo ""
            echo "⛔ 3 consecutive failures — aborting stability test"
            echo "Caps may be too aggressive. Try 70% of Phase B ceiling."
            break
        fi
    fi
done

echo ""
echo "================================================"
echo " Phase C COMPLETE"
echo " Passed: $PASS / $TOTAL_JOBS | Failed: $FAIL / $TOTAL_JOBS"
echo " Results: $RESULTS_FILE"
echo "================================================"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "✅ STABILITY TEST PASSED — caps are safe for production"
    echo "   RTX cap: $RTX_CAP | AMD cap: $AMD_CAP"
    echo ""
    echo "Next step: run apply_caps.py to update coordinator.py + gpu_optimizer.py"
else
    echo ""
    echo "⚠️  STABILITY TEST PARTIAL — review failed jobs before committing caps"
fi

cat "$RESULTS_FILE"
