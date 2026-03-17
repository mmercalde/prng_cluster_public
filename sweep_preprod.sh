#!/bin/bash
# =============================================================================
# sweep_preprod.sh — Pre-Production Validation Test
# Seeds:  0 → 50,000,000
# Trials: 5 (test_both_modes — validates all 4 sieve passes)
# Purpose: Verify PWC + hybrid fixes before full production sweep
# =============================================================================
# RESUME AFTER CRASH:
#   bash sweep_preprod.sh --resume
# =============================================================================

set -e
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

RUN_ID="preprod"
LOG="logs/sweep_${RUN_ID}.log"
STUDY_FILE="logs/sweep_${RUN_ID}_study_name.txt"
MANIFEST="agent_manifests/window_optimizer.json"
PID_FILE="logs/sweep_${RUN_ID}.pid"

mkdir -p logs

# ── RESUME MODE ───────────────────────────────────────────────────────────────
if [[ "$1" == "--resume" ]]; then
    if [[ ! -f "$STUDY_FILE" ]]; then
        echo "❌ No study file found at $STUDY_FILE — cannot resume"
        exit 1
    fi
    STUDY_NAME=$(cat "$STUDY_FILE")
    echo "============================================================"
    echo "RESUMING Pre-Production Test"
    echo "Study: $STUDY_NAME"
    echo "============================================================"

    python3 -c "
import json
m = json.load(open('$MANIFEST'))
m['default_params']['study_name'] = '$STUDY_NAME'
m['default_params']['resume_study'] = True
json.dump(m, open('$MANIFEST', 'w'), indent=2)
print('  Manifest patched: resume_study=true')
"
    PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt 2>/dev/null || true

    nohup bash -c "PYTHONPATH=. python3 agents/watcher_agent.py \
        --run-pipeline --start-step 1 --end-step 1 \
        >> $LOG 2>&1" &
    echo $! > "$PID_FILE"
    echo "✅ Resumed — PID $(cat $PID_FILE)"
    exit 0
fi

# ── FRESH START ───────────────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "⚠️  Pre-prod test already running (PID $EXISTING_PID)"
        echo "   tail -f $LOG"
        exit 1
    fi
fi

# Patch manifest for pre-prod parameters
python3 -c "
import json
m = json.load(open('$MANIFEST'))
# Save original values
orig_seeds = m['default_params'].get('max_seeds')
orig_trials = m['default_params'].get('window_trials')
print(f'  Original: max_seeds={orig_seeds}, trials={orig_trials}')
# Apply pre-prod values
m['default_params']['max_seeds'] = 50000000
m['default_params']['window_trials'] = 5
m['default_params']['study_name'] = ''
m['default_params']['resume_study'] = False
json.dump(m, open('$MANIFEST', 'w'), indent=2)
print('  Manifest patched: max_seeds=50M, trials=5')
"

echo "============================================================"
echo "PRE-PRODUCTION VALIDATION TEST"
echo "Seeds:   0 → 50,000,000"
echo "Trials:  5 (test_both_modes — all 4 sieve passes)"
echo "Log:     $LOG"
echo "============================================================"
echo ""

# Launch pipeline
nohup bash -c "PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 \
    > $LOG 2>&1" &

LAUNCH_PID=$!
echo $LAUNCH_PID > "$PID_FILE"
echo "✅ Launched — PID $LAUNCH_PID"
echo ""

# Wait for study name
echo "Waiting for Optuna study name..."
for i in $(seq 1 60); do
    sleep 1
    STUDY=$(grep -a "Optuna study" "$LOG" 2>/dev/null | tail -1 | sed 's|.*optuna_studies/\(.*\)\.db.*|\1|')
    if [[ -n "$STUDY" ]]; then
        echo "$STUDY" > "$STUDY_FILE"
        echo "✅ Study name captured: $STUDY"
        echo "   Saved to: $STUDY_FILE"
        echo ""
        echo "IF THIS RUN CRASHES — resume with:"
        echo "   bash sweep_preprod.sh --resume"
        echo ""
        echo "Monitor:"
        echo "   tail -f $LOG"
        exit 0
    fi
done

echo "⚠️  Study name not found in log after 60s — check $LOG"
echo "   grep 'Optuna study:' $LOG | tail -1"
echo "   echo '<study_name>' > $STUDY_FILE"
