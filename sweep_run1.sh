#!/bin/bash
# =============================================================================
# sweep_run1.sh — Production Sweep Run 1 of 4
# Seeds:  0 → 1,073,741,824
# Trials: 50 (with pruning — effective ~13-18 hrs depending on prune rate)
# =============================================================================
# RESUME AFTER CRASH:
#   bash sweep_run1.sh --resume
# =============================================================================

set -e
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate

RUN_ID="run1"
LOG="logs/sweep_${RUN_ID}_production.log"
STUDY_FILE="logs/sweep_${RUN_ID}_study_name.txt"
MANIFEST="agent_manifests/window_optimizer.json"
PID_FILE="logs/sweep_${RUN_ID}.pid"

mkdir -p logs

# ── RESUME MODE ───────────────────────────────────────────────────────────────
if [[ "$1" == "--resume" ]]; then
    if [[ ! -f "$STUDY_FILE" ]]; then
        echo "❌ No study file found at $STUDY_FILE — cannot resume"
        echo "   Check $LOG for 'Optuna study:' line manually"
        exit 1
    fi
    STUDY_NAME=$(cat "$STUDY_FILE")
    if [[ -z "$STUDY_NAME" ]]; then
        echo "❌ Study file is empty — study name was never captured"
        echo "   Check $LOG for 'Optuna study:' line manually"
        exit 1
    fi
    echo "============================================================"
    echo "RESUMING sweep Run 1"
    echo "Study: $STUDY_NAME"
    echo "============================================================"

    # Patch manifest: set study_name + resume_study=true
    python3 -c "
import json
m = json.load(open('$MANIFEST'))
m['default_params']['study_name'] = '$STUDY_NAME'
m['default_params']['resume_study'] = True
json.dump(m, open('$MANIFEST', 'w'), indent=2)
print('  Manifest patched: study_name=$STUDY_NAME, resume_study=true')
"
    # Clear halt
    # Start persistent workers on all rigs
    echo "Starting persistent workers on rigs..."
    for RIG in rrig6600 rrig6600b rrig6600c; do
        ssh $RIG "cd ~/distributed_prng_analysis && source ~/rocm_env/bin/activate && pkill -f persistent_gpu_worker 2>/dev/null; sleep 1 && nohup python3 persistent_gpu_worker.py > logs/worker.log 2>&1 &" && echo "  ✅ $RIG workers started" || echo "  ⚠️  $RIG worker start failed"
    done
    echo "Waiting 30s for workers to initialize..."
    sleep 30

    PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt 2>/dev/null || true

    # Re-launch
    nohup bash -c "PYTHONPATH=. python3 agents/watcher_agent.py \
        --run-pipeline --start-step 1 --end-step 1 \
        >> $LOG 2>&1" &
    echo $! > "$PID_FILE"
    echo "✅ Resumed — PID $(cat $PID_FILE)"
    echo "   tail -f $LOG"
    exit 0
fi

# ── FRESH START ───────────────────────────────────────────────────────────────
# Guard: don't launch if already running
if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "⚠️  Run 1 already in progress (PID $EXISTING_PID)"
        echo "   tail -f $LOG"
        exit 1
    fi
fi

# Guard: don't re-run if already completed
if python3 -c "
import sqlite3, sys
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute(\"SELECT seed_range_start, seed_range_end FROM exhaustive_progress WHERE prng_type='java_lcg' AND seed_range_end >= 1073741824\").fetchall()
conn.close()
sys.exit(0 if rows else 1)
" 2>/dev/null; then
    echo "⚠️  Run 1 seed range already covered in exhaustive_progress"
    echo "   Use sweep_run2.sh for the next range"
    exit 1
fi

# Ensure manifest is clean for fresh run
python3 -c "
import json
m = json.load(open('$MANIFEST'))
m['default_params']['study_name'] = ''
m['default_params']['resume_study'] = False
json.dump(m, open('$MANIFEST', 'w'), indent=2)
print('  Manifest reset: study_name cleared, resume_study=false')
"

# Start persistent workers on all rigs
echo "Starting persistent workers on rigs..."
for RIG in rrig6600 rrig6600b rrig6600c; do
    ssh $RIG "cd ~/distributed_prng_analysis && \
        source ~/rocm_env/bin/activate && \
        pkill -f persistent_gpu_worker 2>/dev/null; \
        sleep 1 && \
        nohup python3 persistent_gpu_worker.py > logs/worker.log 2>&1 &" && \
        echo "  ✅ $RIG workers started" || echo "  ⚠️  $RIG worker start failed"
done
echo "Waiting 30s for workers to initialize..."
sleep 30

echo "============================================================"
echo "PRODUCTION SWEEP — Run 1 of 4"
echo "Seeds:   0 → 1,073,741,824"
echo "Trials:  50 (pruning enabled)"
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

# Wait for study name to appear in log (up to 60 seconds)
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
        echo "   bash sweep_run1.sh --resume"
        echo ""
        echo "Monitor:"
        echo "   tail -f $LOG"
        exit 0
    fi
done

# If study name not captured after 60s something is wrong
echo "⚠️  Study name not found in log after 60s — check $LOG"
echo "   If run is still alive, manually run:"
echo "   grep 'Optuna study:' $LOG | tail -1"
echo "   echo '<study_name>' > $STUDY_FILE"
