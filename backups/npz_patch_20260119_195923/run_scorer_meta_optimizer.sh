#!/bin/bash
# run_scorer_meta_optimizer.sh (v2.2 - AUTO CODE PUSH + Legacy Option)
# ============================
# Runs the 26-GPU Scorer Meta-Optimization (Step 2.5).
#
# NEW in v2.2: Automatically pushes latest survivor_scorer.py + reinforcement_engine.py
#             to all remote nodes → eliminates stale file errors forever.

set -e
source /home/michael/venvs/torch/bin/activate

# --- Parse Arguments ---
TRIALS=""
USE_LEGACY_SCORING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --legacy-scoring)
            USE_LEGACY_SCORING=true
            shift
            ;;
        *)
            if [ -z "$TRIALS" ]; then
                TRIALS=$1
            fi
            shift
            ;;
    esac
done

TRIALS=${TRIALS:-100}
STUDY_NAME="scorer_meta_opt_$(date +%s)"

echo "================================================="
echo "26-GPU SCORER META-OPTIMIZATION (Step 2.5) - PULL Mode"
echo "================================================="
echo "Trials: $TRIALS"
if [ "$USE_LEGACY_SCORING" = true ]; then
    echo "Scoring Method: LEGACY (original batch_score)"
else
    echo "Scoring Method: VECTORIZED (GPU-accelerated)"
fi
echo "Study name: $STUDY_NAME"
echo ""

# Input files
SURVIVORS="bidirectional_survivors_binary.npz"
TRAIN_HISTORY="train_history.json"
HOLDOUT_HISTORY="holdout_history.json"

# Validation
for file in "$SURVIVORS" "$TRAIN_HISTORY" "$HOLDOUT_HISTORY"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found!"
        exit 1
    fi
done
echo "Local data files found."

# Create Optuna study
mkdir -p ./optuna_studies
STUDY_DB="sqlite:///./optuna_studies/${STUDY_NAME}.db"
echo "Creating local Optuna study: $STUDY_DB"

# Generate jobs with optional legacy flag
echo "Generating $TRIALS job specifications and pre-sampling parameters..."
if [ "$USE_LEGACY_SCORING" = true ]; then
    python3 generate_scorer_jobs.py \
        --trials $TRIALS \
        --survivors $SURVIVORS \
        --train-history $TRAIN_HISTORY \
        --holdout-history $HOLDOUT_HISTORY \
        --study-name $STUDY_NAME \
        --study-db "$STUDY_DB" \
        --sample-size 450    # TUNED 2026-01-17: 5000 seeds = ~60-90s trials
                              # (vs 25000 default = 400-700s trials)
                              # Benefits: Better Bayesian exploration, less GPU stress \
        --legacy-scoring
else
    python3 generate_scorer_jobs.py \
        --trials $TRIALS \
        --survivors $SURVIVORS \
        --train-history $TRAIN_HISTORY \
        --holdout-history $HOLDOUT_HISTORY \
        --study-name $STUDY_NAME \
        --study-db "$STUDY_DB" \
        --sample-size 450    # TUNED 2026-01-17: 5000 seeds = ~60-90s trials
                              # (vs 25000 default = 400-700s trials)
                              # Benefits: Better Bayesian exploration, less GPU stress
fi

echo "Generated scorer_jobs.json with $TRIALS pre-sampled jobs"
echo ""

# Copy input data + FORCE latest code to remote nodes
echo "Copying input data to remote nodes..."
for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    ssh $node "mkdir -p ~/distributed_prng_analysis/scorer_trial_results" 2>/dev/null || true
    scp $SURVIVORS $TRAIN_HISTORY $HOLDOUT_HISTORY scorer_jobs.json \
        $node:~/distributed_prng_analysis/
done
echo "Data copied to remote nodes"

# CRITICAL: Push latest code so all 26 GPUs use the correct version
echo "Pushing latest survivor_scorer.py and reinforcement_engine.py to all remote nodes..."
for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    scp survivor_scorer.py reinforcement_engine.py scorer_trial_worker.py \
        $node:~/distributed_prng_analysis/ && echo "    Updated on $node" || echo "    FAILED on $node"
done
echo "Latest code pushed to all 26 GPUs"

# Launch scripts_coordinator (Team Beta approved for script jobs)
echo "Launching jobs via scripts_coordinator.py..."
python3 scripts_coordinator.py \
    --jobs-file scorer_jobs.json \
    --output-dir scorer_trial_results \
    --preserve-paths

# Rest of script (collection + Optuna reporting)
echo ""
echo "=========================================="
echo "COLLECTING SCORER RESULTS FROM ALL NODES"
echo "=========================================="
echo "Pulling results from remote nodes..."

python3 -c "
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json, optuna

coord = MultiGPUCoordinator('ml_coordinator_config.json')
all_results = coord.collect_scorer_results(1)
print(f'Collected {len(all_results)} trial results from all nodes')

study = optuna.load_study(study_name='$STUDY_NAME', storage='$STUDY_DB')

reported = 0
for result in all_results:
    try:
        if result.get('trial_id') is not None and result.get('accuracy') is not None:
            study.tell(result['trial_id'], result['accuracy'])
            reported += 1
    except Exception as e:
        print(f'Warning: Could not report trial {result.get(\"trial_id\")}: {e}')

print(f'Reported {reported} / {len(all_results)} trials to Optuna')
print('')
print('Finding best trial...')
best = study.best_trial
print('Best trial:')
print(json.dumps({'trial_number': best.number, 'accuracy': best.value, 'params': best.params}, indent=2))

with open('optimal_scorer_config.json', 'w') as f:
    json.dump(best.params, f, indent=2)
print('')
print('SUCCESS: Best parameters saved to optimal_scorer_config.json')
"

echo ""
echo "=========================================="
echo "SCORER META-OPTIMIZATION COMPLETE"
echo "=========================================="
echo "Best config: optimal_scorer_config.json"
echo "Study DB: $STUDY_DB"
