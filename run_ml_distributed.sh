#!/bin/bash
# run_ml_distributed.sh - 26-GPU ML Training with Optuna (FIXED)

set -e

TRIALS=${1:-30}
SURVIVORS="bidirectional_survivors.json"
SCORES="scores.json"
SHARED="/shared/ml"
STUDY_NAME="antioverfit_distributed_$(date +%s)"

echo "=========================================="
echo "26-GPU DISTRIBUTED ML TRAINING (OPTUNA)"
echo "=========================================="
echo "Trials: $TRIALS"
echo "Survivors: $SURVIVORS"
echo "Scores: $SCORES"
echo "Shared path: $SHARED"
echo "Study name: $STUDY_NAME"
echo ""

# Check shared storage exists
if [ ! -d "$SHARED" ]; then
    echo "ERROR: Shared storage $SHARED not found"
    echo "Creating locally for testing..."
    mkdir -p $SHARED
fi

# Create shared directories
mkdir -p $SHARED/data $SHARED/results $SHARED/models $SHARED/optuna
chmod -R 777 $SHARED 2>/dev/null || true

# Copy latest data to shared storage
echo "Copying data to shared storage..."
cp $SURVIVORS $SCORES $SHARED/data/
cp reinforcement_engine_config.json $SHARED/data/ 2>/dev/null || echo "Warning: No config file found"

# Create Optuna study database
STUDY_DB="sqlite:///$SHARED/optuna/${STUDY_NAME}.db"
echo "Creating Optuna study: $STUDY_DB"

# Initialize Optuna study (creates DB)
python3 -c "
import optuna
study = optuna.create_study(
    study_name='$STUDY_NAME',
    storage='$STUDY_DB',
    direction='minimize',
    load_if_exists=False
)
print(f'✅ Created Optuna study: $STUDY_NAME')
"

# Generate job specs (one per trial)
echo "Generating $TRIALS job specifications..."
python3 generate_ml_jobs.py \
    --trials $TRIALS \
    --survivors $SHARED/data/$SURVIVORS \
    --scores $SHARED/data/$SCORES \
    --study-name $STUDY_NAME \
    --study-db "$STUDY_DB"

echo "Generated ml_jobs.json with $TRIALS jobs"
echo ""

# Check if coordinator exists and has required features
if [ ! -f "coordinator.py" ]; then
    echo "ERROR: coordinator.py not found"
    echo "Running jobs locally as fallback..."
    
    # Fallback: run jobs sequentially locally
    for i in $(seq 0 $((TRIALS-1))); do
        echo "Running trial $i locally..."
        python3 anti_overfit_trial_worker.py \
            $SHARED/data/$SURVIVORS \
            $SHARED/data/$SCORES \
            $STUDY_NAME \
            "$STUDY_DB" \
            $i > $SHARED/results/trial_${i}.json
    done
else
    # Launch via existing coordinator
    echo "Launching jobs via coordinator.py..."
    python3 coordinator.py \
        --jobs-file ml_jobs.json \
        --config ml_coordinator_config.json \
        --max-concurrent 26 || {
            echo "ERROR: coordinator.py failed"
            echo "Check if coordinator supports --jobs-file, --config, --max-concurrent"
            exit 1
        }
fi

echo ""
echo "=========================================="
echo "COLLECTING RESULTS"
echo "=========================================="

# Wait for all results
echo "Waiting for all trial results..."
EXPECTED=$TRIALS
TIMEOUT=7200  # 2 hours max
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    FOUND=$(ls $SHARED/results/trial_*.json 2>/dev/null | wc -l)
    echo "Progress: $FOUND/$EXPECTED trials completed"
    
    if [ $FOUND -eq $EXPECTED ]; then
        break
    fi
    
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

if [ $FOUND -lt $EXPECTED ]; then
    echo "WARNING: Only $FOUND/$EXPECTED trials completed after ${ELAPSED}s"
fi

# Get best trial from Optuna study
echo ""
echo "Finding best trial from Optuna study..."
BEST_TRIAL=$(python3 -c "
import optuna
import json

study = optuna.load_study(
    study_name='$STUDY_NAME',
    storage='$STUDY_DB'
)

if len(study.trials) == 0:
    print('ERROR: No completed trials')
    exit(1)

best = study.best_trial
print(json.dumps({
    'trial_number': best.number,
    'val_loss': best.value,
    'params': best.params
}, indent=2))
")

echo "Best trial:"
echo "$BEST_TRIAL"

# Find best model file
BEST_TRIAL_NUM=$(echo "$BEST_TRIAL" | python3 -c "import sys, json; print(json.load(sys.stdin)['trial_number'])")
BEST_MODEL="$SHARED/models/trial_${BEST_TRIAL_NUM}_best.pth"

if [ -f "$BEST_MODEL" ]; then
    cp "$BEST_MODEL" ./universal_emulator.pth
    echo ""
    echo "✅ SUCCESS: Best model saved as universal_emulator.pth"
    echo "   Val loss: $(echo "$BEST_TRIAL" | python3 -c "import sys, json; print(json.load(sys.stdin)['val_loss'])")"
    echo "   Trial: $BEST_TRIAL_NUM"
else
    echo "ERROR: Best model not found at $BEST_MODEL"
    echo "Searching for any valid model..."
    
    # Fallback: find any model with lowest val_loss from JSON results
    FALLBACK=$(python3 -c "
import json
import glob
import os

results = []
for f in glob.glob('$SHARED/results/trial_*.json'):
    try:
        with open(f) as fp:
            data = json.load(fp)
            if 'val_loss' in data and 'model_path' in data:
                results.append(data)
    except:
        pass

if results:
    best = min(results, key=lambda x: x['val_loss'])
    print(best['model_path'])
else:
    print('')
" 2>/dev/null)
    
    if [ -n "$FALLBACK" ] && [ -f "$FALLBACK" ]; then
        cp "$FALLBACK" ./universal_emulator.pth
        echo "✅ Copied fallback model: universal_emulator.pth"
    else
        echo "❌ FAILED: No valid models found"
        exit 1
    fi
fi

# Create summary report
echo ""
echo "Creating summary report..."
python3 -c "
import optuna
import json

study = optuna.load_study(
    study_name='$STUDY_NAME',
    storage='$STUDY_DB'
)

summary = {
    'study_name': '$STUDY_NAME',
    'total_trials': len(study.trials),
    'best_trial': study.best_trial.number,
    'best_val_loss': study.best_value,
    'best_params': study.best_params,
    'all_trials': [
        {
            'number': t.number,
            'val_loss': t.value,
            'params': t.params,
            'state': str(t.state)
        }
        for t in study.trials
    ]
}

with open('ml_training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('✅ Summary saved: ml_training_summary.json')
"

echo ""
echo "=========================================="
echo "26-GPU ML TRAINING COMPLETE"
echo "=========================================="
echo "Best model: universal_emulator.pth"
echo "Summary: ml_training_summary.json"
echo "Study DB: $STUDY_DB"
echo ""
