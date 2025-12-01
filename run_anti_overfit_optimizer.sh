#!/bin/bash
# run_anti_overfit_optimizer.sh (v1.0)
# ====================================
# Runs the 26-GPU Anti-Overfit Meta-Optimization (Step 5).
#
# Based on run_scorer_meta_optimizer.sh pattern from Step 2.5.
#
# PULL ARCHITECTURE:
# - Jobs created on Zeus with pre-sampled Optuna parameters
# - Workers write results to local filesystem
# - Coordinator pulls results via SCP
# - Results reported back to Optuna on Zeus

set -e
source /home/michael/venvs/torch/bin/activate

# --- Parse Arguments ---
TRIALS=${1:-10}
K_FOLDS=${2:-5}
TEST_HOLDOUT=${3:-0.2}
SURVIVORS=${4:-bidirectional_survivors.json}
LOTTERY_DATA=${5:-synthetic_lottery.json}

STUDY_NAME="anti_overfit_$(date +%s)"

echo "================================================="
echo "26-GPU ANTI-OVERFIT META-OPTIMIZATION (Step 5) - PULL Mode"
echo "================================================="
echo "Trials: $TRIALS"
echo "K-Folds: $K_FOLDS"
echo "Test Holdout: $TEST_HOLDOUT"
echo "Survivors: $SURVIVORS"
echo "Lottery Data: $LOTTERY_DATA"
echo "Study name: $STUDY_NAME"
echo ""

# Validation
for file in "$SURVIVORS" "$LOTTERY_DATA"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found!"
        exit 1
    fi
done
echo "✅ Local data files found."

# Create Optuna study directory
mkdir -p ./optuna_studies
STUDY_DB="sqlite:///./optuna_studies/${STUDY_NAME}.db"
echo "Creating local Optuna study: $STUDY_DB"

# Generate jobs
echo ""
echo "Generating $TRIALS job specifications..."
python3 generate_anti_overfit_jobs.py \
    --trials $TRIALS \
    --survivors $SURVIVORS \
    --lottery-data $LOTTERY_DATA \
    --k-folds $K_FOLDS \
    --test-holdout $TEST_HOLDOUT \
    --study-name $STUDY_NAME \
    --study-db "$STUDY_DB" \
    --output anti_overfit_jobs.json

echo "✅ Generated anti_overfit_jobs.json with $TRIALS pre-sampled jobs"
echo ""

# Copy input data to remote nodes
echo "Copying input data to remote nodes..."
for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    ssh $node "mkdir -p ~/distributed_prng_analysis/anti_overfit_results" 2>/dev/null || true
    scp -q $SURVIVORS $LOTTERY_DATA anti_overfit_jobs.json \
        $node:~/distributed_prng_analysis/
done
echo "✅ Data copied to remote nodes"

# Push latest code to remote nodes (critical for consistency)
echo ""
echo "Pushing latest worker code to all remote nodes..."
for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    scp -q anti_overfit_trial_worker.py reinforcement_engine.py \
        $node:~/distributed_prng_analysis/ && echo "    ✅ Updated on $node" || echo "    ❌ FAILED on $node"
done
echo "✅ Latest code pushed to all 26 GPUs"

# Create local results directory
mkdir -p ./anti_overfit_results

# Launch coordinator
echo ""
echo "Launching jobs via coordinator.py..."
python3 coordinator.py \
    --jobs-file anti_overfit_jobs.json \
    --config ml_coordinator_config.json \
    --max-concurrent 26 \
    --resume-policy restart

# Collect results
echo ""
echo "=========================================="
echo "COLLECTING ANTI-OVERFIT RESULTS FROM ALL NODES"
echo "=========================================="
echo "Pulling results from remote nodes..."

python3 << PYEOF
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json
import optuna
from pathlib import Path

# Initialize coordinator for result collection
coord = MultiGPUCoordinator('ml_coordinator_config.json')

# Collect results from all nodes
# Uses same pattern as scorer results but different directory
print("Collecting results from anti_overfit_results/...")

all_results = []

# Local results
local_dir = Path("anti_overfit_results")
if local_dir.exists():
    for f in local_dir.glob("trial_*.json"):
        try:
            with open(f) as fp:
                all_results.append(json.load(fp))
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

# Remote results via SCP
for node in ["192.168.3.120", "192.168.3.154"]:
    try:
        import subprocess
        # List remote files
        result = subprocess.run(
            ["ssh", node, "ls", "~/distributed_prng_analysis/anti_overfit_results/trial_*.json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            remote_files = result.stdout.strip().split('\n')
            for rf in remote_files:
                if rf:
                    # SCP each file
                    local_name = f"anti_overfit_results/{Path(rf).name}"
                    subprocess.run(
                        ["scp", "-q", f"{node}:{rf}", local_name],
                        timeout=30
                    )
                    try:
                        with open(local_name) as fp:
                            all_results.append(json.load(fp))
                    except:
                        pass
            # Clean up remote files
            subprocess.run(
                ["ssh", node, "rm", "-f", "~/distributed_prng_analysis/anti_overfit_results/trial_*.json"],
                timeout=30
            )
            print(f"  [{node}] Collected {len(remote_files)} results")
    except Exception as e:
        print(f"  [{node}] Warning: {e}")

print(f"Collected {len(all_results)} total trial results")

# Report to Optuna
study = optuna.load_study(study_name='$STUDY_NAME', storage='$STUDY_DB')

reported = 0
for result in all_results:
    try:
        if result.get('trial_id') is not None and result.get('accuracy') is not None:
            study.tell(result['trial_id'], result['accuracy'])
            reported += 1
    except Exception as e:
        print(f"Warning: Could not report trial {result.get('trial_id')}: {e}")

print(f"Reported {reported} / {len(all_results)} trials to Optuna")
print("")

# Find best trial
print("Finding best trial...")
best = study.best_trial
print("Best trial:")
print(json.dumps({
    'trial_number': best.number,
    'score': best.value,
    'params': best.params
}, indent=2))

# Save best config
best_config = {
    'trial_number': best.number,
    'score': best.value,
    **best.params
}
with open('optimal_anti_overfit_config.json', 'w') as f:
    json.dump(best_config, f, indent=2)
print("")
print("✅ Best parameters saved to optimal_anti_overfit_config.json")
PYEOF

echo ""
echo "=========================================="
echo "ANTI-OVERFIT META-OPTIMIZATION COMPLETE"
echo "=========================================="
echo "Best config: optimal_anti_overfit_config.json"
echo "Study DB: $STUDY_DB"
