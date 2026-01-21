#!/bin/bash

# =============================================================================
# NPZ AUTO-CONVERSION BLOCK (Inserted by install_npz_auto_conversion.sh)
# Date: 20260119_195923
# Team Beta Approved: January 19, 2026
# =============================================================================

set -euo pipefail  # Fail-fast, no undefined vars, pipe failures

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SURVIVORS="bidirectional_survivors_binary.npz"
JSON_SOURCE="bidirectional_survivors.json"
TMP_NPZ="${SURVIVORS%.npz}.tmp.$$.npz"
CONFIG_FILE="distributed_config.json"
REMOTE_DIR="distributed_prng_analysis"

trap 'rm -f "$TMP_NPZ"' EXIT

# -----------------------------------------------------------------------------
# Extract remote nodes from distributed_config.json (no hardcoding)
# -----------------------------------------------------------------------------
get_remote_nodes() {
    python3 << 'PYEOF'
import json
import sys

CONFIG_FILE = "distributed_config.json"

try:
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    nodes = config.get("nodes", [])
    for node in nodes:
        hostname = node.get("hostname", "")
        if hostname and hostname != "localhost" and not hostname.startswith("127."):
            print(hostname)
except FileNotFoundError:
    print(f"ERROR: {CONFIG_FILE} not found", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in {CONFIG_FILE}: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# -----------------------------------------------------------------------------
# NPZ Conversion Function (atomic write)
# -----------------------------------------------------------------------------
convert_to_npz() {
    echo "============================================"
    echo "NPZ Conversion Required"
    echo "============================================"
    
    if [ ! -f "$JSON_SOURCE" ]; then
        echo "ERROR: $JSON_SOURCE not found!"
        echo "Run Step 1 (window_optimizer.py) first."
        exit 1
    fi
    
    echo "Converting $JSON_SOURCE → $SURVIVORS (atomic)..."
    rm -f "$TMP_NPZ"
    
    if ! python3 convert_survivors_to_binary.py "$JSON_SOURCE" --output "$TMP_NPZ"; then
        echo "ERROR: NPZ conversion failed!"
        exit 1
    fi
    
    if [ ! -s "$TMP_NPZ" ]; then
        echo "ERROR: Conversion produced empty or missing file!"
        exit 1
    fi
    
    mv "$TMP_NPZ" "$SURVIVORS"
    echo "✓ Conversion complete: $SURVIVORS"
}

# -----------------------------------------------------------------------------
# Distribute NPZ to Remote Nodes
# -----------------------------------------------------------------------------
distribute_npz() {
    echo ""
    echo "Distributing NPZ to remote nodes..."
    
    local nodes
    nodes=$(get_remote_nodes)
    
    if [ -z "$nodes" ]; then
        echo "WARNING: No remote nodes found in $CONFIG_FILE"
        echo "Skipping distribution (localhost-only mode)."
        return 0
    fi
    
    local failed=0
    for node in $nodes; do
        echo -n "  → $node: "
        if scp -q "$SURVIVORS" "${node}:~/${REMOTE_DIR}/" 2>/dev/null; then
            echo "✓"
        else
            echo "✗ FAILED"
            failed=1
        fi
    done
    
    if [ $failed -ne 0 ]; then
        echo "ERROR: Distribution failed to one or more nodes!"
        echo "Cluster is in inconsistent state. Aborting."
        exit 1
    fi
    
    echo "✓ Distribution complete to all nodes."
}

# -----------------------------------------------------------------------------
# Main: Auto-Convert if Needed
# -----------------------------------------------------------------------------
if [ ! -f "$SURVIVORS" ]; then
    echo "NPZ file missing - conversion required."
    convert_to_npz
    distribute_npz
elif [ "$JSON_SOURCE" -nt "$SURVIVORS" ]; then
    echo "JSON newer than NPZ - reconversion required."
    convert_to_npz
    distribute_npz
else
    echo "NPZ file up-to-date, skipping conversion."
fi

echo "============================================"
echo ""
# =============================================================================
# END NPZ AUTO-CONVERSION BLOCK
# =============================================================================

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

# ============================================================
# RAMDISK PRELOAD (Team Beta Approved 2026-01-20)
# Eliminates disk I/O contention during worker startup
# ============================================================
echo ""
echo "[INFO] Preloading data to RAM disk on remote nodes..."

# Extract remote nodes from distributed_config.json (single source of truth)
REMOTE_NODES=$(python3 -c "
import json
with open('distributed_config.json') as f:
    cfg = json.load(f)
for node in cfg['nodes']:
    if node['hostname'] != 'localhost':
        print(node['hostname'])
")

for REMOTE in $REMOTE_NODES; do
    echo "  → $REMOTE"
    
    # Sanity check: verify /dev/shm is available
    ssh "$REMOTE" "df -h /dev/shm | grep -q shm" || {
        echo "    ⚠️  WARNING: /dev/shm not available on $REMOTE, skipping ramdisk"
        continue
    }
    
    # Copy-once guard: only copy if .ready sentinel missing
    ssh "$REMOTE" "
        mkdir -p /dev/shm/prng &&
        if [ ! -f /dev/shm/prng/.ready ]; then
            cp ~/distributed_prng_analysis/bidirectional_survivors_binary.npz /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/train_history.json /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/holdout_history.json /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/scorer_jobs.json /dev/shm/prng/ 2>/dev/null || true &&
            touch /dev/shm/prng/.ready &&
            echo '    ✓ Ramdisk preload complete'
        else
            echo '    ✓ Ramdisk already loaded (skipped)'
        fi
    "
done

echo "[INFO] Ramdisk preload phase complete"
echo ""
# ============================================================


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
