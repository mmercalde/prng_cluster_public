#!/bin/bash
# ============================================================================
# Run Full Scoring - Step 3 Orchestration Script
# ============================================================================
# 
# Version: 2.0.0 - Uses scripts_coordinator.py (v1.3.0)
#
# CHANGES FROM v1.x:
# - Phase 3 now uses scripts_coordinator.py instead of coordinator.py
# - Run ID scoping: results in full_scoring_results/{run_id}/
# - Manifest file for validation: scripts_run_manifest.json
# - File-based success detection (no stdout parsing)
#
# Usage:
#   ./run_step3_full_scoring.sh [options]
#
# Options:
#   --survivors FILE      Bidirectional survivors JSON (required)
#   --train-history FILE  Training history JSON (required)
#   --config FILE         Optimal scorer config (optional)
#   --chunk-size N        Seeds per chunk (default: auto)
#   --forward-survivors FILE  Forward sieve survivors for metadata merge
#   --reverse-survivors FILE  Reverse sieve survivors for metadata merge
#   --dry-run             Generate jobs only, don't execute
#
# ============================================================================

set -e  # Exit on error

# Default values
SURVIVORS_FILE="bidirectional_survivors.json"
TRAIN_HISTORY="train_history.json"
HOLDOUT_HISTORY="holdout_history.json"
CONFIG_FILE="optimal_scorer_config.json"
CHUNK_SIZE=auto
FORWARD_SURVIVORS=""
REVERSE_SURVIVORS=""
DRY_RUN=false
REMOTE_NODES="192.168.3.120 192.168.3.154"
REMOTE_USER="michael"
REMOTE_BASE="/home/michael/distributed_prng_analysis"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --survivors)
            SURVIVORS_FILE="$2"
            shift 2
            ;;
        --train-history)
            TRAIN_HISTORY="$2"
            shift 2
            ;;
        --holdout-history)
            HOLDOUT_HISTORY="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --forward-survivors)
            FORWARD_SURVIVORS="$2"
            shift 2
            ;;
        --reverse-survivors)
            REVERSE_SURVIVORS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "STEP 3: FULL DISTRIBUTED SCORING (v2.0.0)"
echo "============================================================"
echo "  Survivors: $SURVIVORS_FILE"
echo "  Train History: $TRAIN_HISTORY"
echo "  Config: $CONFIG_FILE"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Dry Run: $DRY_RUN"
echo "============================================================"

# Validate input files
if [[ ! -f "$SURVIVORS_FILE" ]]; then
    echo "ERROR: Survivors file not found: $SURVIVORS_FILE"
    exit 1
fi

if [[ ! -f "$TRAIN_HISTORY" ]]; then
    echo "ERROR: Training history file not found: $TRAIN_HISTORY"
    exit 1
fi

# Create directories
mkdir -p scoring_chunks
mkdir -p full_scoring_results

# ============================================================================
# Phase 1: Generate Job Specifications
# ============================================================================
echo ""
echo "Phase 1: Generating job specifications..."
echo "------------------------------------------------------------"

# Build generate command
GEN_CMD="python3 generate_step3_scoring_jobs.py \
    --survivors $SURVIVORS_FILE \
    --train-history $TRAIN_HISTORY \
    --holdout-history $HOLDOUT_HISTORY \
    --chunk-size $CHUNK_SIZE \
    --output-file scoring_jobs.json"

if [[ -f "$CONFIG_FILE" ]]; then
    GEN_CMD="$GEN_CMD --config $CONFIG_FILE"
fi
if [[ -n "$FORWARD_SURVIVORS" ]]; then
    GEN_CMD="$GEN_CMD --forward-survivors $FORWARD_SURVIVORS"
fi
if [[ -n "$REVERSE_SURVIVORS" ]]; then
    GEN_CMD="$GEN_CMD --reverse-survivors $REVERSE_SURVIVORS"
fi

echo "Running: $GEN_CMD"
eval $GEN_CMD

if [[ ! -f "scoring_jobs.json" ]]; then
    echo "ERROR: Job generation failed - scoring_jobs.json not created"
    exit 1
fi

NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('scoring_jobs.json'))))")
echo "✓ Generated $NUM_JOBS jobs"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY RUN - Stopping before execution"
    echo "Review scoring_jobs.json and scoring_chunks/ before running again"
    exit 0
fi

# ============================================================================
# Phase 2: Distribute Data to Remote Nodes
# ============================================================================
echo ""
echo "Phase 2: Distributing data to remote nodes..."
echo "------------------------------------------------------------"

for NODE in $REMOTE_NODES; do
    echo "  → $NODE"
    
    # Create directories
    ssh ${REMOTE_USER}@${NODE} "mkdir -p ${REMOTE_BASE}/scoring_chunks" 2>/dev/null || true
    ssh ${REMOTE_USER}@${NODE} "mkdir -p ${REMOTE_BASE}/full_scoring_results" 2>/dev/null || true
    
    # Copy required files
    echo "    Copying survivors file..."
    scp -q "$SURVIVORS_FILE" ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
    
    echo "    Copying training history..."
    scp -q "$TRAIN_HISTORY" ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
    
    echo "    Copying holdout history..."
    scp -q "$HOLDOUT_HISTORY" ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
    
    echo "    Copying chunk files..."
    scp -q scoring_chunks/*.json ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/scoring_chunks/ 2>/dev/null || true
    
    echo "    Copying full_scoring_worker.py..."
    scp -q full_scoring_worker.py ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
    
    # v1.9.1: Removed forward/reverse survivor file copying (1.7GB)
    # Metadata is already embedded in chunk files
    
    echo "    ✓ Data distributed to $NODE"
done

echo "✓ Data distribution complete"

# ============================================================================
# Phase 3: Execute Distributed Jobs (UPDATED: uses scripts_coordinator.py)
# ============================================================================
echo ""
echo "Phase 3: Executing distributed jobs..."
echo "------------------------------------------------------------"

# UPDATED: Use scripts_coordinator.py instead of coordinator.py
python3 scripts_coordinator.py --jobs-file scoring_jobs.json --config distributed_config.json

echo "✓ Job execution complete"

# ============================================================================
# Phase 4: Pull Results from Remote Nodes
# ============================================================================
echo ""
echo "Phase 4: Pulling results from remote nodes..."
echo "------------------------------------------------------------"

# Find the latest run directory (scripts_coordinator creates run-scoped dirs)
LATEST_RUN=$(ls -td full_scoring_results/full_scoring_results_* 2>/dev/null | head -1)

if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: No run directory found in full_scoring_results/"
    exit 1
fi

echo "  Latest run: $LATEST_RUN"
RUN_NAME=$(basename "$LATEST_RUN")

for NODE in $REMOTE_NODES; do
    echo "  ← $NODE"
    
    # Create local run directory if not exists
    mkdir -p "$LATEST_RUN"
    
    # Pull result files from run-scoped directory
    scp -q ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/full_scoring_results/${RUN_NAME}/*.json \
        "$LATEST_RUN/" 2>/dev/null || echo "    (no results from $NODE)"
    
done

# Also include local results (from zeus GPUs)
if [[ -d "$LATEST_RUN" ]]; then
    echo "  ← localhost (already local)"
fi

echo "✓ Results pulled"

# ============================================================================
# Phase 5: Aggregate Results
# ============================================================================
echo ""
echo "Phase 5: Aggregating results..."
echo "------------------------------------------------------------"

# Aggregation script (Python inline for portability)
TRAIN_HISTORY="$TRAIN_HISTORY" python3 << 'AGGREGATE_EOF'
import json
import sys
import os
from pathlib import Path

# ============================================================================
# Global Features Integration (Team Beta Approved - Dec 26, 2025)
# ============================================================================
# Add 14 global features from GlobalStateTracker to each survivor.
# Features are prefixed with "global_" to prevent namespace collision.
# These are identical for all survivors (computed from lottery history).
# ============================================================================

# Import GlobalStateTracker
sys.path.insert(0, '.')
try:
    from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_NAMES
    GLOBAL_TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"  ⚠️  GlobalStateTracker not available: {e}")
    GLOBAL_TRACKER_AVAILABLE = False
    GLOBAL_FEATURE_NAMES = []

# Find latest run directory
run_dirs = sorted(Path("full_scoring_results").glob("full_scoring_results_*"), reverse=True)
if not run_dirs:
    print("ERROR: No run directories found")
    sys.exit(1)

results_dir = run_dirs[0]
output_file = Path("survivors_with_scores.json")

print(f"Scanning {results_dir} for result files...")

# Check manifest first
manifest_path = results_dir / "scripts_run_manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  Run ID: {manifest.get('run_id', 'unknown')}")
    print(f"  Jobs expected: {manifest.get('jobs_expected', '?')}")
    print(f"  Jobs completed: {manifest.get('jobs_completed', '?')}")
    if manifest.get('jobs_failed', 0) > 0:
        print(f"  ⚠️  Jobs failed: {manifest.get('jobs_failed')}")

# ============================================================================
# Compute Global Features (once for all survivors)
# ============================================================================
global_features_prefixed = {}
if GLOBAL_TRACKER_AVAILABLE:
    train_history_file = os.environ.get('TRAIN_HISTORY', 'train_history.json')
    try:
        with open(train_history_file) as f:
            lottery_data = json.load(f)
        
        # Handle both formats: list of ints or list of dicts
        if lottery_data and isinstance(lottery_data[0], dict):
            lottery_history = [d.get('draw', d.get('number', 0)) for d in lottery_data]
        else:
            lottery_history = lottery_data
        
        print(f"  Loading lottery history: {len(lottery_history)} draws from {train_history_file}")
        
        # Compute global features
        global_tracker = GlobalStateTracker(lottery_history, {'mod': 1000})
        global_features = global_tracker.get_global_state()
        
        # Team Beta Fix #1: Prefix with "global_" to prevent namespace collision
        global_features_prefixed = {
            f"global_{k}": v for k, v in global_features.items()
        }
        
        # Team Beta Fix #2: Variance guardrail
        unique_values = len(set(global_features.values()))
        if unique_values <= 1:
            print(f"  ⚠️  WARNING: Global features have no variance ({unique_values} unique values)")
        
        print(f"  ✅ Global features computed: {len(global_features_prefixed)} features")
        
    except Exception as e:
        print(f"  ⚠️  Could not compute global features: {e}")
        global_features_prefixed = {}

all_survivors = []
error_count = 0
files_processed = 0

for result_file in sorted(results_dir.glob("chunk_*.json")):
    try:
        with open(result_file, 'r') as f:
            chunk_results = json.load(f)
        
        if isinstance(chunk_results, list):
            for survivor in chunk_results:
                if isinstance(survivor, dict) and 'seed' in survivor:
                    all_survivors.append(survivor)
                    if 'error' in survivor:
                        error_count += 1
        
        files_processed += 1
        print(f"  ✓ {result_file.name}: {len(chunk_results)} survivors")
        
    except Exception as e:
        print(f"  ✗ {result_file.name}: {e}")
        error_count += 1

if not all_survivors:
    print("")
    print("❌ ERROR: No survivors aggregated!")
    print("Check scripts_run_manifest.json for failure details")
    sys.exit(1)

# ============================================================================
# Merge Global Features into Each Survivor
# ============================================================================
# NOTE (Team Beta Fix #2): Global features are replicated per survivor.
# ML libraries don't know they're identical. This is intentional but
# may cause over-weighting in feature importance. Document for future.
# ============================================================================
if global_features_prefixed:
    print(f"  Merging {len(global_features_prefixed)} global features into {len(all_survivors)} survivors...")
    for survivor in all_survivors:
        if 'features' in survivor:
            survivor['features'].update(global_features_prefixed)
        else:
            survivor['features'] = dict(global_features_prefixed)

# Sort by score (highest first)
all_survivors.sort(key=lambda x: x.get('score', 0), reverse=True)

# Save aggregated results
with open(output_file, 'w') as f:
    json.dump(all_survivors, f, indent=2)

print("")
print("=" * 60)
print("AGGREGATION COMPLETE")
print("=" * 60)
print(f"  Run directory: {results_dir}")
print(f"  Files processed: {files_processed}")
print(f"  Total survivors: {len(all_survivors)}")
print(f"  Errors: {error_count}")
print(f"  Output: {output_file}")

# Validate output format
sample = all_survivors[0]
print("")
print("Sample survivor structure:")
print(f"  Keys: {list(sample.keys())}")
if 'features' in sample:
    # Separate per-seed and global feature counts
    per_seed_count = len([k for k in sample['features'] if not k.startswith('global_')])
    global_count = len([k for k in sample['features'] if k.startswith('global_')])
    print(f"  Per-seed features: {per_seed_count}")
    print(f"  Global features: {global_count}")
    print(f"  Total features: {len(sample['features'])}")
else:
    print("  WARNING: No 'features' key found!")

# Show top 5 by score
print("")
print("Top 5 survivors by score:")
for i, s in enumerate(all_survivors[:5], 1):
    print(f"  {i}. Seed {s['seed']}: score={s.get('score', 0):.4f}")

print("=" * 60)
AGGREGATE_EOF

# ============================================================================
# Phase 6: Validate Output
# ============================================================================
echo ""
echo "Phase 6: Validating output format..."
echo "------------------------------------------------------------"

python3 << 'VALIDATE_EOF'
import json
import sys

output_file = "survivors_with_scores.json"

try:
    with open(output_file) as f:
        data = json.load(f)
    
    if not data:
        print("❌ VALIDATION FAILED: Empty output file")
        sys.exit(1)
    
    sample = data[0]
    
    # Check required keys
    required_keys = ['seed', 'score', 'features']
    missing_keys = [k for k in required_keys if k not in sample]
    
    if missing_keys:
        print(f"❌ VALIDATION FAILED: Missing keys: {missing_keys}")
        sys.exit(1)
    
    # Check feature count (should be 50)
    feature_count = len(sample.get('features', {}))
    if feature_count < 46:
        print(f"❌ VALIDATION FAILED: Only {feature_count} features found (expected 46+)")
        sys.exit(1)
    
    print(f"✅ VALIDATION PASSED")
    print(f"   - {len(data)} survivors")
    print(f"   - {feature_count} features per survivor")
    print(f"   - All required keys present: {required_keys}")
    
except Exception as e:
    print(f"❌ VALIDATION FAILED: {e}")
    sys.exit(1)
VALIDATE_EOF

# ============================================================================
# Done
# ============================================================================
echo ""
echo "============================================================"
echo "✅ STEP 3 COMPLETE: Full Distributed Scoring"
echo "============================================================"
echo "  Output: survivors_with_scores.json"
echo ""
echo "Next: Run Step 4 (Adaptive Meta-Optimizer)"
echo "  python3 adaptive_meta_optimizer.py --survivors survivors_with_scores.json"
echo "============================================================"
