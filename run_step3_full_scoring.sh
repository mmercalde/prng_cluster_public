#!/bin/bash
# ============================================================================
# Run Full Scoring - Step 3 Orchestration Script
# ============================================================================
# 
# CRITICAL FIX: Previous version aggregated prediction floats from 
# scorer_trial_worker.py. This version properly aggregates full 46-feature
# survivor objects from full_scoring_worker.py.
#
# Usage:
#   ./run_full_scoring.sh [options]
#
# Options:
#   --survivors FILE      Bidirectional survivors JSON (required)
#   --train-history FILE  Training history JSON (required)
#   --config FILE         Optimal scorer config (optional)
#   --chunk-size N        Seeds per chunk (default: 5000)
#   --dry-run             Generate jobs only, don't execute
#
# ============================================================================

set -e  # Exit on error

# Default values
SURVIVORS_FILE="bidirectional_survivors.json"
TRAIN_HISTORY="train_history.json"
CONFIG_FILE="optimal_scorer_config.json"
CHUNK_SIZE=auto
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
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
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
echo "STEP 3: FULL DISTRIBUTED SCORING"
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
    --chunk-size $CHUNK_SIZE \
    --output-file scoring_jobs.json"

if [[ -f "$CONFIG_FILE" ]]; then
    GEN_CMD="$GEN_CMD --config $CONFIG_FILE"
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
    
    echo "    Copying chunk files..."
    scp -q scoring_chunks/*.json ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/scoring_chunks/ 2>/dev/null || true
    
    echo "    Copying full_scoring_worker.py..."
    scp -q full_scoring_worker.py ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
    
    echo "    ✓ Data distributed to $NODE"
done

echo "✓ Data distribution complete"

# ============================================================================
# Phase 3: Execute Distributed Jobs
# ============================================================================
echo ""
echo "Phase 3: Executing distributed jobs..."
echo "------------------------------------------------------------"

python3 coordinator.py --jobs-file scoring_jobs.json

echo "✓ Job execution complete"

# ============================================================================
# Phase 4: Pull Results from Remote Nodes
# ============================================================================
echo ""
echo "Phase 4: Pulling results from remote nodes..."
echo "------------------------------------------------------------"

for NODE in $REMOTE_NODES; do
    echo "  ← $NODE"
    
    # Pull result files
    scp -q ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/full_scoring_results/*.json \
        full_scoring_results/ 2>/dev/null || echo "    (no results from $NODE)"
    
    # Clean up remote result files (optional)
    # ssh ${REMOTE_USER}@${NODE} "rm -f ${REMOTE_BASE}/full_scoring_results/*.json" 2>/dev/null || true
done

# Also include local results (from zeus GPUs)
if [[ -d "full_scoring_results" ]]; then
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
python3 << 'AGGREGATE_EOF'
import json
import sys
from pathlib import Path

results_dir = Path("full_scoring_results")
output_file = Path("survivors_with_scores.json")

print(f"Scanning {results_dir} for result files...")

all_survivors = []
error_count = 0
files_processed = 0

for result_file in sorted(results_dir.glob("chunk_*.json")):
    try:
        with open(result_file, 'r') as f:
            chunk_results = json.load(f)
        
        if isinstance(chunk_results, list):
            # FIXED: chunk_results are now full survivor objects, not floats
            for survivor in chunk_results:
                if isinstance(survivor, dict) and 'seed' in survivor:
                    all_survivors.append(survivor)
                    if 'error' in survivor:
                        error_count += 1
                else:
                    # This should not happen with full_scoring_worker.py
                    print(f"  WARNING: Unexpected format in {result_file.name}")
        
        files_processed += 1
        print(f"  ✓ {result_file.name}: {len(chunk_results)} survivors")
        
    except Exception as e:
        print(f"  ✗ {result_file.name}: {e}")
        error_count += 1

if not all_survivors:
    print("")
    print("❌ ERROR: No survivors aggregated!")
    print("Check that full_scoring_worker.py ran successfully on workers")
    sys.exit(1)

# Sort by score (highest first)
all_survivors.sort(key=lambda x: x.get('score', 0), reverse=True)

# Save aggregated results
with open(output_file, 'w') as f:
    json.dump(all_survivors, f, indent=2)

print("")
print("=" * 60)
print("AGGREGATION COMPLETE")
print("=" * 60)
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
    print(f"  Feature count: {len(sample['features'])}")
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
    
    # Check feature count
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
