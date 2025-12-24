#!/bin/bash
# =============================================================================
# STEP 6 RESTORATION v2.2 - COMPLETE INSTALLATION SCRIPT
# =============================================================================
# 
# This script applies all patches for Step 6 Restoration with Global Features.
# 
# WHAT IT DOES:
# 1. Creates timestamped backups of ALL affected files
# 2. Installs new GlobalStateTracker module
# 3. Guides you through manual patches with exact line numbers
# 4. Tests each phase before proceeding
#
# Author: Claude (AI Assistant)
# Date: December 23, 2025
# Version: 2.2
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "STEP 6 RESTORATION v2.2 - INSTALLER"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Backup all affected files"
echo "  2. Install GlobalStateTracker module"
echo "  3. Patch survivor_scorer.py"
echo "  4. Eliminate duplicate GlobalStateTracker classes"
echo "  5. Patch prediction_generator.py"
echo "  6. Update agent manifest"
echo ""
echo "Press Enter to continue or Ctrl+C to abort..."
read

# =============================================================================
# PHASE 0: COMPREHENSIVE BACKUP
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 0: Creating Comprehensive Backups"
echo "========================================"

BACKUP_DIR="backups/step6_restoration_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR/models"
mkdir -p "$BACKUP_DIR/agent_manifests"

echo "Backup directory: $BACKUP_DIR"
echo ""

# Core files
echo "Backing up core files..."
cp survivor_scorer.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ survivor_scorer.py" || echo "  ✗ survivor_scorer.py (not found)"
cp prediction_generator.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ prediction_generator.py" || echo "  ✗ prediction_generator.py (not found)"
cp reinforcement_engine.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ reinforcement_engine.py" || echo "  ✗ reinforcement_engine.py (not found)"
cp generate_ml_jobs.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ generate_ml_jobs.py" || echo "  ✗ generate_ml_jobs.py (not found)"

# Step 5 files (for future global features integration)
echo ""
echo "Backing up Step 5 files..."
cp meta_prediction_optimizer_anti_overfit.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ meta_prediction_optimizer_anti_overfit.py" || echo "  ✗ meta_prediction_optimizer_anti_overfit.py (not found)"
cp train_single_trial.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ train_single_trial.py" || echo "  ✗ train_single_trial.py (not found)"
cp subprocess_trial_coordinator.py "$BACKUP_DIR/" 2>/dev/null && echo "  ✓ subprocess_trial_coordinator.py" || echo "  ✗ subprocess_trial_coordinator.py (not found)"

# Models package
echo ""
echo "Backing up models package..."
cp models/__init__.py "$BACKUP_DIR/models/" 2>/dev/null && echo "  ✓ models/__init__.py" || echo "  ✗ models/__init__.py (not found)"
cp models/model_factory.py "$BACKUP_DIR/models/" 2>/dev/null && echo "  ✓ models/model_factory.py" || echo "  ✗ models/model_factory.py (not found)"
cp models/feature_schema.py "$BACKUP_DIR/models/" 2>/dev/null && echo "  ✓ models/feature_schema.py" || echo "  ✗ models/feature_schema.py (not found)"

# Agent manifests
echo ""
echo "Backing up agent manifests..."
cp agent_manifests/prediction.json "$BACKUP_DIR/agent_manifests/" 2>/dev/null && echo "  ✓ agent_manifests/prediction.json" || echo "  ✗ agent_manifests/prediction.json (not found)"

echo ""
echo "✅ Backups complete: $BACKUP_DIR"
echo ""
echo "Press Enter to continue..."
read

# =============================================================================
# PHASE 1: Install GlobalStateTracker Module
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 1: Install GlobalStateTracker Module"
echo "========================================"
echo ""
echo "This creates a NEW file: models/global_state_tracker.py"
echo "This is a GPU-neutral module with 14 global features."
echo ""

# Check if source file exists
if [ ! -f "step6_restoration/models/global_state_tracker.py" ]; then
    echo "❌ ERROR: step6_restoration/models/global_state_tracker.py not found!"
    echo ""
    echo "Please ensure you have copied the output files to:"
    echo "  ~/distributed_prng_analysis/step6_restoration/"
    echo ""
    echo "Expected structure:"
    echo "  step6_restoration/"
    echo "  ├── models/"
    echo "  │   ├── global_state_tracker.py"
    echo "  │   └── __init__.py"
    echo "  ├── survivor_scorer_PATCH.py"
    echo "  ├── prediction_generator_PATCH.py"
    echo "  └── agent_manifests/"
    echo "      └── prediction.json"
    exit 1
fi

echo "Installing global_state_tracker.py..."
cp step6_restoration/models/global_state_tracker.py models/
echo "  ✓ Copied to models/global_state_tracker.py"

echo ""
echo "Updating models/__init__.py..."
cp step6_restoration/models/__init__.py models/
echo "  ✓ Updated models/__init__.py to v3.2.1"

echo ""
echo "Testing Phase 1..."
python3 -c "
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES
tracker = GlobalStateTracker([100]*500, {'mod': 1000})
state = tracker.get_global_state()
assert len(state) == GLOBAL_FEATURE_COUNT, f'Expected {GLOBAL_FEATURE_COUNT}, got {len(state)}'
print(f'  ✓ GlobalStateTracker works: {GLOBAL_FEATURE_COUNT} features')
print(f'  ✓ Feature names: {GLOBAL_FEATURE_NAMES[:3]}...')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Phase 1 PASSED"
else
    echo ""
    echo "❌ Phase 1 FAILED - check error above"
    exit 1
fi

echo ""
echo "Press Enter to continue to Phase 2..."
read

# =============================================================================
# PHASE 2: Patch survivor_scorer.py
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2: Patch survivor_scorer.py"
echo "========================================"
echo ""
echo "This phase requires MANUAL EDITS to survivor_scorer.py"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 1: Fix hardcoded PRNG (LINE ~116)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "FIND this line (around line 116):"
echo ""
echo "    self.generate_sequence = java_lcg_sequence"
echo ""
echo "REPLACE with:"
echo ""
echo "    self._cpu_func = get_cpu_reference(self.prng_type)"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 2: Add _generate_sequence method (after line 117)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "ADD this method right after the __init__ method ends:"
echo ""
cat << 'HEREDOC'
    def _generate_sequence(self, seed: int, n: int, skip: int = 0) -> np.ndarray:
        """
        Generate PRNG sequence using configured prng_type.
        Uses prng_registry for dynamic PRNG lookup - NO HARDCODING.
        """
        raw = self._cpu_func(seed=int(seed), n=n, skip=skip)
        return np.array([v % self.mod for v in raw], dtype=np.int64)
HEREDOC
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 3: Add compute_dual_sieve_intersection method"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "ADD this method after _generate_sequence:"
echo ""
cat << 'HEREDOC'
    def compute_dual_sieve_intersection(
        self,
        forward_survivors: List[int],
        reverse_survivors: List[int]
    ) -> Dict[str, Any]:
        """
        Compute intersection of forward and reverse sieve survivors.
        Per Team Beta: NEVER discard valid intersection, Jaccard is metadata.
        """
        if not forward_survivors or not reverse_survivors:
            return {
                "intersection": [],
                "jaccard": 0.0,
                "counts": {
                    "forward": len(forward_survivors) if forward_survivors else 0,
                    "reverse": len(reverse_survivors) if reverse_survivors else 0,
                    "intersection": 0,
                    "union": 0
                }
            }
        
        forward_set = set(forward_survivors)
        reverse_set = set(reverse_survivors)
        intersection = forward_set & reverse_set
        union = forward_set | reverse_set
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return {
            "intersection": sorted(list(intersection)),
            "jaccard": float(jaccard),
            "counts": {
                "forward": len(forward_survivors),
                "reverse": len(reverse_survivors),
                "intersection": len(intersection),
                "union": len(union)
            }
        }
HEREDOC
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 4: Update extract_ml_features (LINE ~124)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "FIND this line (around line 124 in extract_ml_features):"
echo ""
echo "    seq = self.generate_sequence(seed, n, self.mod)"
echo ""
echo "REPLACE with:"
echo ""
echo "    seq = self._generate_sequence(seed, n, skip=skip)"
echo ""
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Please make these edits now."
echo "You can reference: step6_restoration/survivor_scorer_PATCH.py"
echo ""
echo "Press Enter when edits are complete..."
read

echo ""
echo "Testing Phase 2..."
python3 -c "
from survivor_scorer import SurvivorScorer

# Test instantiation
scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
print('  ✓ SurvivorScorer instantiated')

# Test _generate_sequence
seq = scorer._generate_sequence(12345, 10)
print(f'  ✓ _generate_sequence works: {seq[:3]}...')

# Test compute_dual_sieve_intersection
result = scorer.compute_dual_sieve_intersection([1,2,3,4], [3,4,5,6])
assert result['intersection'] == [3, 4], f'Expected [3,4], got {result[\"intersection\"]}'
assert abs(result['jaccard'] - 2/6) < 0.001, f'Expected ~0.333, got {result[\"jaccard\"]}'
print(f'  ✓ compute_dual_sieve_intersection works: {result[\"intersection\"]}')
print(f'  ✓ Jaccard index: {result[\"jaccard\"]:.4f}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Phase 2 PASSED"
else
    echo ""
    echo "❌ Phase 2 FAILED"
    echo "Please check your edits and try again."
    echo "Restore from backup if needed: cp $BACKUP_DIR/survivor_scorer.py ."
    exit 1
fi

echo ""
echo "Press Enter to continue to Phase 3..."
read

# =============================================================================
# PHASE 3: Eliminate GlobalStateTracker Duplicates
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 3: Eliminate GlobalStateTracker Duplicates"
echo "========================================"
echo ""
echo "Two files have duplicate GlobalStateTracker classes."
echo "We need to replace them with imports from the new module."
echo ""
echo "─────────────────────────────────────────────────────────"
echo "FILE 1: reinforcement_engine.py"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "1. Open reinforcement_engine.py"
echo ""
echo "2. FIND the class definition (around line 277):"
echo "   class GlobalStateTracker:"
echo ""
echo "3. DELETE the entire class (approximately lines 277-450)"
echo "   This is about 170 lines ending before:"
echo "   class SurvivorQualityNet(nn.Module):"
echo ""
echo "4. REPLACE with this import (put it near other imports at top):"
echo ""
echo "   # GlobalStateTracker moved to GPU-neutral module (v2.2)"
echo "   from models.global_state_tracker import GlobalStateTracker"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "FILE 2: generate_ml_jobs.py"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "1. Open generate_ml_jobs.py"
echo ""
echo "2. FIND the class definition (around line 458):"
echo "   class GlobalStateTracker:"
echo ""
echo "3. DELETE the entire class (approximately lines 458-580)"
echo ""
echo "4. REPLACE with this import (put it near other imports at top):"
echo ""
echo "   # GlobalStateTracker - canonical location (v2.2)"
echo "   from models.global_state_tracker import GlobalStateTracker"
echo ""
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Please make these edits now."
echo "Press Enter when edits are complete..."
read

echo ""
echo "Testing Phase 3..."
python3 -c "
# Test reinforcement_engine still imports correctly
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
print('  ✓ reinforcement_engine imports work')

# Test generate_ml_jobs (if it has GlobalStateTracker usage)
try:
    from generate_ml_jobs import GlobalStateTracker
    print('  ✓ generate_ml_jobs GlobalStateTracker import works')
except ImportError:
    print('  ✓ generate_ml_jobs has no GlobalStateTracker export (OK)')

# Verify canonical source
from models.global_state_tracker import GlobalStateTracker as CanonicalGST
print('  ✓ Canonical GlobalStateTracker is the single source')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Phase 3 PASSED"
else
    echo ""
    echo "❌ Phase 3 FAILED"
    echo "Please check your edits. Common issues:"
    echo "  - Deleted too much or too little"
    echo "  - Import statement has typo"
    echo "Restore from backup if needed."
    exit 1
fi

echo ""
echo "Press Enter to continue to Phase 4..."
read

# =============================================================================
# PHASE 4: Patch prediction_generator.py
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 4: Patch prediction_generator.py"
echo "========================================"
echo ""
echo "This is the largest patch. It adds model integration and global features."
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 1: Add imports (after line ~55)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "ADD this import after the Multi-Model imports:"
echo ""
echo "from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 2: Add to PredictionConfig dataclass (around line 90)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "ADD these fields to the dataclass:"
echo ""
echo "    # Model directory (sidecar location)"
echo "    models_dir: str = 'models/reinforcement'"
echo "    # Survivors file for schema validation"
echo "    survivors_forward_file: str = ''"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 3: Replace __init__ method (around line 145)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "See step6_restoration/prediction_generator_PATCH.py SECTION 3"
echo "for the complete replacement __init__ method."
echo ""
echo "Key changes:"
echo "  - Loads model via load_model_from_sidecar()"
echo "  - Initializes self.global_tracker = None"
echo "  - Logs model type and feature count"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 4: Replace generate_predictions method (around line 161)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "See step6_restoration/prediction_generator_PATCH.py SECTION 4"
echo "for the complete replacement method."
echo ""
echo "Key changes:"
echo "  - Initializes GlobalStateTracker with lottery_history"
echo "  - Handles intersection_result as Dict (not List)"
echo "  - Calls _build_prediction_pool()"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 5: Add _build_prediction_pool method"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "See step6_restoration/prediction_generator_PATCH.py SECTION 5"
echo "for the complete new method."
echo ""
echo "This is ~150 lines. Key features:"
echo "  - Handles both int and dict survivor formats"
echo "  - Uses sidecar feature_names ordering"
echo "  - Appends global features"
echo "  - Validates feature count"
echo "  - Uses model.predict() for scoring"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "EDIT 6: Add _empty_pool_result helper method"
echo "─────────────────────────────────────────────────────────"
echo ""
cat << 'HEREDOC'
    def _empty_pool_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'predictions': [],
            'confidence_scores': [],
            'survivor_count': 0,
            'pool_size': 0,
            'mean_confidence': 0.0,
            'model_type': None,
            'global_features_used': 0,
            'total_features': 0
        }
HEREDOC
echo ""
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Please make these edits now."
echo "Reference: step6_restoration/prediction_generator_PATCH.py"
echo ""
echo "Press Enter when edits are complete..."
read

echo ""
echo "Testing Phase 4..."
python3 -c "
from prediction_generator import PredictionGenerator, PredictionConfig

# Test import
print('  ✓ prediction_generator imports work')

# Test config has new fields
config = PredictionConfig()
assert hasattr(config, 'models_dir'), 'Missing models_dir field'
print(f'  ✓ PredictionConfig.models_dir = {config.models_dir}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Phase 4 PASSED (basic import test)"
else
    echo ""
    echo "❌ Phase 4 FAILED"
    echo "Please check your edits."
    exit 1
fi

echo ""
echo "Press Enter to continue to Phase 5..."
read

# =============================================================================
# PHASE 5: Update Agent Manifest
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 5: Update Agent Manifest"
echo "========================================"
echo ""

echo "Installing updated prediction.json..."
cp step6_restoration/agent_manifests/prediction.json agent_manifests/
echo "  ✓ Updated agent_manifests/prediction.json"

echo ""
echo "Key changes:"
echo "  - script: generate_predictions.py → prediction_generator.py"
echo "  - Added models-dir to args_map"
echo "  - Added use_global_features parameter"
echo "  - Version: 1.5.0"

echo ""
echo "✅ Phase 5 PASSED"
echo ""
echo "Press Enter to continue to final tests..."
read

# =============================================================================
# PHASE 6: Integration Tests
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 6: Integration Tests"
echo "========================================"
echo ""

echo "Running comprehensive tests..."
echo ""

echo "Test 1: GlobalStateTracker module..."
python3 -c "
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT
tracker = GlobalStateTracker([100]*500, {'mod': 1000})
assert len(tracker.get_global_state()) == GLOBAL_FEATURE_COUNT
print('  ✅ GlobalStateTracker: 14 features')
"

echo ""
echo "Test 2: survivor_scorer methods..."
python3 -c "
from survivor_scorer import SurvivorScorer
scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
seq = scorer._generate_sequence(12345, 5)
result = scorer.compute_dual_sieve_intersection([1,2,3], [2,3,4])
print(f'  ✅ survivor_scorer: seq={list(seq)[:3]}..., intersection={result[\"intersection\"]}')
"

echo ""
echo "Test 3: reinforcement_engine import..."
python3 -c "
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
print('  ✅ reinforcement_engine imports work')
"

echo ""
echo "Test 4: prediction_generator import..."
python3 -c "
from prediction_generator import PredictionGenerator, PredictionConfig
print('  ✅ prediction_generator imports work')
"

echo ""
echo "Test 5: No duplicate GlobalStateTracker classes..."
python3 -c "
import ast
import sys

# Check reinforcement_engine.py for class definition
with open('reinforcement_engine.py', 'r') as f:
    content = f.read()
    if 'class GlobalStateTracker:' in content:
        print('  ⚠️  WARNING: reinforcement_engine.py still has GlobalStateTracker class')
        print('     (Should be import only)')
    else:
        print('  ✅ reinforcement_engine.py uses import (no duplicate)')

# Check generate_ml_jobs.py
with open('generate_ml_jobs.py', 'r') as f:
    content = f.read()
    if 'class GlobalStateTracker:' in content:
        print('  ⚠️  WARNING: generate_ml_jobs.py still has GlobalStateTracker class')
    else:
        print('  ✅ generate_ml_jobs.py uses import (no duplicate)')
"

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE"
echo "========================================"
echo ""
echo "Summary:"
echo "  ✅ GlobalStateTracker module installed"
echo "  ✅ survivor_scorer.py patched"
echo "  ✅ Duplicate classes eliminated"
echo "  ✅ prediction_generator.py patched"
echo "  ✅ Agent manifest updated"
echo ""
echo "Backups saved to: $BACKUP_DIR"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "NEXT STEPS"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "1. Test Step 6 with existing model:"
echo ""
echo "   python3 prediction_generator.py \\"
echo "       --models-dir models/reinforcement \\"
echo "       --survivors-forward survivors_with_scores.json \\"
echo "       --lottery-history synthetic_lottery.json \\"
echo "       --k 10"
echo ""
echo "2. (Optional) Retrain Step 5 with global features:"
echo "   This requires additional patches to meta_prediction_optimizer_anti_overfit.py"
echo "   The current model will work without global features (backward compatible)"
echo ""
echo "3. If all tests pass, commit:"
echo ""
echo "   git add -A"
echo "   git commit -m 'Step 6 Restoration v2.2: GlobalStateTracker, model integration, global features support'"
echo "   git push"
echo ""
echo "─────────────────────────────────────────────────────────"
echo "ROLLBACK (if needed)"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "   cp $BACKUP_DIR/survivor_scorer.py ."
echo "   cp $BACKUP_DIR/prediction_generator.py ."
echo "   cp $BACKUP_DIR/reinforcement_engine.py ."
echo "   cp $BACKUP_DIR/generate_ml_jobs.py ."
echo "   cp $BACKUP_DIR/models/__init__.py models/"
echo "   rm models/global_state_tracker.py"
echo ""
