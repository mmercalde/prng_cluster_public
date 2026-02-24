#!/usr/bin/env python3
"""
apply_s109_documentation_updates.py
Session 109: Documentation Update Patcher

Updates all documentation files to reflect changes from S100-S109:
  - Chapter 1: v2.0 → v3.1 (per-seed match rates, 7 intersection fields)
  - Chapter 3: v3.4 → v4.2 (bidirectional_count objective, WSI removal)
  - COMPLETE_OPERATING_GUIDE: v2.0.0 → v2.1.0 (threshold bounds, search space, selfplay)

Usage:
    cd ~/distributed_prng_analysis
    python3 apply_s109_documentation_updates.py [--dry-run]
"""

import os
import sys
import re
import shutil
from datetime import datetime

DRY_RUN = "--dry-run" in sys.argv
DOCS_DIR = "docs"
BACKUP_SUFFIX = f".bak_s109_{datetime.now().strftime('%Y%m%d')}"

changes_applied = []
changes_failed = []


def backup_and_patch(filepath, patches, description):
    """Apply a list of (old, new) string replacement patches to a file."""
    if not os.path.exists(filepath):
        changes_failed.append(f"FILE NOT FOUND: {filepath}")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    for i, (old, new) in enumerate(patches):
        if old not in content:
            changes_failed.append(f"PATCH {i+1} NOT FOUND in {filepath}: {old[:60]}...")
            continue
        count = content.count(old)
        if count > 1:
            print(f"  ⚠️  Patch {i+1} matches {count} times in {filepath} — applying first only")
            content = content.replace(old, new, 1)
        else:
            content = content.replace(old, new)
        changes_applied.append(f"{filepath} patch {i+1}: {description}")

    if content == original:
        print(f"  ℹ️  No changes needed for {filepath}")
        return False

    if DRY_RUN:
        print(f"  🔍 DRY RUN: Would patch {filepath}")
        return True

    # Create backup
    backup = filepath + BACKUP_SUFFIX
    shutil.copy2(filepath, backup)
    print(f"  📋 Backup: {backup}")

    with open(filepath, "w") as f:
        f.write(content)

    print(f"  ✅ Patched: {filepath}")
    return True


def patch_chapter_1():
    """Update Chapter 1: Window Optimizer (v2.0 → v3.1)"""
    print("\n" + "=" * 60)
    print("CHAPTER 1: Window Optimizer — v2.0 → v3.1")
    print("=" * 60)

    filepath = os.path.join(DOCS_DIR, "CHAPTER_1_WINDOW_OPTIMIZER.md")
    if not os.path.exists(filepath):
        filepath = "CHAPTER_1_WINDOW_OPTIMIZER.md"

    patches = [
        # Patch 1: Version header
        (
            "**Version:** 2.0  \n**File:** `window_optimizer.py`  \n**Lines:** 868  ",
            "**Version:** 3.1  \n**File:** `window_optimizer.py` + `window_optimizer_integration_final.py`  \n**Lines:** ~868 + ~595  "
        ),

        # Patch 2: Version features section
        (
            """### 1.2 Version 2.0 Features

```
NEW IN V2.0:
- --test-both-modes flag: Test constant AND variable skip patterns
- Survivors tagged with skip_mode metadata for ML feature engineering
- Backward compatible: defaults to constant skip only
```""",
            """### 1.2 Version History

```
VERSION 3.1 (S104, Feb 2026):
- RESTORED: 7 intersection fields accidentally omitted in v3.0 rewrite
  (intersection_count, intersection_ratio, forward_only_count,
   reverse_only_count, survivor_overlap_ratio, bidirectional_selectivity,
   intersection_weight)
- These fields represent ~32% of ML feature importance (Chapter 6/11)

VERSION 3.0 (S103, Feb 2026):
- Per-seed match rates: forward_matches/reverse_matches now store per-seed
  values from GPU sieve kernel, not trial-level aggregates
- extract_survivor_records() preserves individual match_rate per seed
- Legacy alias extract_survivors_from_result() retained for compatibility
- convert_survivors_to_binary.py v3.1: maps to per-seed match rates
- NPZ percentage-based variance health check added

VERSION 2.0:
- --test-both-modes flag: Test constant AND variable skip patterns
- Survivors tagged with skip_mode metadata for ML feature engineering
- Backward compatible: defaults to constant skip only
```"""
        ),

        # Patch 3: SearchBounds threshold values (CRITICAL — old values are wrong)
        (
            """    # Threshold bounds (LOW for discovery)
    min_forward_threshold: float = 0.001
    max_forward_threshold: float = 0.10
    min_reverse_threshold: float = 0.001
    max_reverse_threshold: float = 0.10
    
    # Defaults
    default_forward_threshold: float = 0.01
    default_reverse_threshold: float = 0.01""",
            """    # Threshold bounds (Governance ruling Jan 25, 2026)
    min_forward_threshold: float = 0.15
    max_forward_threshold: float = 0.60
    min_reverse_threshold: float = 0.15
    max_reverse_threshold: float = 0.60
    
    # Defaults
    default_forward_threshold: float = 0.25
    default_reverse_threshold: float = 0.25"""
        ),

        # Patch 4: Survivor record structure — add per-seed match rates
        (
            """### 12.3 Survivor Record Structure

```json
{
    "seed": 12345678,
    "score": 0.85,
    "prng_type": "java_lcg",
    "skip_mode": "constant",
    "window_config": {
        "window_size": 256,
        "offset": 50,
        "skip_min": 0,
        "skip_max": 30
    },
    "trial_number": 23,
    "timestamp": "2025-12-15T14:30:52"
}""",
            """### 12.3 Survivor Record Structure (v3.1)

```json
{
    "seed": 12345678,
    "score": 0.85,
    "forward_match_rate": 0.75,
    "reverse_match_rate": 0.50,
    "prng_type": "java_lcg",
    "skip_mode": "constant",
    "window_config": {
        "window_size": 4,
        "offset": 26,
        "skip_min": 1,
        "skip_max": 108
    },
    "trial_number": 6,
    "bidirectional_count": 8929,
    "intersection_ratio": 0.488,
    "forward_only_count": 8987,
    "reverse_only_count": 8855,
    "survivor_overlap_ratio": 0.498,
    "bidirectional_selectivity": 1.007,
    "intersection_weight": 0.244,
    "timestamp": "2026-02-23T16:00:00"
}

# v3.0+: forward_match_rate and reverse_match_rate are PER-SEED
# values from GPU sieve kernel (not trial-level aggregates).
# v3.1: All 7 intersection fields restored (S104 fix).
```"""
        ),

        # Patch 5: CLI threshold values
        (
            """# Threshold parameters
--forward-threshold   # Override Optuna optimization (0.5-0.95)
--reverse-threshold   # Override Optuna optimization (0.6-0.98)""",
            """# Threshold parameters (governance bounds: 0.15-0.60)
--forward-threshold   # Override Optuna optimization (0.15-0.60)
--reverse-threshold   # Override Optuna optimization (0.15-0.60)"""
        ),
    ]

    backup_and_patch(filepath, patches, "Chapter 1 v2.0→v3.1")


def patch_chapter_3():
    """Update Chapter 3: Scorer Meta-Optimizer (v3.4 → v4.2)"""
    print("\n" + "=" * 60)
    print("CHAPTER 3: Scorer Meta-Optimizer — v3.4 → v4.2")
    print("=" * 60)

    filepath = os.path.join(DOCS_DIR, "CHAPTER_3_SCORER_META_OPTIMIZER.md")
    if not os.path.exists(filepath):
        filepath = "CHAPTER_3_SCORER_META_OPTIMIZER.md"

    patches = [
        # Patch 1: Version header
        (
            "**Version:** 3.4  \n**File:** `scorer_trial_worker.py`  \n**Lines:** ~350  ",
            "**Version:** 4.2  \n**File:** `scorer_trial_worker.py`  \n**Lines:** ~640  "
        ),

        # Patch 2: Key features table
        (
            """| **Holdout Evaluation** | v3.4 critical fix for proper validation |""",
            """| **Holdout Evaluation** | v3.4 critical fix for proper validation |
| **NPZ-Based Objective** | v4.2: bidirectional_count from NPZ (no draw history) |
| **Subset Selection** | v4.2: Per-trial survivor subsets with unique seeds |"""
        ),

        # Patch 3: Version history — add v3.5 through v4.2
        (
            """### 4.1 Version Timeline

```
Version 3.4 - December 2025 (CURRENT)
├── CRITICAL FIX: Holdout evaluation uses sampled seeds
├── Previous: Holdout evaluated on ALL seeds (wrong!)
└── Now: Holdout evaluated on SAME sampled seeds as training

Version 3.3 - November 2025
├── GPU-vectorized scoring via SurvivorScorer.extract_ml_features_batch()
├── Adaptive memory batching for 8GB VRAM
└── 3.8x performance improvement

Version 3.2 - November 2025
├── --params-file support for shorter SSH commands
├── Previous: All params passed via CLI args (SSH command too long)
└── Now: Params written to JSON file, worker reads file

Version 3.1 - October 2025
├── ROCm environment setup at file top
└── Mining mode support for AMD rigs

Version 3.0 - October 2025
├── Initial distributed implementation
└── Basic trial execution
```""",
            """### 4.1 Version Timeline

```
Version 4.2 - February 2026 (CURRENT — S107/S108)
├── bidirectional_count as primary objective signal
├── bidirectional_selectivity dropped (98.8% at floor — unusable)
├── Secondary bonus: intersection_ratio (weight=0.10)
├── Global percentile rank objective (stable across trials)
├── Median for robustness (TB Q2 ruling)
├── ir_disabled guard added
└── 640 lines total

Version 4.1 - February 2026 (S107 draft, superseded)
├── Weighted Survivor Index (WSI) objective proposed
├── Identified as tautological (batch_score measures same as sieve)
└── Replaced by bidirectional_count in v4.2

Version 4.0 - February 2026 (S105/S106)
├── WSI + IQR composite objective designed
├── IQR component confirmed tautological by Team Beta
└── Blocked pending TB ruling → led to v4.1/v4.2

Version 3.6 - February 2026 (S101)
├── neg-MSE → Spearman correlation (rank-based, more robust)
├── random.seed(42) → per-trial unique seed
└── Fixed all-identical survivor sampling across trials

Version 3.5 - January 2026 (S101)
├── Degenerate guard: std < 1e-12 early exit
└── Draw-history dependency still present (architectural issue)

Version 3.4 - December 2025
├── CRITICAL FIX: Holdout evaluation uses sampled seeds
├── Previous: Holdout evaluated on ALL seeds (wrong!)
└── Now: Holdout evaluated on SAME sampled seeds as training

Version 3.3 - November 2025
├── GPU-vectorized scoring via SurvivorScorer.extract_ml_features_batch()
├── Adaptive memory batching for 8GB VRAM
└── 3.8x performance improvement

Version 3.2 - November 2025
├── --params-file support for shorter SSH commands
└── Params written to JSON file, worker reads file

Version 3.1 - October 2025
├── ROCm environment setup at file top
└── Mining mode support for AMD rigs

Version 3.0 - October 2025
├── Initial distributed implementation
└── Basic trial execution
```

### 4.2 v4.2 Architectural Redesign (S103-S108)

**The Problem (v3.4-v3.6):** Step 2 scored survivors by comparing PRNG outputs
to draw history — duplicating Chapter 13's function. Under mod1000, all scores
were 0.0 (degenerate). Spearman correlation failed on zero-variance input.

**The Fix (v4.2):** Step 2 now uses NPZ intrinsic fields as ground truth:
- `bidirectional_count`: How many seeds survived both sieves (primary signal)
- `intersection_ratio`: Jaccard overlap quality (secondary, weight=0.10)
- NO draw history dependency. The sieve itself is the ground truth.

**Clean Separation Restored:**
```
Step 2: Find optimal SCORING PARAMETERS using SIEVE QUALITY as ground truth
Chapter 13: Compare predictions to real draws (post-prediction validation)
```"""
        ),

        # Patch 4: Trial parameters search space — add full v4.2 params
        (
            """### 6.1 Optuna Search Space

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `residue_mod_1` | 5-20 | Small residue modulus |
| `residue_mod_2` | 50-150 | Medium residue modulus |
| `residue_mod_3` | 500-1500 | Large residue modulus |
| `max_offset` | 1-15 | Temporal alignment offset |
| `temporal_window_size` | [50, 100, 150, 200] | Stability analysis window |""",
            """### 6.1 Optuna Search Space (v4.2)

| Parameter | Range | Purpose | Consumer |
|-----------|-------|---------|----------|
| `residue_mod_1` | 5-20 | Small residue modulus | Step 3 SurvivorScorer |
| `residue_mod_2` | 50-150 | Medium residue modulus | Step 3 SurvivorScorer |
| `residue_mod_3` | 500-1500 | Large residue modulus | Step 3 SurvivorScorer |
| `max_offset` | 1-15 | Temporal alignment offset | Step 3 SurvivorScorer |
| `temporal_window_size` | 50-200 | Stability analysis window | Step 3 SurvivorScorer |
| `temporal_num_windows` | 1-10 | Number of temporal windows | Step 3 SurvivorScorer |
| `min_confidence_threshold` | 0.05-0.50 | Min confidence for inclusion | Step 3 SurvivorScorer |
| `hidden_layers` | categorical | NN architecture string | Step 5 anti_overfit |
| `dropout` | 0.1-0.5 | NN dropout rate | Step 5 anti_overfit |
| `learning_rate` | 1e-4 to 1e-2 | NN learning rate | Step 5 anti_overfit |
| `batch_size` | [32,64,128,256] | NN batch size | Step 5 anti_overfit |

**Note (S108 audit):** All parameters are downstream consumers. rm1/rm2/rm3 +
max_offset + tw_size + tw_windows + min_conf feed Step 3 SurvivorScorer.
hidden_layers + dropout + lr + batch_size feed Step 5 anti_overfit_trial_worker.
Nothing to remove from the search space."""
        ),

        # Patch 5: run_trial function — replace old draw-dependent version
        (
            """### 7.2 Holdout Evaluation (v3.4)

```python
def run_trial(survivors, train_history, holdout_history, params, trial_id):
    \"\"\"
    Run single scorer trial with proper holdout evaluation.
    
    v3.4 CRITICAL FIX: Use SAME sampled seeds for both train and holdout.
    \"\"\"
    # Initialize scorer with trial params
    scorer = SurvivorScorer(
        prng_type='java_lcg',
        mod=1000,
        config_dict={
            'residue_mod_1': params['residue_mod_1'],
            'residue_mod_2': params['residue_mod_2'],
            'residue_mod_3': params['residue_mod_3'],
            'max_offset': params['max_offset'],
            'temporal_window_size': params['temporal_window_size']
        }
    )
    
    # Sample survivors (same sample for both evaluations!)
    sample_size = min(10000, len(survivors))
    sampled_seeds = random.sample([s['seed'] for s in survivors], sample_size)
    
    # Score on training data
    train_features = scorer.extract_ml_features_batch(
        seeds=sampled_seeds,
        lottery_history=train_history
    )
    train_scores = [f['score'] for f in train_features]
    
    # Score on holdout data (SAME seeds, different history)
    holdout_features = scorer.extract_ml_features_batch(
        seeds=sampled_seeds,  # v3.4 FIX: Same seeds!
        lottery_history=holdout_history
    )
    holdout_scores = [f['score'] for f in holdout_features]
    
    # Compute metrics
    return {
        'trial_id': trial_id,
        'params': params,
        'mean_train_score': np.mean(train_scores),
        'mean_holdout_score': np.mean(holdout_scores),
        'generalization_gap': np.mean(train_scores) - np.mean(holdout_scores),
        'std_train_score': np.std(train_scores),
        'std_holdout_score': np.std(holdout_scores),
        'n_survivors_scored': len(sampled_seeds),
        'top_k_holdout_score': np.mean(sorted(holdout_scores, reverse=True)[:100])
    }
```""",
            """### 7.2 Objective Function (v4.2)

**v4.2 uses NPZ intrinsic fields — NO draw history dependency.**

```python
def run_trial(params, npz_data, trial_number):
    \"\"\"
    v4.2: Score using bidirectional_count from NPZ as ground truth.
    
    The sieve's own evidence (how many seeds survived both directions)
    is the objective signal. No draw history needed.
    \"\"\"
    # Load NPZ fields for this trial's survivor subset
    trial_mask = npz_data['trial_number'] == trial_number
    bid_counts = npz_data['bidirectional_count'][trial_mask]
    ir_values = npz_data['intersection_ratio'][trial_mask]

    if len(bid_counts) == 0:
        return {'accuracy': 0.0, 'params': params}

    # Primary signal: bidirectional_count (survival frequency)
    # Use median for robustness against heavy-tail distributions (TB Q2)
    bid_score = np.median(bid_counts)

    # Normalize to global percentile rank (stable across trials)
    all_bid = npz_data['bidirectional_count']
    percentile = np.searchsorted(np.sort(all_bid), bid_score) / len(all_bid)

    # Secondary bonus: intersection_ratio (quality of overlap)
    ir_bonus = 0.0
    if np.std(ir_values) > 1e-12:  # ir_disabled guard
        ir_bonus = 0.10 * np.median(ir_values)

    accuracy = percentile + ir_bonus

    return {
        'accuracy': accuracy,
        'params': params,
        'bid_median': float(bid_score),
        'ir_median': float(np.median(ir_values)),
        'n_survivors': int(trial_mask.sum()),
    }
```

**Key Design Decisions (S103-S108):**
- `bidirectional_selectivity` dropped (98.8% at floor — unusable signal)
- Median over mean: robust to heavy-tail count distributions
- Global percentile: stable across different trial sizes
- `ir_disabled` guard: prevents NaN when intersection_ratio is constant"""
        ),

        # Patch 6: Trial metrics table
        (
            """### 7.3 Why Holdout Matters

| Metric | Meaning |
|--------|---------|
| `mean_train_score` | How well params fit the training data |
| `mean_holdout_score` | How well params generalize to unseen data |
| `generalization_gap` | Train - Holdout (large = overfitting) |
| `top_k_holdout_score` | Best survivors on holdout (what we care about) |""",
            """### 7.3 Trial Metrics (v4.2)

| Metric | Meaning |
|--------|---------|
| `accuracy` | Primary objective: percentile(bid_count) + 0.10×ir_bonus |
| `bid_median` | Median bidirectional_count for this trial's survivors |
| `ir_median` | Median intersection_ratio for this trial's survivors |
| `n_survivors` | Number of survivors in this trial's subset |

**Note:** v3.4's holdout train/test split is no longer relevant in v4.2
since the objective uses NPZ intrinsic fields, not draw-based scoring."""
        ),
    ]

    backup_and_patch(filepath, patches, "Chapter 3 v3.4→v4.2")


def patch_operating_guide():
    """Update COMPLETE_OPERATING_GUIDE (v2.0.0 → v2.1.0)"""
    print("\n" + "=" * 60)
    print("COMPLETE_OPERATING_GUIDE — v2.0.0 → v2.1.0")
    print("=" * 60)

    filepath = os.path.join(DOCS_DIR, "COMPLETE_OPERATING_GUIDE_v2_0.md")
    if not os.path.exists(filepath):
        filepath = "COMPLETE_OPERATING_GUIDE_v2_0.md"

    patches = [
        # Patch 1: Version header
        (
            "**Version 2.0.0**  \n**February 2026**  \n**Updated: Session 83 (Feb 13, 2026)**",
            "**Version 2.1.0**  \n**February 2026**  \n**Updated: Session 109 (Feb 23, 2026)**"
        ),

        # Patch 2: Step 1 threshold bounds
        (
            "**Key parameters:** window_size (2-500), offset (0-100), skip_min (0-10), skip_max (10-500), forward_threshold (0.001-0.1), reverse_threshold (0.001-0.1)",
            "**Key parameters:** window_size (2-500), offset (0-100), skip_min (0-10), skip_max (10-500), forward_threshold (0.15-0.60), reverse_threshold (0.15-0.60)\n\n**Integration layer:** `window_optimizer_integration_final.py` v3.1 — per-seed match rates + 7 intersection fields (S103/S104 fixes)"
        ),

        # Patch 3: Step 2 search space — add missing params
        (
            """### Search Space

| Parameter | Range | Purpose |
|-----------|-------|---------|
| residue_mod_1 | 5-20 | Small residue modulus |
| residue_mod_2 | 50-150 | Medium residue modulus |
| residue_mod_3 | 500-1500 | Large residue modulus |
| max_offset | 1-15 | Temporal alignment offset |
| temporal_window_size | [50,100,150,200] | Stability analysis window |""",
            """### Search Space (v4.2 — S108 verified)

| Parameter | Range | Purpose | Consumer |
|-----------|-------|---------|----------|
| residue_mod_1 | 5-20 | Small residue modulus | Step 3 |
| residue_mod_2 | 50-150 | Medium residue modulus | Step 3 |
| residue_mod_3 | 500-1500 | Large residue modulus | Step 3 |
| max_offset | 1-15 | Temporal alignment offset | Step 3 |
| temporal_window_size | 50-200 | Stability analysis window | Step 3 |
| temporal_num_windows | 1-10 | Number of temporal windows | Step 3 |
| min_confidence_threshold | 0.05-0.50 | Min confidence | Step 3 |
| hidden_layers | categorical | NN architecture | Step 5 |
| dropout | 0.1-0.5 | NN dropout | Step 5 |
| learning_rate | 1e-4 to 1e-2 | NN learning rate | Step 5 |
| batch_size | [32,64,128,256] | NN batch size | Step 5 |

**v4.2 Objective:** `bidirectional_count` percentile rank + 0.10×`intersection_ratio` bonus (NPZ-based, no draw history dependency)"""
        ),
    ]

    backup_and_patch(filepath, patches, "Operating Guide v2.0→v2.1")


def main():
    print("=" * 60)
    print("S109 DOCUMENTATION UPDATE PATCHER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 60)

    # Check we're in the right directory
    if not os.path.exists("agents") and not os.path.exists(DOCS_DIR):
        print("❌ Not in distributed_prng_analysis directory!")
        print("   Run: cd ~/distributed_prng_analysis")
        sys.exit(1)

    # Run all patches
    patch_chapter_1()
    patch_chapter_3()
    patch_operating_guide()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Changes applied: {len(changes_applied)}")
    for c in changes_applied:
        print(f"   • {c}")

    if changes_failed:
        print(f"\n⚠️  Changes failed: {len(changes_failed)}")
        for c in changes_failed:
            print(f"   • {c}")

    if not DRY_RUN and changes_applied:
        print(f"\n📋 Backups created with suffix: {BACKUP_SUFFIX}")
        print("\nNext steps:")
        print("  1. Review changes: git diff docs/")
        print("  2. Commit: git add docs/ && git commit -m 'docs(S109): Chapter 1 v3.1, Chapter 3 v4.2, Guide v2.1'")
        print("  3. Push: git push origin main && git push public main")
        print("  4. Copy to ser8: scp docs/CHAPTER_*.md docs/COMPLETE_OPERATING_GUIDE_v2_0.md michael@ser8:~/Downloads/CONCISE_OPERATING_GUIDE_v1.0/")


if __name__ == "__main__":
    main()
