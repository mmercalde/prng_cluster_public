#!/usr/bin/env python3
"""
Sanity Tests for Step 5 Model Validation
=========================================

Team Beta Required Tests (December 22, 2025):
1. Shuffle-label test - R² should collapse to ~0 with random y
2. Holdout-by-time/window - Split by contiguous blocks, not random
3. Dedup test - No duplicate feature vectors across splits
4. Single-feature ablation - Remove top features, check if R² stays 1.0

These tests detect data leakage vs legitimate learning.

Usage:
    python3 sanity_tests.py --survivors survivors_with_scores.json --test shuffle
    python3 sanity_tests.py --survivors survivors_with_scores.json --test dedup
    python3 sanity_tests.py --survivors survivors_with_scores.json --test ablation
    python3 sanity_tests.py --survivors survivors_with_scores.json --test all

Author: PRNG Analysis System
Date: December 22, 2025
Version: 1.0.0
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
from collections import Counter

# Lazy imports for GPU libraries
def get_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor

def get_xgboost():
    import xgboost as xgb
    return xgb


@dataclass
class SanityTestResult:
    """Result from a sanity test."""
    test_name: str
    passed: bool
    message: str
    metrics: Dict
    recommendation: str


def load_features_and_labels(survivors_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features and labels from survivors file."""
    print(f"Loading {survivors_path}...")
    
    with open(survivors_path) as f:
        data = json.load(f)
    
    # Extract features
    feature_names = None
    X_list = []
    y_list = []
    
    for record in data:
        features = record.get('features', {})
        
        # Get feature names from first record
        if feature_names is None:
            feature_names = sorted([k for k in features.keys() if k != 'score'])
        
        # Extract feature values (excluding 'score')
        row = [features.get(name, 0.0) for name in feature_names]
        X_list.append(row)
        
        # Extract label
        y_list.append(features.get('score', 0.0))
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"  Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"  y range: [{y.min():.6f}, {y.max():.6f}]")
    
    return X, y, feature_names


def train_quick_model(X_train, y_train, X_val, y_val, model_type='catboost') -> Dict:
    """Train a quick model and return metrics."""
    
    if model_type == 'catboost':
        CatBoostRegressor = get_catboost()
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            task_type='GPU',
            devices='0'
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        xgb = get_xgboost()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'device': 'cuda'
        }
        model = xgb.train(params, dtrain, num_boost_round=100, 
                         evals=[(dval, 'val')], verbose_eval=False)
    
    # Predictions
    if model_type == 'catboost':
        val_preds = model.predict(X_val)
    else:
        val_preds = model.predict(dval)
    
    # Metrics
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {'val_mse': val_mse, 'r2': r2, 'model': model}


# =============================================================================
# TEST 1: SHUFFLE-LABEL TEST
# =============================================================================

def test_shuffle_label(X: np.ndarray, y: np.ndarray, n_trials: int = 3) -> SanityTestResult:
    """
    Shuffle-label test: randomly permute y and confirm R² collapses to ~0.
    
    If R² stays high with shuffled labels, there's likely data leakage.
    """
    print("\n" + "="*70)
    print("TEST: SHUFFLE-LABEL")
    print("="*70)
    print("If model can predict shuffled labels, features leak target info.")
    print()
    
    # Split data
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_orig, y_val_orig = y[train_idx], y[val_idx]
    
    # First, train with real labels
    print("Training with REAL labels...")
    real_result = train_quick_model(X_train, y_train_orig, X_val, y_val_orig)
    print(f"  Real R²: {real_result['r2']:.6f}")
    print(f"  Real MSE: {real_result['val_mse']:.6f}")
    
    # Now train with shuffled labels
    shuffle_r2_scores = []
    for trial in range(n_trials):
        print(f"\nTraining with SHUFFLED labels (trial {trial+1}/{n_trials})...")
        
        # Shuffle training labels
        y_train_shuffled = np.random.permutation(y_train_orig)
        # Note: validation labels stay original for fair comparison
        
        shuffle_result = train_quick_model(X_train, y_train_shuffled, X_val, y_val_orig)
        shuffle_r2_scores.append(shuffle_result['r2'])
        print(f"  Shuffled R²: {shuffle_result['r2']:.6f}")
    
    avg_shuffle_r2 = np.mean(shuffle_r2_scores)
    r2_drop = real_result['r2'] - avg_shuffle_r2
    
    print(f"\n{'─'*70}")
    print(f"RESULTS:")
    print(f"  Real R²:           {real_result['r2']:.6f}")
    print(f"  Avg Shuffled R²:   {avg_shuffle_r2:.6f}")
    print(f"  R² Drop:           {r2_drop:.6f}")
    
    # Pass criteria: shuffled R² should be near 0 (< 0.1)
    passed = avg_shuffle_r2 < 0.1 and r2_drop > 0.5
    
    if passed:
        message = "✅ PASSED: R² collapsed with shuffled labels. Model is learning real patterns."
        recommendation = "No action needed."
    else:
        message = f"❌ FAILED: Shuffled R² = {avg_shuffle_r2:.4f} is too high! Possible data leakage."
        recommendation = "Investigate feature derivation. Check if features are computed from target."
    
    print(f"\n{message}")
    
    return SanityTestResult(
        test_name="shuffle_label",
        passed=passed,
        message=message,
        metrics={
            'real_r2': real_result['r2'],
            'real_mse': real_result['val_mse'],
            'avg_shuffled_r2': avg_shuffle_r2,
            'shuffle_r2_scores': shuffle_r2_scores,
            'r2_drop': r2_drop
        },
        recommendation=recommendation
    )


# =============================================================================
# TEST 2: DEDUP TEST
# =============================================================================

def test_dedup(X: np.ndarray, y: np.ndarray) -> SanityTestResult:
    """
    Dedup test: verify no duplicate feature vectors across train/val/test.
    
    Duplicates can cause leakage if same sample appears in train and val.
    """
    print("\n" + "="*70)
    print("TEST: DEDUPLICATION")
    print("="*70)
    print("Checking for duplicate feature vectors...")
    print()
    
    # Hash each row
    row_hashes = []
    for i, row in enumerate(X):
        row_hash = hashlib.md5(row.tobytes()).hexdigest()
        row_hashes.append(row_hash)
    
    # Count duplicates
    hash_counts = Counter(row_hashes)
    duplicates = {h: c for h, c in hash_counts.items() if c > 1}
    
    n_unique = len(hash_counts)
    n_total = len(X)
    n_duplicate_groups = len(duplicates)
    n_duplicate_rows = sum(c - 1 for c in duplicates.values())
    
    print(f"  Total rows: {n_total}")
    print(f"  Unique rows: {n_unique}")
    print(f"  Duplicate groups: {n_duplicate_groups}")
    print(f"  Extra duplicate rows: {n_duplicate_rows}")
    
    # Check if duplicates have same labels
    if duplicates:
        print(f"\n  Checking if duplicates have consistent labels...")
        inconsistent_labels = 0
        for h, count in list(duplicates.items())[:5]:  # Check first 5
            indices = [i for i, rh in enumerate(row_hashes) if rh == h]
            labels = [y[i] for i in indices]
            if len(set(labels)) > 1:
                inconsistent_labels += 1
                print(f"    Hash {h[:8]}...: {count} copies, labels vary: {labels[:3]}...")
        
        if inconsistent_labels > 0:
            print(f"  ⚠️  {inconsistent_labels} duplicate groups have inconsistent labels!")
    
    # Pass criteria: < 1% duplicates
    dup_ratio = n_duplicate_rows / n_total
    passed = dup_ratio < 0.01
    
    if passed:
        message = f"✅ PASSED: Only {dup_ratio*100:.2f}% duplicates ({n_duplicate_rows} rows)."
        recommendation = "No action needed."
    else:
        message = f"❌ FAILED: {dup_ratio*100:.2f}% duplicates! ({n_duplicate_rows} rows)"
        recommendation = "Deduplicate data or use grouped splits to prevent leakage."
    
    print(f"\n{message}")
    
    return SanityTestResult(
        test_name="dedup",
        passed=passed,
        message=message,
        metrics={
            'n_total': n_total,
            'n_unique': n_unique,
            'n_duplicate_groups': n_duplicate_groups,
            'n_duplicate_rows': n_duplicate_rows,
            'duplicate_ratio': dup_ratio
        },
        recommendation=recommendation
    )


# =============================================================================
# TEST 3: FEATURE ABLATION
# =============================================================================

def test_feature_ablation(X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str],
                          top_n: int = 5) -> SanityTestResult:
    """
    Feature ablation test: remove top features and check if R² remains ~1.0.
    
    If R² stays at 1.0 after removing top features, target may be over-determined.
    """
    print("\n" + "="*70)
    print("TEST: FEATURE ABLATION")
    print("="*70)
    print(f"Removing top {top_n} features to check for over-determination...")
    print()
    
    # Split data
    n = len(X)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train with all features to get importance
    print("Training with ALL features to get importance ranking...")
    CatBoostRegressor = get_catboost()
    model = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        task_type='GPU',
        devices='0'
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # Get feature importance
    importance = model.get_feature_importance()
    importance_pairs = list(zip(feature_names, importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} most important features:")
    for i, (name, imp) in enumerate(importance_pairs[:top_n]):
        print(f"  {i+1}. {name}: {imp:.2f}")
    
    # Baseline R²
    val_preds = model.predict(X_val)
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    baseline_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    print(f"\nBaseline R² (all features): {baseline_r2:.6f}")
    
    # Progressive ablation
    ablation_results = []
    top_feature_indices = [feature_names.index(name) for name, _ in importance_pairs[:top_n]]
    
    for n_remove in range(1, top_n + 1):
        remove_indices = top_feature_indices[:n_remove]
        keep_indices = [i for i in range(len(feature_names)) if i not in remove_indices]
        
        X_train_ablated = X_train[:, keep_indices]
        X_val_ablated = X_val[:, keep_indices]
        
        # Train on ablated data
        model_ablated = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            task_type='GPU',
            devices='0'
        )
        model_ablated.fit(X_train_ablated, y_train, verbose=False)
        
        val_preds_ablated = model_ablated.predict(X_val_ablated)
        ss_res_ablated = np.sum((y_val - val_preds_ablated) ** 2)
        r2_ablated = float(1 - (ss_res_ablated / ss_tot)) if ss_tot > 0 else 0.0
        
        removed_names = [importance_pairs[i][0] for i in range(n_remove)]
        ablation_results.append({
            'n_removed': n_remove,
            'removed_features': removed_names,
            'r2': r2_ablated,
            'r2_drop': baseline_r2 - r2_ablated
        })
        
        print(f"\nRemoved top {n_remove} features: R² = {r2_ablated:.6f} (drop: {baseline_r2 - r2_ablated:.6f})")
    
    # Check if R² drops significantly
    final_r2 = ablation_results[-1]['r2']
    total_drop = baseline_r2 - final_r2
    
    # Pass criteria: removing top 5 features should drop R² by at least 0.3
    passed = total_drop > 0.3 or final_r2 < 0.7
    
    if passed:
        message = f"✅ PASSED: R² dropped to {final_r2:.4f} after ablation. Features contribute meaningfully."
        recommendation = "No action needed."
    else:
        message = f"❌ WARNING: R² still {final_r2:.4f} after removing top {top_n} features. Target may be over-determined."
        recommendation = "Check if multiple features encode the same information as target."
    
    print(f"\n{message}")
    
    return SanityTestResult(
        test_name="feature_ablation",
        passed=passed,
        message=message,
        metrics={
            'baseline_r2': baseline_r2,
            'final_r2': final_r2,
            'total_r2_drop': total_drop,
            'ablation_results': ablation_results,
            'top_features': [name for name, _ in importance_pairs[:top_n]]
        },
        recommendation=recommendation
    )


# =============================================================================
# TEST 4: LABEL VARIANCE GUARD
# =============================================================================

def test_label_variance(y: np.ndarray) -> SanityTestResult:
    """
    Label variance guard: ensure labels have sufficient variance.
    
    If y_max == y_min, training is meaningless.
    """
    print("\n" + "="*70)
    print("TEST: LABEL VARIANCE")
    print("="*70)
    
    y_min = float(y.min())
    y_max = float(y.max())
    y_std = float(y.std())
    y_unique = len(np.unique(y))
    
    print(f"  y_min: {y_min:.6f}")
    print(f"  y_max: {y_max:.6f}")
    print(f"  y_std: {y_std:.6f}")
    print(f"  y_unique: {y_unique}")
    
    # Pass criteria: variance > 0
    passed = y_std > 1e-6 and y_unique > 10
    
    if passed:
        message = f"✅ PASSED: Labels have sufficient variance (std={y_std:.6f}, unique={y_unique})."
        recommendation = "No action needed."
    else:
        message = f"❌ FAILED: Labels have insufficient variance! std={y_std:.6f}"
        recommendation = "Check label extraction. Ensure correct column is used."
    
    print(f"\n{message}")
    
    return SanityTestResult(
        test_name="label_variance",
        passed=passed,
        message=message,
        metrics={
            'y_min': y_min,
            'y_max': y_max,
            'y_std': y_std,
            'y_unique': y_unique
        },
        recommendation=recommendation
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sanity tests for Step 5 model validation (Team Beta requirements)'
    )
    parser.add_argument('--survivors', required=True, help='Path to survivors_with_scores.json')
    parser.add_argument('--test', choices=['shuffle', 'dedup', 'ablation', 'variance', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Load data
    X, y, feature_names = load_features_and_labels(args.survivors)
    
    results = []
    
    # Run requested tests
    if args.test in ['variance', 'all']:
        results.append(test_label_variance(y))
    
    if args.test in ['dedup', 'all']:
        results.append(test_dedup(X, y))
    
    if args.test in ['shuffle', 'all']:
        results.append(test_shuffle_label(X, y))
    
    if args.test in ['ablation', 'all']:
        results.append(test_feature_ablation(X, y, feature_names))
    
    # Summary
    print("\n" + "="*70)
    print("SANITY TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"{status} {r.test_name}: {r.message}")
        if not r.passed:
            all_passed = False
            print(f"   → {r.recommendation}")
    
    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Model learning appears legitimate")
    else:
        print("⚠️  SOME TESTS FAILED - Investigate before trusting results")
    
    # Save results
    if args.output:
        output_data = {
            'all_passed': all_passed,
            'tests': [
                {
                    'name': r.test_name,
                    'passed': r.passed,
                    'message': r.message,
                    'metrics': r.metrics,
                    'recommendation': r.recommendation
                }
                for r in results
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
