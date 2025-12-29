#!/usr/bin/env python3
"""
Feature Integrity Validator
===========================
Phase 0 requirement from Team Beta.

This harness validates feature quality BEFORE ML training to prevent
"garbage in, garbage out" scenarios.

Checks:
1. Variance report - identify constant/zero-variance features
2. Correlation scan - flag potential label leakage (|corr| > 0.98)
3. Shuffle-label test - R² must collapse near 0 when labels shuffled

MANDATORY (Team Beta):
All leakage tests operate on the FINAL Step-5 training label (post-aggregation),
not intermediate scores or surrogate targets.

Usage:
    python3 feature_integrity_validator.py --survivors survivors_with_scores.json --sample 50000
    python3 feature_integrity_validator.py --survivors survivors_with_scores.json --full-check
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings

# Suppress sklearn warnings during import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_survivors_sample(filepath: str, sample_size: int = 50000) -> Tuple[List[Dict], int]:
    """Load survivors, optionally sampling for speed."""
    print(f"Loading survivors from {filepath}...")
    with open(filepath, 'r') as f:
        survivors = json.load(f)
    
    total_count = len(survivors)
    
    if sample_size and sample_size < total_count:
        # Deterministic sampling for reproducibility
        np.random.seed(42)
        indices = np.random.choice(total_count, sample_size, replace=False)
        survivors = [survivors[i] for i in sorted(indices)]
        print(f"  Sampled {len(survivors):,} from {total_count:,} survivors")
    else:
        print(f"  Loaded {total_count:,} survivors (full dataset)")
    
    return survivors, total_count


def extract_feature_matrix(survivors: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and labels y from survivors."""
    # Get feature names from first survivor
    feature_names = sorted([k for k in survivors[0]['features'].keys() 
                           if k not in ('score', 'confidence')])
    
    X = np.zeros((len(survivors), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(survivors), dtype=np.float32)
    
    for i, surv in enumerate(survivors):
        features = surv['features']
        y[i] = features.get('score', 0.0)
        for j, fname in enumerate(feature_names):
            X[i, j] = features.get(fname, 0.0)
    
    return X, y, feature_names


def compute_variance_report(X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict:
    """
    Compute per-feature variance statistics.
    
    Returns dict with per-feature stats and summary.
    """
    print("\n[1/3] Computing variance report...")
    
    report = {
        'features': {},
        'summary': {},
        'flags': []
    }
    
    zero_variance_count = 0
    low_variance_count = 0
    nan_features = []
    
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        
        # Handle NaN/Inf
        nan_count = np.sum(np.isnan(col))
        inf_count = np.sum(np.isinf(col))
        valid_col = col[np.isfinite(col)]
        
        if len(valid_col) == 0:
            stats = {
                'mean': None, 'std': None, 'min': None, 'max': None,
                'unique_count': 0, 'pct_zero': 100.0, 'pct_nan': 100.0,
                'status': 'ALL_INVALID'
            }
            nan_features.append(fname)
        else:
            unique_count = len(np.unique(valid_col))
            pct_zero = 100.0 * np.sum(valid_col == 0) / len(valid_col)
            pct_nan = 100.0 * nan_count / len(col) if len(col) > 0 else 0
            
            stats = {
                'mean': float(np.mean(valid_col)),
                'std': float(np.std(valid_col)),
                'min': float(np.min(valid_col)),
                'max': float(np.max(valid_col)),
                'unique_count': int(unique_count),
                'pct_zero': float(pct_zero),
                'pct_nan': float(pct_nan),
                'status': 'OK'
            }
            
            # Flag issues
            if unique_count <= 1:
                stats['status'] = 'ZERO_VARIANCE'
                zero_variance_count += 1
                report['flags'].append(f"ZERO_VARIANCE: {fname} (unique={unique_count})")
            elif unique_count < 10:
                stats['status'] = 'LOW_VARIANCE'
                low_variance_count += 1
                report['flags'].append(f"LOW_VARIANCE: {fname} (unique={unique_count})")
            
            if pct_nan > 0:
                report['flags'].append(f"HAS_NAN: {fname} ({pct_nan:.1f}%)")
        
        report['features'][fname] = stats
    
    # Summary
    report['summary'] = {
        'total_features': len(feature_names),
        'zero_variance': zero_variance_count,
        'low_variance': low_variance_count,
        'features_with_nan': len(nan_features),
        'features_ok': len(feature_names) - zero_variance_count - len(nan_features),
        'sample_size': X.shape[0]
    }
    
    # Print summary
    print(f"  Total features: {len(feature_names)}")
    print(f"  Zero variance: {zero_variance_count}")
    print(f"  Low variance (<10 unique): {low_variance_count}")
    print(f"  Features with NaN: {len(nan_features)}")
    print(f"  Features OK: {report['summary']['features_ok']}")
    
    return report


def compute_correlation_scan(X: np.ndarray, y: np.ndarray,
                            feature_names: List[str],
                            threshold: float = 0.98) -> Dict:
    """
    Scan for features highly correlated with label (potential leakage).
    
    Flags any feature with |correlation| > threshold.
    """
    print(f"\n[2/3] Computing correlation scan (threshold={threshold})...")
    
    report = {
        'correlations': {},
        'flags': [],
        'threshold': threshold
    }
    
    high_corr_count = 0
    
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        
        # Skip if no variance
        if np.std(col) < 1e-10 or np.std(y) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        
        report['correlations'][fname] = corr
        
        if abs(corr) > threshold:
            high_corr_count += 1
            report['flags'].append(f"HIGH_CORRELATION: {fname} (r={corr:.4f})")
    
    report['high_correlation_count'] = high_corr_count
    
    # Print top correlations
    sorted_corrs = sorted(report['correlations'].items(), 
                         key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top 5 correlations with label:")
    for fname, corr in sorted_corrs[:5]:
        flag = " ⚠️ LEAKAGE?" if abs(corr) > threshold else ""
        print(f"    {fname}: {corr:.4f}{flag}")
    
    if high_corr_count > 0:
        print(f"  ⚠️  {high_corr_count} features exceed threshold!")
    else:
        print(f"  ✓ No features exceed threshold")
    
    return report


def run_shuffle_label_test(X: np.ndarray, y: np.ndarray,
                          n_iterations: int = 3) -> Dict:
    """
    Shuffle labels and verify R² collapses near 0.
    
    If R² stays high after shuffling, there's data leakage.
    """
    print(f"\n[3/3] Running shuffle-label test ({n_iterations} iterations)...")
    
    report = {
        'original_r2': None,
        'shuffled_r2_values': [],
        'shuffled_r2_mean': None,
        'test_passed': False,
        'threshold': 0.05
    }
    
    # Train on original labels
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use lightweight model for speed
    model = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                  n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    original_r2 = r2_score(y_val, y_pred)
    report['original_r2'] = float(original_r2)
    print(f"  Original R²: {original_r2:.4f}")
    
    # Test with shuffled labels
    shuffled_r2s = []
    for i in range(n_iterations):
        y_shuffled = y.copy()
        np.random.seed(i + 100)  # Different seed each iteration
        np.random.shuffle(y_shuffled)
        
        X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
            X, y_shuffled, test_size=0.2, random_state=42
        )
        
        model_s = RandomForestRegressor(n_estimators=50, max_depth=10,
                                        n_jobs=-1, random_state=42)
        model_s.fit(X_train_s, y_train_s)
        y_pred_s = model_s.predict(X_val_s)
        r2_s = r2_score(y_val_s, y_pred_s)
        shuffled_r2s.append(float(r2_s))
        print(f"  Shuffled R² (iter {i+1}): {r2_s:.4f}")
    
    report['shuffled_r2_values'] = shuffled_r2s
    report['shuffled_r2_mean'] = float(np.mean(shuffled_r2s))
    
    # Test passes if shuffled R² is within threshold of 0
    test_passed = abs(report['shuffled_r2_mean']) < report['threshold']
    report['test_passed'] = test_passed
    
    if test_passed:
        print(f"  ✓ PASSED: Shuffled R² mean = {report['shuffled_r2_mean']:.4f} (< {report['threshold']})")
    else:
        print(f"  ⚠️ FAILED: Shuffled R² mean = {report['shuffled_r2_mean']:.4f} (>= {report['threshold']})")
        print(f"    This may indicate data leakage!")
    
    return report


def generate_full_report(survivors_path: str, sample_size: int = 50000,
                        output_dir: str = 'validation_reports') -> Dict:
    """Generate full integrity report."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compute file hash
    file_hash = compute_file_hash(survivors_path)
    
    # Load data
    survivors, total_count = load_survivors_sample(survivors_path, sample_size)
    X, y, feature_names = extract_feature_matrix(survivors)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Feature names: {len(feature_names)}")
    
    # Run all checks
    variance_report = compute_variance_report(X, y, feature_names)
    correlation_report = compute_correlation_scan(X, y, feature_names)
    shuffle_report = run_shuffle_label_test(X, y)
    
    # Compile full report
    full_report = {
        'metadata': {
            'timestamp': timestamp,
            'survivors_file': survivors_path,
            'file_hash': file_hash,
            'total_survivors': total_count,
            'sample_size': len(survivors),
            'feature_count': len(feature_names)
        },
        'variance': variance_report,
        'correlation': correlation_report,
        'shuffle_test': shuffle_report,
        'acceptance': {}
    }
    
    # Check acceptance criteria (Team Beta requirements)
    acceptance = {
        'variance_ok': variance_report['summary']['features_ok'] >= 30,
        'no_leakage': shuffle_report['test_passed'],
        'low_correlation': correlation_report['high_correlation_count'] == 0,
        'overall': False
    }
    acceptance['overall'] = all([
        acceptance['variance_ok'],
        acceptance['no_leakage']
    ])
    full_report['acceptance'] = acceptance
    
    # Print summary
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA")
    print("="*70)
    print(f"  Features with variance >= 30: {'✓ PASS' if acceptance['variance_ok'] else '✗ FAIL'}")
    print(f"  No label leakage: {'✓ PASS' if acceptance['no_leakage'] else '✗ FAIL'}")
    print(f"  No high correlations: {'✓ PASS' if acceptance['low_correlation'] else '⚠️ WARNING'}")
    print(f"\n  OVERALL: {'✓ READY FOR TRAINING' if acceptance['overall'] else '✗ DO NOT TRAIN'}")
    print("="*70)
    
    # Save reports
    variance_file = output_path / f"variance_report_{timestamp}.json"
    correlation_file = output_path / f"correlation_report_{timestamp}.json"
    full_file = output_path / f"integrity_report_{timestamp}.json"
    
    with open(variance_file, 'w') as f:
        json.dump(variance_report, f, indent=2)
    print(f"\nVariance report: {variance_file}")
    
    with open(correlation_file, 'w') as f:
        json.dump(correlation_report, f, indent=2)
    print(f"Correlation report: {correlation_file}")
    
    with open(full_file, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"Full report: {full_file}")
    
    return full_report


def main():
    parser = argparse.ArgumentParser(
        description='Feature Integrity Validator - Phase 0 Team Beta requirement'
    )
    parser.add_argument('--survivors', type=str, required=True,
                       help='Path to survivors JSON file')
    parser.add_argument('--sample', type=int, default=50000,
                       help='Sample size for validation (default: 50000)')
    parser.add_argument('--output-dir', type=str, default='validation_reports',
                       help='Output directory for reports')
    parser.add_argument('--full-check', action='store_true',
                       help='Run on full dataset (no sampling)')
    parser.add_argument('--correlation-threshold', type=float, default=0.98,
                       help='Correlation threshold for leakage detection')
    
    args = parser.parse_args()
    
    sample_size = None if args.full_check else args.sample
    
    print("="*70)
    print("FEATURE INTEGRITY VALIDATOR")
    print("Phase 0 - Team Beta Requirement")
    print("="*70)
    
    report = generate_full_report(
        survivors_path=args.survivors,
        sample_size=sample_size,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    if report['acceptance']['overall']:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
