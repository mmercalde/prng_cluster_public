#!/usr/bin/env python3
"""
Quick Model Comparison Test (v1.0)
==================================

Tests all 4 model types using ModelSelector.train_and_compare().
This is Option 2: Quick test before full integration.

Output is agent_metadata compatible for AI decision making.

Usage:
    python3 test_model_comparison.py \
        --survivors survivors_with_scores.json \
        --max-survivors 10000 \
        --k-folds 3

Author: Distributed PRNG Analysis System
Date: December 21, 2025
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_survivors_for_comparison(survivors_file: str, max_survivors: int = None):
    """Load survivors and prepare X, y arrays for model comparison."""
    logger.info(f"Loading survivors from {survivors_file}...")
    
    try:
        import ijson
        survivors = []
        with open(survivors_file, 'rb') as f:
            for i, record in enumerate(ijson.items(f, 'item')):
                survivors.append(record)
                if max_survivors and i >= max_survivors - 1:
                    break
    except ImportError:
        logger.warning("ijson not available, using standard json")
        with open(survivors_file) as f:
            survivors = json.load(f)
            if max_survivors:
                survivors = survivors[:max_survivors]
    
    logger.info(f"  Loaded {len(survivors)} survivors")
    
    exclude_features = ['score', 'confidence']
    first_features = survivors[0].get('features', {})
    feature_names = sorted([k for k in first_features.keys() if k not in exclude_features])
    logger.info(f"  Features: {len(feature_names)}")
    
    X = []
    y = []
    for s in survivors:
        features = s.get('features', {})
        row = [features.get(f, 0.0) for f in feature_names]
        X.append(row)
        y.append(features.get('score', 0.0))
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y, feature_names


def run_comparison(args):
    """Run the model comparison test."""
    
    print("=" * 70)
    print("MULTI-MODEL COMPARISON TEST")
    print("=" * 70)
    print(f"Survivors: {args.survivors}")
    print(f"Max survivors: {args.max_survivors or 'all'}")
    print(f"Test size: {args.test_size}")
    print("=" * 70)
    
    start_time = time.time()
    
    X, y, feature_names = load_survivors_for_comparison(
        args.survivors, max_survivors=args.max_survivors
    )
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    from models import ModelSelector
    
    selector = ModelSelector(device='cuda:0')
    
    configs = {
        'neural_net': {
            'hidden_layers': [87, 59, 32],
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 256,
            'dropout': 0.2
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'tree_method': 'hist',
            'device': 'cuda'
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'device': 'gpu'
        },
        'catboost': {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'task_type': 'GPU',
            'verbose': False
        }
    }
    
    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS...")
    print("=" * 70)
    
    model_types = args.models.split(',') if args.models else ['lightgbm', 'neural_net', 'xgboost', 'catboost']
    
    results = selector.train_and_compare(
        X_train, y_train, X_val, y_val,
        model_types=model_types,
        configs=configs,
        metric='r2'
    )
    
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    test_results = selector.evaluate_all(X_test, y_test, metric='r2')
    print("\n" + selector.get_comparison_summary(test_results))
    
    print("\n" + "=" * 70)
    print("DETAILED METRICS (Test Set)")
    print("=" * 70)
    
    detailed_metrics = {}
    for model_type in selector.models:
        preds = test_results['predictions'][model_type]
        mse = np.mean((preds - y_test) ** 2)
        mae = np.mean(np.abs(preds - y_test))
        rmse = np.sqrt(mse)
        r2 = test_results['scores'][model_type]
        
        detailed_metrics[model_type] = {
            'mse': float(mse), 'mae': float(mae),
            'rmse': float(rmse), 'r2': float(r2)
        }
        
        print(f"\n{model_type}:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.4f}")
    
    elapsed = time.time() - start_time
    
    output = {
        'schema_version': '3.1.2',
        'test_type': 'model_comparison',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'data_info': {
            'survivors_file': args.survivors,
            'total_samples': len(X),
            'n_features': len(feature_names),
            'y_range': float(y.max() - y.min())
        },
        'model_comparison': selector.to_agent_metadata(test_results)['model_comparison'],
        'detailed_metrics': detailed_metrics,
        'agent_metadata': {
            'pipeline_step': 5,
            'pipeline_step_name': 'model_comparison_test',
            'run_id': f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'success_criteria_met': test_results['best_score'] > 0.5,
            'suggested_params': {'model_type': test_results['best_model']},
            'reasoning': f"Best model: {test_results['best_model']} R²={test_results['best_score']:.4f}"
        }
    }
    
    output_file = Path(args.output_dir) / 'model_comparison_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best model: {test_results['best_model']}")
    print(f"Best R²:    {test_results['best_score']:.4f}")
    print(f"Elapsed:    {elapsed:.1f}s")
    print(f"Results:    {output_file}")
    print("=" * 70)
    print(f"\n🤖 AI RECOMMENDATION: Use --model-type {test_results['best_model']}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Quick model comparison test')
    parser.add_argument('--survivors', type=str, required=True)
    parser.add_argument('--max-survivors', type=int, default=10000)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='models/reinforcement')
    args = parser.parse_args()
    run_comparison(args)


if __name__ == '__main__':
    main()
