#!/usr/bin/env python3
"""
Zeus Integration Test - Feature Importance with meta_prediction_optimizer.py
============================================================================

This script simulates the integration that will happen in Step 4.
Run this on Zeus where PyTorch and the full environment are available.

Usage on Zeus:
    cd ~/prng_cluster_project
    python3 test_feature_importance_zeus.py

Author: Distributed PRNG Analysis System  
Date: December 9, 2025
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_zeus_integration_test():
    """Full integration test with PyTorch."""
    
    print("=" * 70)
    print("Zeus Integration Test - Feature Importance Module")
    print("=" * 70)
    
    # Step 1: Import checks
    print("\n[1] Checking imports...")
    
    try:
        import torch
        import torch.nn as nn
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    try:
        from feature_importance import (
            FeatureImportanceExtractor,
            FeatureImportanceResult,
            compare_importance,
            get_importance_summary_for_agent
        )
        print("  ✅ feature_importance module imported")
    except ImportError as e:
        print(f"  ❌ feature_importance import failed: {e}")
        return False
    
    try:
        from reinforcement_engine import SurvivorQualityNet, ReinforcementConfig
        print("  ✅ reinforcement_engine module imported")
        USE_REAL_MODEL = True
    except ImportError:
        print("  ⚠️  reinforcement_engine not available, using mock model")
        USE_REAL_MODEL = False
    
    # Step 2: Create or load model
    print("\n[2] Setting up model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if USE_REAL_MODEL:
        model = SurvivorQualityNet(input_size=60, hidden_layers=[128, 64, 32], dropout=0.3)
        print(f"  ✅ Created SurvivorQualityNet (60 -> 128 -> 64 -> 32 -> 1)")
    else:
        class MockSurvivorQualityNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(60, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                return self.network(x)
        
        model = MockSurvivorQualityNet()
        print(f"  ✅ Created MockSurvivorQualityNet (60 -> 128 -> 64 -> 32 -> 1)")
    
    model = model.to(device)
    model.eval()
    print(f"  ✅ Model moved to {device}")
    
    # Step 3: Generate feature names (matching survivor_scorer.py)
    print("\n[3] Setting up feature names...")
    
    feature_names = FeatureImportanceExtractor.STATISTICAL_FEATURES.copy()
    # Add global state features to reach 60
    while len(feature_names) < 60:
        feature_names.append(f"global_state_{len(feature_names) - 46}")
    
    print(f"  ✅ {len(feature_names)} feature names configured")
    print(f"     First 5: {feature_names[:5]}")
    print(f"     Last 5: {feature_names[-5:]}")
    
    # Step 4: Create synthetic test data
    print("\n[4] Generating synthetic test data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Create realistic-looking feature distributions
    X = np.zeros((n_samples, 60), dtype=np.float32)
    
    # Statistical features (normalized scores, ratios, etc.)
    X[:, 0] = np.random.uniform(0, 1, n_samples)  # score
    X[:, 1] = np.random.uniform(0, 1, n_samples)  # confidence
    X[:, 2] = np.random.poisson(5, n_samples)     # exact_matches
    X[:, 3] = np.random.poisson(100, n_samples)   # total_predictions
    X[:, 4] = np.random.randint(0, 10, n_samples) # best_offset
    
    # Residue features
    for i in range(5, 14):
        X[:, i] = np.random.uniform(0, 1, n_samples)
    
    # Temporal features
    for i in range(14, 19):
        X[:, i] = np.random.normal(0.5, 0.2, n_samples)
    
    # Remaining features
    for i in range(19, 60):
        X[:, i] = np.random.randn(n_samples) * 0.5
    
    # Target: quality scores
    y = np.random.uniform(0, 1, n_samples).astype(np.float32)
    
    print(f"  ✅ Generated {n_samples} samples")
    print(f"     X shape: {X.shape}")
    print(f"     y shape: {y.shape}")
    
    # Step 5: Create extractor and run analysis
    print("\n[5] Running feature importance extraction...")
    
    extractor = FeatureImportanceExtractor(
        model=model,
        feature_names=feature_names,
        device=device
    )
    
    # Test gradient method first (faster)
    print("\n  [5a] Gradient saliency method...")
    start_time = datetime.now()
    
    result_gradient = extractor.extract(
        X=X,
        y=y,
        method='gradient',
        model_version='zeus_test_gradient_v1'
    )
    
    gradient_time = (datetime.now() - start_time).total_seconds()
    print(f"  ✅ Gradient extraction completed in {gradient_time:.2f}s")
    
    # Test permutation method (more robust)
    print("\n  [5b] Permutation importance method (n_repeats=3)...")
    start_time = datetime.now()
    
    result_permutation = extractor.extract(
        X=X,
        y=y,
        method='permutation',
        model_version='zeus_test_permutation_v1',
        n_repeats=3  # Fewer repeats for testing
    )
    
    permutation_time = (datetime.now() - start_time).total_seconds()
    print(f"  ✅ Permutation extraction completed in {permutation_time:.2f}s")
    
    # Step 6: Validate results
    print("\n[6] Validating results...")
    
    for name, result in [("gradient", result_gradient), ("permutation", result_permutation)]:
        print(f"\n  [{name}]")
        assert len(result.importance_by_feature) == 60, f"Expected 60 features, got {len(result.importance_by_feature)}"
        print(f"    ✅ 60 feature importances computed")
        
        assert len(result.top_10_features) == 10
        print(f"    ✅ Top 10 features identified")
        
        assert "statistical_features" in result.importance_by_category
        assert "global_state_features" in result.importance_by_category
        print(f"    ✅ Category breakdown: stat={result.importance_by_category['statistical_features']:.1%}, global={result.importance_by_category['global_state_features']:.1%}")
        
        print(f"    ✅ Top 3 features: {[f['name'] for f in result.top_10_features[:3]]}")
    
    # Step 7: Test drift comparison
    print("\n[7] Testing drift comparison...")
    
    drift = compare_importance(result_gradient, result_permutation, threshold=0.15)
    print(f"  ✅ Drift score between methods: {drift['drift_score']:.4f}")
    print(f"  ✅ Drift alert: {drift['drift_alert']}")
    print(f"  ✅ Top gainer: {drift['top_gainers'][0] if drift['top_gainers'] else 'None'}")
    
    # Step 8: Test agent summary
    print("\n[8] Testing agent_metadata integration...")
    
    agent_summary = get_importance_summary_for_agent(result_permutation)
    print(f"  ✅ Agent summary keys: {list(agent_summary.keys())}")
    print(f"  ✅ Statistical weight: {agent_summary['statistical_weight']:.1%}")
    print(f"  ✅ Top features: {agent_summary['top_features'][:3]}")
    
    # Simulate agent_metadata injection
    mock_agent_metadata = {
        "pipeline_step": 4,
        "pipeline_step_name": "ml_meta_optimizer",
        "feature_importance": agent_summary
    }
    
    json_str = json.dumps(mock_agent_metadata, indent=2)
    print(f"  ✅ agent_metadata JSON serializable ({len(json_str)} bytes)")
    
    # Step 9: Save results
    print("\n[9] Saving test results...")
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    result_gradient.save(output_dir / "feature_importance_gradient_test.json")
    result_permutation.save(output_dir / "feature_importance_permutation_test.json")
    
    print(f"  ✅ Saved to {output_dir}/")
    
    # Step 10: Print summary
    print("\n[10] Summary...")
    print(result_permutation.get_summary())
    
    print("\n" + "=" * 70)
    print("✅ ALL ZEUS INTEGRATION TESTS PASSED")
    print("=" * 70)
    print(f"\nReady for Phase 2: Integration with meta_prediction_optimizer.py")
    print(f"Gradient method: {gradient_time:.2f}s for {n_samples} samples")
    print(f"Permutation method: {permutation_time:.2f}s for {n_samples} samples (n_repeats=3)")
    
    return True


if __name__ == "__main__":
    success = run_zeus_integration_test()
    sys.exit(0 if success else 1)
