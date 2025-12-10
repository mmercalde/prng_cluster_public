#!/usr/bin/env python3
"""
Dry Test Suite for feature_importance.py
=========================================

Comprehensive tests that validate all module functionality
without requiring the GPU cluster or trained models.

Run: python3 test_feature_importance.py

Author: Distributed PRNG Analysis System
Date: December 9, 2025
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Import the module under test
from feature_importance import (
    FeatureImportanceResult,
    FeatureImportanceExtractor,
    compare_importance,
    get_importance_summary_for_agent,
    ImportanceMethod,
    FeatureCategory,
    TORCH_AVAILABLE
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  ❌ {name}: {error}")
    
    def skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: SKIPPED ({reason})")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.errors:
            print("\nFailures:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_enums(results: TestResults):
    """Test enum definitions."""
    print("\n[1] Testing Enums...")
    
    # ImportanceMethod enum
    try:
        assert ImportanceMethod.PERMUTATION.value == "permutation"
        assert ImportanceMethod.GRADIENT.value == "gradient"
        assert ImportanceMethod.SHAP.value == "shap"
        results.record("ImportanceMethod enum values", True)
    except AssertionError as e:
        results.record("ImportanceMethod enum values", False, str(e))
    
    # FeatureCategory enum
    try:
        assert FeatureCategory.STATISTICAL.value == "statistical"
        assert FeatureCategory.GLOBAL_STATE.value == "global_state"
        results.record("FeatureCategory enum values", True)
    except AssertionError as e:
        results.record("FeatureCategory enum values", False, str(e))


def test_feature_importance_result_creation(results: TestResults):
    """Test FeatureImportanceResult dataclass creation."""
    print("\n[2] Testing FeatureImportanceResult Creation...")
    
    try:
        result = FeatureImportanceResult(
            computation_method="permutation",
            model_version="test_v1.0",
            timestamp="2025-12-09T14:30:00Z",
            total_features=60,
            importance_by_feature={"feature_1": 0.15, "feature_2": 0.10},
            importance_by_category={"statistical_features": 0.72, "global_state_features": 0.28},
            top_10_features=[{"name": "feature_1", "importance": 0.15, "category": "statistical", "rank": 1}],
            bottom_10_features=[{"name": "feature_60", "importance": 0.001, "category": "global", "rank": 60}],
            computation_time_seconds=5.5,
            samples_used=1000,
            baseline_metric=0.0234
        )
        results.record("Basic creation", True)
    except Exception as e:
        results.record("Basic creation", False, str(e))
        return
    
    # Test field access
    try:
        assert result.computation_method == "permutation"
        assert result.total_features == 60
        assert result.samples_used == 1000
        results.record("Field access", True)
    except AssertionError as e:
        results.record("Field access", False, str(e))


def test_feature_importance_result_serialization(results: TestResults):
    """Test JSON serialization and deserialization."""
    print("\n[3] Testing Serialization...")
    
    original = FeatureImportanceResult(
        computation_method="gradient",
        model_version="test_v2.0",
        timestamp=datetime.now().isoformat(),
        total_features=60,
        importance_by_feature={f"feature_{i}": np.random.rand() for i in range(60)},
        importance_by_category={"statistical_features": 0.65, "global_state_features": 0.35},
        top_10_features=[{"name": f"feature_{i}", "importance": 0.1-i*0.01, "category": "statistical", "rank": i+1} for i in range(10)],
        bottom_10_features=[{"name": f"feature_{50+i}", "importance": 0.001*i, "category": "global", "rank": 50+i} for i in range(10)],
        computation_time_seconds=2.5,
        samples_used=500
    )
    
    # Test to_dict()
    try:
        d = original.to_dict()
        assert "feature_importance" in d
        assert d["feature_importance"]["computation_method"] == "gradient"
        assert d["feature_importance"]["total_features"] == 60
        results.record("to_dict()", True)
    except Exception as e:
        results.record("to_dict()", False, str(e))
    
    # Test save/load roundtrip
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        original.save(temp_path)
        loaded = FeatureImportanceResult.load(temp_path)
        
        assert loaded.computation_method == original.computation_method
        assert loaded.model_version == original.model_version
        assert loaded.total_features == original.total_features
        assert len(loaded.importance_by_feature) == len(original.importance_by_feature)
        results.record("save/load roundtrip", True)
    except Exception as e:
        results.record("save/load roundtrip", False, str(e))
    finally:
        os.unlink(temp_path)
    
    # Test JSON structure
    try:
        json_str = json.dumps(original.to_dict())
        parsed = json.loads(json_str)
        assert "feature_importance" in parsed
        results.record("JSON serializable", True)
    except Exception as e:
        results.record("JSON serializable", False, str(e))


def test_get_summary(results: TestResults):
    """Test summary generation."""
    print("\n[4] Testing Summary Generation...")
    
    result = FeatureImportanceResult(
        computation_method="permutation",
        model_version="summary_test",
        timestamp="2025-12-09T15:00:00Z",
        total_features=60,
        importance_by_feature={"lane_agreement_8": 0.15, "skip_entropy": 0.12},
        importance_by_category={"statistical_features": 0.75, "global_state_features": 0.25},
        top_10_features=[
            {"name": "lane_agreement_8", "importance": 0.15, "category": "statistical", "rank": 1},
            {"name": "skip_entropy", "importance": 0.12, "category": "statistical", "rank": 2},
            {"name": "temporal_stability_mean", "importance": 0.10, "category": "statistical", "rank": 3},
            {"name": "residue_8_match_rate", "importance": 0.08, "category": "statistical", "rank": 4},
            {"name": "global_feature_1", "importance": 0.07, "category": "global", "rank": 5},
        ],
        bottom_10_features=[{"name": "low_feature", "importance": 0.001, "category": "global", "rank": 60}],
        computation_time_seconds=10.0,
        samples_used=2000
    )
    
    try:
        summary = result.get_summary()
        assert "Feature Importance Analysis" in summary
        assert "permutation" in summary
        assert "summary_test" in summary
        assert "75.0%" in summary  # statistical_features
        assert "lane_agreement_8" in summary
        results.record("Summary content", True)
    except Exception as e:
        results.record("Summary content", False, str(e))
    
    try:
        lines = summary.split('\n')
        assert len(lines) >= 10  # Should have reasonable length
        results.record("Summary format", True)
    except Exception as e:
        results.record("Summary format", False, str(e))


def test_compare_importance(results: TestResults):
    """Test drift comparison between two results."""
    print("\n[5] Testing compare_importance()...")
    
    # Create two results with known differences
    features = {f"feature_{i}": 0.1 for i in range(20)}
    
    result1 = FeatureImportanceResult(
        computation_method="permutation",
        model_version="v1",
        timestamp="2025-12-09T14:00:00Z",
        total_features=20,
        importance_by_feature=features.copy(),
        importance_by_category={"statistical_features": 0.70, "global_state_features": 0.30},
        top_10_features=[],
        bottom_10_features=[]
    )
    
    # Modify features for drift
    features_modified = features.copy()
    features_modified["feature_0"] = 0.5  # Big change
    features_modified["feature_1"] = 0.01  # Big drop
    
    result2 = FeatureImportanceResult(
        computation_method="permutation",
        model_version="v2",
        timestamp="2025-12-09T15:00:00Z",
        total_features=20,
        importance_by_feature=features_modified,
        importance_by_category={"statistical_features": 0.60, "global_state_features": 0.40},
        top_10_features=[],
        bottom_10_features=[]
    )
    
    try:
        comparison = compare_importance(result1, result2, threshold=0.15)
        assert "drift_score" in comparison
        assert "drift_alert" in comparison
        assert "top_gainers" in comparison
        assert "top_losers" in comparison
        results.record("Comparison structure", True)
    except Exception as e:
        results.record("Comparison structure", False, str(e))
        return
    
    # Test drift detection
    try:
        # feature_0 should be a top gainer
        gainers = [g[0] for g in comparison["top_gainers"]]
        assert "feature_0" in gainers, f"Expected feature_0 in gainers, got {gainers}"
        results.record("Top gainers detection", True)
    except AssertionError as e:
        results.record("Top gainers detection", False, str(e))
    
    try:
        # feature_1 should be a top loser
        losers = [l[0] for l in comparison["top_losers"]]
        assert "feature_1" in losers, f"Expected feature_1 in losers, got {losers}"
        results.record("Top losers detection", True)
    except AssertionError as e:
        results.record("Top losers detection", False, str(e))
    
    # Test no drift scenario
    try:
        same_comparison = compare_importance(result1, result1, threshold=0.15)
        assert same_comparison["drift_score"] == 0.0
        assert same_comparison["drift_alert"] == False
        results.record("No drift case", True)
    except Exception as e:
        results.record("No drift case", False, str(e))


def test_agent_summary(results: TestResults):
    """Test agent metadata summary generation."""
    print("\n[6] Testing get_importance_summary_for_agent()...")
    
    result = FeatureImportanceResult(
        computation_method="permutation",
        model_version="agent_test_v1",
        timestamp="2025-12-09T16:00:00Z",
        total_features=60,
        importance_by_feature={"f1": 0.2, "f2": 0.15},
        importance_by_category={"statistical_features": 0.72, "global_state_features": 0.28},
        top_10_features=[
            {"name": "lane_agreement_8", "importance": 0.15, "category": "statistical", "rank": 1},
            {"name": "skip_entropy", "importance": 0.12, "category": "statistical", "rank": 2},
            {"name": "temporal_mean", "importance": 0.10, "category": "statistical", "rank": 3},
            {"name": "residue_8", "importance": 0.08, "category": "statistical", "rank": 4},
            {"name": "global_1", "importance": 0.07, "category": "global", "rank": 5},
        ],
        bottom_10_features=[],
        samples_used=1500
    )
    
    try:
        summary = get_importance_summary_for_agent(result)
        
        assert summary["statistical_weight"] == 0.72
        assert summary["global_weight"] == 0.28
        assert summary["computation_method"] == "permutation"
        assert summary["model_version"] == "agent_test_v1"
        assert summary["samples_analyzed"] == 1500
        assert len(summary["top_features"]) == 5
        assert summary["top_features"][0] == "lane_agreement_8"
        
        results.record("Agent summary structure", True)
    except Exception as e:
        results.record("Agent summary structure", False, str(e))
    
    # Test JSON serializable for agent_metadata injection
    try:
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        assert parsed == summary
        results.record("Agent summary JSON serializable", True)
    except Exception as e:
        results.record("Agent summary JSON serializable", False, str(e))


def test_statistical_features_list(results: TestResults):
    """Test the STATISTICAL_FEATURES constant."""
    print("\n[7] Testing STATISTICAL_FEATURES list...")
    
    try:
        features = FeatureImportanceExtractor.STATISTICAL_FEATURES
        assert len(features) >= 40, f"Expected at least 40 statistical features, got {len(features)}"
        results.record(f"Feature count ({len(features)} features)", True)
    except Exception as e:
        results.record("Feature count", False, str(e))
    
    # Check expected features are present
    expected = [
        "score", "confidence", "exact_matches",
        "residue_8_match_rate", "residue_8_coherence",
        "temporal_stability_mean", "temporal_stability_std",
        "lane_agreement_8", "lane_agreement_125",
        "skip_entropy", "skip_mean", "skip_std",
        "survivor_velocity", "intersection_weight"
    ]
    
    try:
        features = FeatureImportanceExtractor.STATISTICAL_FEATURES
        missing = [f for f in expected if f not in features]
        assert len(missing) == 0, f"Missing expected features: {missing}"
        results.record("Expected features present", True)
    except AssertionError as e:
        results.record("Expected features present", False, str(e))
    
    # Check no duplicates
    try:
        features = FeatureImportanceExtractor.STATISTICAL_FEATURES
        assert len(features) == len(set(features)), "Duplicate features found"
        results.record("No duplicate features", True)
    except AssertionError as e:
        results.record("No duplicate features", False, str(e))


def test_extractor_initialization(results: TestResults):
    """Test FeatureImportanceExtractor initialization (without model)."""
    print("\n[8] Testing Extractor Initialization...")
    
    if not TORCH_AVAILABLE:
        results.skip("Extractor initialization", "PyTorch not available")
        return
    
    try:
        import torch
        import torch.nn as nn
        
        # Create minimal model
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(60, 1)
            def forward(self, x):
                return torch.sigmoid(self.linear(x))
        
        model = MinimalModel()
        feature_names = [f"feature_{i}" for i in range(60)]
        
        extractor = FeatureImportanceExtractor(
            model=model,
            feature_names=feature_names,
            device='cpu'
        )
        
        assert extractor.model is not None
        assert len(extractor.feature_names) == 60
        assert extractor.device == 'cpu'
        
        results.record("Extractor creation", True)
    except Exception as e:
        results.record("Extractor creation", False, str(e))


def test_categorize_importance(results: TestResults):
    """Test importance categorization logic."""
    print("\n[9] Testing Categorization Logic...")
    
    if not TORCH_AVAILABLE:
        results.skip("Categorization logic", "PyTorch not available")
        return
    
    try:
        import torch
        import torch.nn as nn
        
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(60, 1)
            def forward(self, x):
                return torch.sigmoid(self.linear(x))
        
        model = MinimalModel()
        feature_names = list(FeatureImportanceExtractor.STATISTICAL_FEATURES[:46]) + [f"global_{i}" for i in range(14)]
        
        extractor = FeatureImportanceExtractor(model, feature_names, device='cpu')
        
        # Test with known importance values
        importance = {}
        for i, name in enumerate(feature_names[:46]):
            importance[name] = 0.02  # Statistical = 46 * 0.02 = 0.92
        for i, name in enumerate(feature_names[46:]):
            importance[name] = 0.01  # Global = 14 * 0.01 = 0.14
        
        categories = extractor.categorize_importance(importance)
        
        # Statistical should be ~0.868 (0.92 / 1.06)
        assert 0.8 < categories["statistical_features"] < 0.95, f"Got {categories['statistical_features']}"
        assert 0.05 < categories["global_state_features"] < 0.2, f"Got {categories['global_state_features']}"
        
        results.record("Category proportions", True)
    except Exception as e:
        results.record("Category proportions", False, str(e))


def test_full_extraction_mock(results: TestResults):
    """Test full extraction with mock model."""
    print("\n[10] Testing Full Extraction (Mock)...")
    
    if not TORCH_AVAILABLE:
        results.skip("Full extraction", "PyTorch not available")
        return
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a model with known behavior
        class PredictableModel(nn.Module):
            def __init__(self):
                super().__init__()
                # First feature is "important" - directly affects output
                self.weights = nn.Parameter(torch.zeros(60))
                self.weights.data[0] = 1.0  # feature_0 is important
                self.weights.data[1] = 0.5  # feature_1 is somewhat important
            
            def forward(self, x):
                return torch.sigmoid((x * self.weights).sum(dim=1, keepdim=True))
        
        model = PredictableModel()
        feature_names = [f"feature_{i}" for i in range(60)]
        
        extractor = FeatureImportanceExtractor(model, feature_names, device='cpu')
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 60).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        
        # Test gradient method (faster)
        result = extractor.extract(
            X=X,
            y=y,
            method='gradient',
            model_version='mock_test_v1'
        )
        
        assert result.computation_method == 'gradient'
        assert result.total_features == 60
        assert result.samples_used == 100
        assert len(result.importance_by_feature) == 60
        assert len(result.top_10_features) == 10
        assert len(result.bottom_10_features) == 10
        
        results.record("Gradient extraction", True)
        
        # Verify feature_0 is highly ranked (it should be most important)
        top_names = [f["name"] for f in result.top_10_features[:3]]
        if "feature_0" in top_names:
            results.record("Important feature detected", True)
        else:
            results.record("Important feature detected", False, 
                          f"feature_0 not in top 3: {top_names}")
        
    except Exception as e:
        results.record("Full extraction", False, str(e))


def test_edge_cases(results: TestResults):
    """Test edge cases and error handling."""
    print("\n[11] Testing Edge Cases...")
    
    # Empty importance dict
    try:
        result = FeatureImportanceResult(
            computation_method="test",
            model_version="edge_test",
            timestamp="2025-12-09T17:00:00Z",
            total_features=0,
            importance_by_feature={},
            importance_by_category={"statistical_features": 0.5, "global_state_features": 0.5},
            top_10_features=[],
            bottom_10_features=[]
        )
        _ = result.to_dict()
        _ = result.get_summary()
        results.record("Empty features handling", True)
    except Exception as e:
        results.record("Empty features handling", False, str(e))
    
    # Very large importance values
    try:
        large_importance = {f"f{i}": float(i * 1000) for i in range(100)}
        result = FeatureImportanceResult(
            computation_method="test",
            model_version="edge_test",
            timestamp="2025-12-09T17:00:00Z",
            total_features=100,
            importance_by_feature=large_importance,
            importance_by_category={"statistical_features": 0.5, "global_state_features": 0.5},
            top_10_features=[],
            bottom_10_features=[]
        )
        d = result.to_dict()
        json.dumps(d)  # Should be JSON serializable
        results.record("Large values handling", True)
    except Exception as e:
        results.record("Large values handling", False, str(e))
    
    # Negative importance values
    try:
        neg_importance = {f"f{i}": float(-0.1 * i) for i in range(10)}
        neg_importance.update({f"f{10+i}": float(0.1 * i) for i in range(10)})
        
        result1 = FeatureImportanceResult(
            computation_method="test", model_version="v1",
            timestamp="2025-12-09T17:00:00Z", total_features=20,
            importance_by_feature=neg_importance,
            importance_by_category={"statistical_features": 0.5, "global_state_features": 0.5},
            top_10_features=[], bottom_10_features=[]
        )
        result2 = FeatureImportanceResult(
            computation_method="test", model_version="v2",
            timestamp="2025-12-09T17:00:00Z", total_features=20,
            importance_by_feature=neg_importance,
            importance_by_category={"statistical_features": 0.5, "global_state_features": 0.5},
            top_10_features=[], bottom_10_features=[]
        )
        
        comparison = compare_importance(result1, result2)
        assert comparison["drift_score"] == 0.0
        results.record("Negative values handling", True)
    except Exception as e:
        results.record("Negative values handling", False, str(e))


def main():
    """Run all tests."""
    print("=" * 60)
    print("Feature Importance Module - Comprehensive Dry Test")
    print("=" * 60)
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = TestResults()
    
    # Run all test suites
    test_enums(results)
    test_feature_importance_result_creation(results)
    test_feature_importance_result_serialization(results)
    test_get_summary(results)
    test_compare_importance(results)
    test_agent_summary(results)
    test_statistical_features_list(results)
    test_extractor_initialization(results)
    test_categorize_importance(results)
    test_full_extraction_mock(results)
    test_edge_cases(results)
    
    # Print summary
    success = results.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
