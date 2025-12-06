#!/usr/bin/env python3
"""
Test Script - Verify Sub-Phase 3 Implementation

Run this on Zeus to verify specialized agent contexts work.

Usage:
    cd ~/distributed_prng_analysis
    python3 test_subphase3.py
"""

import json
import sys


def test_imports():
    """Test all Sub-Phase 3 imports work."""
    print("Testing imports...")
    
    # Sub-Phase 3 imports
    from agents.contexts import (
        BaseAgentContext,
        WindowOptimizerContext,
        ScorerMetaContext,
        FullScoringContext,
        MLMetaContext,
        AntiOverfitContext,
        PredictionContext,
        get_context_for_step,
        EvaluationResult
    )
    
    print("  ✅ All imports successful")
    return True


def test_window_optimizer_context():
    """Test Step 1: Window Optimizer context."""
    print("\nTesting WindowOptimizerContext (Step 1)...")
    
    from agents.contexts import WindowOptimizerContext
    
    # Mock results
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "reverse_count": 156,
        "best_window_size": 512,
        "best_skip": 3,
        "optimization_score": 0.85,
        "execution_time_seconds": 342.5
    }
    
    ctx = WindowOptimizerContext(results=results)
    
    # Test key metrics
    metrics = ctx.get_key_metrics()
    assert "bidirectional_count" in metrics
    print(f"  ✅ Key metrics: {len(metrics)} defined")
    
    # Test evaluation
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    
    # Test interpretation
    interp = ctx.interpret_results()
    assert len(interp) > 0
    print(f"  ✅ Interpretation: {interp[:80]}...")
    
    # Test context dict
    ctx_dict = ctx.to_context_dict()
    assert "evaluations" in ctx_dict
    assert "interpretation" in ctx_dict
    print(f"  ✅ Context dict has {len(ctx_dict)} keys")
    
    return ctx


def test_scorer_meta_context():
    """Test Step 2: Scorer Meta context."""
    print("\nTesting ScorerMetaContext (Step 2)...")
    
    from agents.contexts import ScorerMetaContext
    
    results = {
        "best_validation_score": 0.92,
        "convergence_trial": 35,
        "total_trials": 100,
        "cv_std": 0.03,
        "best_threshold": 0.05,
        "best_k_folds": 5
    }
    
    ctx = ScorerMetaContext(results=results)
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    print(f"  ✅ Interpretation: {ctx.interpret_results()[:80]}...")
    
    return ctx


def test_full_scoring_context():
    """Test Step 3: Full Scoring context."""
    print("\nTesting FullScoringContext (Step 3)...")
    
    from agents.contexts import FullScoringContext
    
    results = {
        "completion_rate": 0.998,
        "survivors_scored": 998,
        "survivors_total": 1000,
        "feature_dimensions": 64,
        "mean_score": 0.45,
        "score_std": 0.22,
        "top_candidates": 50
    }
    
    ctx = FullScoringContext(results=results)
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    print(f"  ✅ Interpretation: {ctx.interpret_results()[:80]}...")
    
    return ctx


def test_ml_meta_context():
    """Test Step 4: ML Meta context."""
    print("\nTesting MLMetaContext (Step 4)...")
    
    from agents.contexts import MLMetaContext
    
    results = {
        "architecture_score": 0.82,
        "best_layers": 3,
        "best_neurons": [64, 32, 16],
        "best_dropout": 0.3,
        "best_learning_rate": 0.001,
        "validation_loss": 0.15,
        "model_parameters": 85000
    }
    
    ctx = MLMetaContext(results=results)
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    print(f"  ✅ Interpretation: {ctx.interpret_results()[:80]}...")
    
    return ctx


def test_anti_overfit_context():
    """Test Step 5: Anti-Overfit context."""
    print("\nTesting AntiOverfitContext (Step 5)...")
    
    from agents.contexts import AntiOverfitContext
    
    results = {
        "overfit_ratio": 1.08,
        "train_loss": 0.12,
        "validation_loss": 0.13,
        "kfold_mean": 0.87,
        "kfold_std": 0.03,
        "best_epoch": 45,
        "total_epochs": 100,
        "early_stopped": True,
        "dropout_used": 0.3
    }
    
    ctx = AntiOverfitContext(results=results)
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    print(f"  ✅ Interpretation: {ctx.interpret_results()[:80]}...")
    
    return ctx


def test_prediction_context():
    """Test Step 6: Prediction context."""
    print("\nTesting PredictionContext (Step 6)...")
    
    from agents.contexts import PredictionContext
    
    results = {
        "pool_size": 200,
        "mean_confidence": 0.75,
        "confidence_std": 0.12,
        "min_confidence": 0.55,
        "max_confidence": 0.95,
        "diversity_score": 0.72,
        "coverage_pct": 65,
        "unique_predictions": 198
    }
    
    ctx = PredictionContext(results=results)
    ctx.evaluate_all_metrics()
    success, confidence = ctx.get_overall_success()
    
    print(f"  ✅ Evaluation: success={success}, confidence={confidence}")
    print(f"  ✅ Interpretation: {ctx.interpret_results()[:80]}...")
    
    return ctx


def test_context_factory():
    """Test factory function for creating contexts."""
    print("\nTesting get_context_for_step factory...")
    
    from agents.contexts import get_context_for_step
    
    # Test each step
    for step in range(1, 7):
        mock_results = {"test": f"step_{step}"}
        ctx = get_context_for_step(step, mock_results)
        assert ctx.pipeline_step == step
        print(f"  ✅ Step {step}: {ctx.agent_name}")
    
    return True


def test_retry_suggestions():
    """Test retry suggestion generation."""
    print("\nTesting retry suggestions...")
    
    from agents.contexts import WindowOptimizerContext
    
    # Results that should trigger retry suggestions
    bad_results = {
        "bidirectional_count": 5000,  # Too many
        "forward_count": 50000,
        "reverse_count": 10000,
        "best_window_size": 512,
        "optimization_score": 0.4  # Low score
    }
    
    ctx = WindowOptimizerContext(results=bad_results)
    suggestions = ctx.get_retry_suggestions()
    
    assert len(suggestions) > 0
    print(f"  ✅ Generated {len(suggestions)} retry suggestions:")
    for s in suggestions:
        print(f"     - {s['param']}: {s['suggestion']} ({s['reason'][:50]}...)")
    
    return suggestions


def test_full_context_dict():
    """Test full context dict generation."""
    print("\nTesting full context dict output...")
    
    from agents.contexts import WindowOptimizerContext
    
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "reverse_count": 156,
        "best_window_size": 512,
        "best_skip": 3,
        "optimization_score": 0.85
    }
    
    ctx = WindowOptimizerContext(results=results, run_number=3)
    ctx_dict = ctx.to_context_dict()
    
    # Verify structure
    assert ctx_dict["agent"] == "window_optimizer_agent"
    assert ctx_dict["step"] == 1
    assert ctx_dict["run_number"] == 3
    assert "evaluations" in ctx_dict
    assert "interpretation" in ctx_dict
    assert "thresholds" in ctx_dict
    
    print(f"  ✅ Context dict structure verified")
    
    # Show sample
    json_str = json.dumps(ctx_dict, indent=2, default=str)
    print(f"  ✅ JSON output: {len(json_str)} chars")
    
    return ctx_dict


def main():
    """Run all tests."""
    print("=" * 60)
    print("SUB-PHASE 3 VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_window_optimizer_context()
        test_scorer_meta_context()
        test_full_scoring_context()
        test_ml_meta_context()
        test_anti_overfit_context()
        test_prediction_context()
        test_context_factory()
        test_retry_suggestions()
        ctx_dict = test_full_context_dict()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
        print("\nSample context dict (Window Optimizer):")
        print("-" * 40)
        print(json.dumps(ctx_dict, indent=2, default=str)[:800])
        print("...")
        print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
