#!/usr/bin/env python3
"""
Test Script - Verify Sub-Phase 4 Implementation

Run this on Zeus to verify full context integration works.

Usage:
    cd ~/distributed_prng_analysis
    python3 test_subphase4.py
"""

import json
import sys


def test_imports():
    """Test all Sub-Phase 4 imports work."""
    print("Testing imports...")
    
    # Sub-Phase 4 imports
    from agents import (
        FullAgentContext,
        build_full_context,
        get_doctrine,
        get_doctrine_summary,
        validate_decision_against_doctrine
    )
    
    # Verify main entry point
    from agents import FullAgentContext as FAC
    assert FAC is not None
    
    print("  ✅ All imports successful")
    return True


def test_doctrine():
    """Test doctrine module."""
    print("\nTesting Doctrine...")
    
    from agents.doctrine import get_doctrine, get_doctrine_summary, validate_decision_against_doctrine
    
    # Test full doctrine
    doctrine = get_doctrine()
    assert "version" in doctrine
    assert "decision_framework" in doctrine
    assert "confidence_calibration" in doctrine
    print(f"  ✅ Doctrine version: {doctrine['version']}")
    print(f"  ✅ Decision framework has {len(doctrine['decision_framework'])} actions")
    
    # Test summary
    summary = get_doctrine_summary()
    assert "PROCEED" in summary
    assert "RETRY" in summary
    assert "ESCALATE" in summary
    print(f"  ✅ Doctrine summary: {len(summary)} chars")
    
    # Test validation - valid decision
    valid_decision = {
        "success_condition_met": True,
        "confidence": 0.85,
        "reasoning": "All metrics good",
        "recommended_action": "proceed"
    }
    is_valid, violations = validate_decision_against_doctrine(valid_decision, {})
    assert is_valid
    print(f"  ✅ Valid decision passes validation")
    
    # Test validation - invalid decision (proceed without success)
    invalid_decision = {
        "success_condition_met": False,
        "confidence": 0.85,
        "reasoning": "Trying to proceed anyway",
        "recommended_action": "proceed"
    }
    is_valid, violations = validate_decision_against_doctrine(invalid_decision, {})
    assert not is_valid
    assert len(violations) > 0
    print(f"  ✅ Invalid decision caught: {violations[0]}")
    
    return doctrine


def test_full_context_build():
    """Test building full context."""
    print("\nTesting FullAgentContext.build()...")
    
    from agents import FullAgentContext
    
    # Mock results for Step 1
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "reverse_count": 156,
        "best_window_size": 512,
        "best_skip": 3,
        "optimization_score": 0.85,
        "execution_time_seconds": 342.5,
        "config": {
            "trials": 50,
            "window_size": 512
        }
    }
    
    # Build context
    ctx = FullAgentContext.build(
        step=1,
        results=results,
        run_number=1,
        detect_hardware=True
    )
    
    assert ctx.step == 1
    assert ctx.run_number == 1
    assert ctx.agent_context is not None
    assert ctx.safety is not None
    assert ctx.pipeline is not None
    
    print(f"  ✅ Context built for step {ctx.step}")
    print(f"  ✅ Run ID: {ctx.run_id}")
    print(f"  ✅ Safety check: {'SAFE' if ctx.is_safe() else 'UNSAFE'}")
    
    return ctx


def test_context_dict():
    """Test context dict generation."""
    print("\nTesting to_context_dict()...")
    
    from agents import FullAgentContext
    
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "optimization_score": 0.85
    }
    
    ctx = FullAgentContext.build(step=1, results=results)
    ctx_dict = ctx.to_context_dict()
    
    # Verify structure
    assert "meta" in ctx_dict
    assert "doctrine" in ctx_dict
    assert "results" in ctx_dict
    assert "evaluation" in ctx_dict
    assert "safety" in ctx_dict
    assert "pipeline" in ctx_dict
    
    print(f"  ✅ Context dict has {len(ctx_dict)} top-level keys")
    print(f"  ✅ Meta: run_id={ctx_dict['meta']['run_id']}")
    print(f"  ✅ Doctrine version: {ctx_dict['doctrine']['version']}")
    
    return ctx_dict


def test_llm_prompt():
    """Test LLM prompt generation."""
    print("\nTesting to_llm_prompt()...")
    
    from agents import FullAgentContext
    
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "optimization_score": 0.85
    }
    
    ctx = FullAgentContext.build(step=1, results=results)
    prompt = ctx.to_llm_prompt()
    
    # Verify prompt structure
    assert "DOCTRINE:" in prompt
    assert "CONTEXT:" in prompt
    assert "TASK:" in prompt
    assert "OUTPUT FORMAT:" in prompt
    assert "success_condition_met" in prompt
    
    print(f"  ✅ Prompt generated: {len(prompt)} chars")
    print(f"  ✅ Contains all required sections")
    
    # Show first part of prompt
    print(f"\n  First 500 chars of prompt:")
    print("  " + "-" * 40)
    for line in prompt[:500].split('\n'):
        print(f"  {line}")
    print("  ...")
    print("  " + "-" * 40)
    
    return prompt


def test_evaluation_summary():
    """Test evaluation summary."""
    print("\nTesting get_evaluation_summary()...")
    
    from agents import FullAgentContext
    
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "optimization_score": 0.85
    }
    
    ctx = FullAgentContext.build(step=1, results=results)
    summary = ctx.get_evaluation_summary()
    
    assert "step" in summary
    assert "success" in summary
    assert "confidence" in summary
    assert "interpretation" in summary
    
    print(f"  ✅ Step: {summary['step']}")
    print(f"  ✅ Success: {summary['success']}")
    print(f"  ✅ Confidence: {summary['confidence']}")
    print(f"  ✅ Interpretation: {summary['interpretation'][:60]}...")
    
    return summary


def test_history_recording():
    """Test recording to history."""
    print("\nTesting record_to_history()...")
    
    from agents import FullAgentContext, AnalysisHistory
    
    results = {
        "bidirectional_count": 47,
        "execution_time_seconds": 120
    }
    
    history = AnalysisHistory()
    ctx = FullAgentContext.build(step=1, results=results, history=history)
    
    # Mock decision
    decision = {
        "success_condition_met": True,
        "confidence": 0.85,
        "reasoning": "Good results",
        "recommended_action": "proceed"
    }
    
    # Record to history
    ctx.record_to_history(decision)
    
    assert len(history.runs) == 1
    assert history.runs[0].success == True
    assert history.runs[0].confidence == 0.85
    
    print(f"  ✅ Recorded run to history")
    print(f"  ✅ History now has {len(history.runs)} runs")
    
    return history


def test_convenience_function():
    """Test build_full_context convenience function."""
    print("\nTesting build_full_context()...")
    
    from agents import build_full_context
    
    results = {"bidirectional_count": 100}
    
    ctx = build_full_context(
        step=1,
        results=results,
        run_number=5
    )
    
    assert ctx.step == 1
    assert ctx.run_number == 5
    
    print(f"  ✅ Convenience function works")
    print(f"  ✅ Built context for step {ctx.step}, run {ctx.run_number}")
    
    return ctx


def test_all_steps():
    """Test building context for all 6 steps."""
    print("\nTesting all 6 pipeline steps...")
    
    from agents import build_full_context
    
    step_results = {
        1: {"bidirectional_count": 47, "optimization_score": 0.85},
        2: {"best_validation_score": 0.92, "cv_std": 0.03},
        3: {"completion_rate": 0.998, "feature_dimensions": 64},
        4: {"architecture_score": 0.82, "best_layers": 3},
        5: {"overfit_ratio": 1.08, "kfold_std": 0.03},
        6: {"pool_size": 200, "mean_confidence": 0.75}
    }
    
    for step, results in step_results.items():
        ctx = build_full_context(step=step, results=results)
        summary = ctx.get_evaluation_summary()
        print(f"  ✅ Step {step}: success={summary['success']}, confidence={summary['confidence']}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SUB-PHASE 4 VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_doctrine()
        ctx = test_full_context_build()
        ctx_dict = test_context_dict()
        prompt = test_llm_prompt()
        summary = test_evaluation_summary()
        history = test_history_recording()
        test_convenience_function()
        test_all_steps()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
        print("\nFramework Summary:")
        print("-" * 40)
        print(f"  Version: 3.2.0")
        print(f"  Modules: manifest, parameters, registry, history,")
        print(f"           runtime, safety, pipeline, contexts, doctrine")
        print(f"  Entry Point: FullAgentContext.build() or build_full_context()")
        print(f"  Output: to_llm_prompt() for LLM, to_context_dict() for JSON")
        print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
