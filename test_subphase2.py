#!/usr/bin/env python3
"""
Test Script - Verify Sub-Phase 2 Implementation

Run this on Zeus to verify Team Beta infrastructure works.

Usage:
    cd ~/distributed_prng_analysis
    python3 test_subphase2.py
"""

import json
import sys
from datetime import datetime, timedelta


def test_imports():
    """Test all Sub-Phase 2 imports work."""
    print("Testing imports...")
    
    # Sub-Phase 1 (should still work)
    from agents.manifest import AgentManifest
    from agents.agent_decision import AgentDecision, parse_llm_response
    from agents.parameters import ParameterContext
    from agents.prompt_builder import build_evaluation_prompt
    
    # Sub-Phase 2 (new)
    from agents.history import AnalysisHistory, RunRecord, MetricTrend, TrendDirection
    from agents.runtime import RuntimeContext, GPUInfo, detect_runtime, get_gpu_summary
    from agents.safety import KillSwitch, SafetyLevel, check_safety
    from agents.pipeline import PipelineStepContext, PipelineStep, get_step_info
    
    print("  ✅ All imports successful")
    return True


def test_analysis_history():
    """Test analysis history and trend detection."""
    print("\nTesting AnalysisHistory...")
    
    from agents.history import AnalysisHistory, RunRecord, TrendDirection
    
    history = AnalysisHistory()
    
    # Add some mock runs with improving trend
    for i in range(5):
        record = RunRecord(
            run_id=f"test_run_{i}",
            run_number=i + 1,
            timestamp=datetime.utcnow() - timedelta(hours=5-i),
            agent_name="window_optimizer_agent",
            pipeline_step=1,
            success=True,
            confidence=0.7 + (i * 0.05),  # Improving confidence
            execution_time_seconds=300 - (i * 20),  # Improving time
            metrics={
                "bidirectional_count": 100 - (i * 10),  # Decreasing survivors (good)
                "forward_count": 500 - (i * 50)
            },
            action_taken="proceed"
        )
        history.add_run(record)
    
    print(f"  ✅ Added {len(history.runs)} test runs")
    
    # Test trend analysis
    trend = history.analyze_metric("confidence", agent_name="window_optimizer_agent")
    print(f"  ✅ Confidence trend: {trend.direction.value} (slope={trend.slope:.4f})")
    
    # Test success rate
    rate = history.get_success_rate("window_optimizer_agent")
    print(f"  ✅ Success rate: {rate:.0%}")
    
    # Test context dict
    ctx = history.to_context_dict("window_optimizer_agent")
    assert "trends" in ctx
    assert "success_rate" in ctx
    print(f"  ✅ Context dict has {len(ctx)} keys")
    
    return history


def test_runtime_context():
    """Test runtime/GPU detection."""
    print("\nTesting RuntimeContext...")
    
    from agents.runtime import RuntimeContext, detect_runtime, get_gpu_summary
    
    # Detect actual runtime
    runtime = detect_runtime()
    
    print(f"  ✅ Detection method: {runtime.detection_method}")
    print(f"  ✅ Total GPUs: {runtime.total_gpus}")
    print(f"  ✅ CUDA GPUs: {runtime.total_cuda_gpus}")
    print(f"  ✅ ROCm GPUs: {runtime.total_rocm_gpus}")
    print(f"  ✅ Estimated TFLOPS: {runtime.estimated_tflops}")
    
    # Test context dict
    ctx = runtime.to_context_dict()
    assert "cluster_summary" in ctx
    assert "nodes" in ctx
    print(f"  ✅ Context dict generated")
    
    # Test convenience function
    summary = get_gpu_summary()
    print(f"  ✅ GPU summary: {summary}")
    
    return runtime


def test_kill_switch():
    """Test safety kill switch."""
    print("\nTesting KillSwitch...")
    
    from agents.safety import KillSwitch, SafetyLevel
    
    ks = KillSwitch()
    
    # Run all checks
    is_safe = ks.check_all()
    print(f"  ✅ Safety check: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"  ✅ Current level: {ks.current_level.value}")
    
    # Test failure counting
    ks.record_failure()
    ks.record_failure()
    print(f"  ✅ Consecutive failures: {ks.consecutive_failures}")
    
    ks.record_success()
    print(f"  ✅ After success: {ks.consecutive_failures} failures (should be 0)")
    
    # Test context dict
    ctx = ks.to_context_dict()
    assert "safe_to_proceed" in ctx
    assert "current_level" in ctx
    print(f"  ✅ Context dict has {len(ctx)} keys")
    
    return ks


def test_pipeline_context():
    """Test pipeline step context."""
    print("\nTesting PipelineStepContext...")
    
    from agents.pipeline import PipelineStepContext, get_step_info, get_pipeline_overview
    
    # Create context for step 1
    ctx = PipelineStepContext.for_step(1)
    
    print(f"  ✅ Current step: {ctx.current_step}")
    print(f"  ✅ Step name: {ctx.current_step_info.name}")
    print(f"  ✅ Expected inputs: {ctx.current_step_info.required_inputs}")
    print(f"  ✅ Expected outputs: {ctx.current_step_info.expected_outputs}")
    
    # Test input validation
    available = ["lottery_data.json", "some_other_file.txt"]
    valid = ctx.validate_inputs(available)
    print(f"  ✅ Inputs valid: {valid}")
    
    # Test step advancement
    ctx.mark_completed()
    advanced = ctx.advance_to_next()
    print(f"  ✅ Advanced to step {ctx.current_step}: {advanced}")
    
    # Test context dict
    ctx_dict = ctx.to_context_dict()
    assert "current_step" in ctx_dict
    assert "progress" in ctx_dict
    print(f"  ✅ Context dict has {len(ctx_dict)} keys")
    
    # Test pipeline overview
    overview = get_pipeline_overview()
    print(f"  ✅ Pipeline has {len(overview)} steps")
    
    return ctx


def test_integrated_context():
    """Test all contexts together."""
    print("\nTesting Integrated Context...")
    
    from agents.history import AnalysisHistory
    from agents.runtime import detect_runtime
    from agents.safety import KillSwitch
    from agents.pipeline import PipelineStepContext
    
    # Build full context
    history = AnalysisHistory()
    runtime = detect_runtime()
    safety = KillSwitch()
    pipeline = PipelineStepContext.for_step(1)
    
    # Get all context dicts
    full_context = {
        "history": history.to_context_dict(),
        "runtime": runtime.to_context_dict(),
        "safety": safety.to_context_dict(),
        "pipeline": pipeline.to_context_dict()
    }
    
    # Serialize to JSON
    json_str = json.dumps(full_context, indent=2, default=str)
    print(f"  ✅ Full context JSON: {len(json_str)} chars")
    print(f"  ✅ All sub-contexts integrated")
    
    return full_context


def main():
    """Run all tests."""
    print("=" * 60)
    print("SUB-PHASE 2 VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_analysis_history()
        test_runtime_context()
        test_kill_switch()
        test_pipeline_context()
        full_ctx = test_integrated_context()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
        print("\nSample integrated context (summary):")
        print("-" * 40)
        print(f"  History runs: {full_ctx['history'].get('total_runs', 0)}")
        print(f"  Runtime GPUs: {full_ctx['runtime']['cluster_summary']['total_gpus']}")
        print(f"  Safety level: {full_ctx['safety']['current_level']}")
        print(f"  Pipeline step: {full_ctx['pipeline']['current_step']}")
        print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
