#!/usr/bin/env python3
"""
Test Script - Verify Sub-Phase 1 Implementation

Run this on Zeus to verify the Pydantic Context Framework works.

Usage:
    cd ~/prng_cluster_project
    python3 test_subphase1.py
"""

import json
import sys

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    from agents.manifest import AgentManifest, AgentAction, ActionType
    from agents.agent_decision import AgentDecision, parse_llm_response, repair_json
    from agents.registry import ParameterSpec, ParamType, get_registry, get_registry_for_agent
    from agents.parameters import ParameterInfo, ParameterContext
    from agents.prompt_builder import build_evaluation_prompt, get_thresholds_for_step
    
    print("  ✅ All imports successful")
    return True


def test_manifest_loading():
    """Test manifest loading and context dict."""
    print("\nTesting AgentManifest...")
    
    from agents.manifest import AgentManifest
    
    # Create a test manifest
    manifest_data = {
        "agent_name": "window_optimizer_agent",
        "pipeline_step": 1,
        "inputs": ["lottery_file"],
        "outputs": ["optimal_window_config.json", "bidirectional_survivors.json"],
        "actions": [{
            "type": "run_script",
            "script": "window_optimizer.py",
            "args_map": {
                "trials": "window_trials",
                "max-seeds": "seed_count"
            }
        }],
        "success_condition": "bidirectional_count >= 1 AND bidirectional_count <= 1000",
        "follow_up_agents": ["scorer_meta_agent"],
        "retry": 2
    }
    
    manifest = AgentManifest.model_validate(manifest_data)
    
    # Test attribute access (not dict)
    assert manifest.agent_name == "window_optimizer_agent"
    assert manifest.pipeline_step == 1
    print(f"  ✅ Manifest loaded: {manifest.agent_name}")
    
    # Test context dict (hybrid JSON approach)
    ctx = manifest.to_context_dict()
    assert "agent_name" in ctx
    assert "adjustable_parameters" in ctx
    print(f"  ✅ Context dict keys: {list(ctx.keys())}")
    
    return manifest


def test_parameter_registry():
    """Test parameter registry lookup."""
    print("\nTesting ParameterRegistry...")
    
    from agents.registry import get_registry, get_registry_for_agent
    
    # Test by script name
    reg = get_registry("window_optimizer.py")
    assert reg is not None
    print(f"  ✅ Registry found for window_optimizer.py")
    
    # Test context dict
    ctx = reg.to_context_dict()
    assert "parameters" in ctx
    print(f"  ✅ Registry has {len(ctx['parameters'])} parameters")
    
    # Test by agent name
    reg2 = get_registry_for_agent("window_optimizer_agent")
    assert reg2 is not None
    print(f"  ✅ Registry lookup by agent name works")
    
    return reg


def test_parameter_context(manifest):
    """Test parameter context building."""
    print("\nTesting ParameterContext...")
    
    from agents.parameters import ParameterContext
    
    # Build context with current values
    current_values = {"trials": 50, "max_seeds": 10000000}
    ctx = ParameterContext.build(manifest, current_values)
    
    # Test context dict
    ctx_dict = ctx.to_context_dict()
    assert "adjustable_parameters" in ctx_dict
    print(f"  ✅ Parameter context has {len(ctx_dict['adjustable_parameters'])} params")
    
    # Test validation
    errors = ctx.validate_adjustments({"trials": 100})
    assert len(errors) == 0
    print(f"  ✅ Valid adjustment passes")
    
    errors = ctx.validate_adjustments({"trials": 99999})  # Over max
    assert len(errors) > 0
    print(f"  ✅ Invalid adjustment caught: {errors[0]}")
    
    return ctx


def test_agent_decision():
    """Test agent decision parsing."""
    print("\nTesting AgentDecision...")
    
    from agents.agent_decision import AgentDecision, parse_llm_response
    
    # Test valid response
    valid_response = json.dumps({
        "success_condition_met": True,
        "confidence": 0.85,
        "reasoning": "Bidirectional count of 47 is within range",
        "recommended_action": "proceed",
        "suggested_param_adjustments": {},
        "warnings": []
    })
    
    decision = parse_llm_response(valid_response)
    assert decision.success_condition_met == True
    assert decision.recommended_action == "proceed"
    print(f"  ✅ Valid response parsed: action={decision.recommended_action}")
    
    # Test malformed response (needs repair)
    malformed = """```json
    {
        'success_condition_met': True,
        'confidence': 0.9,
        'reasoning': 'Good results',
        'recommended_action': 'continue',  // variant of proceed
        'warnings': [],
    }
    ```"""
    
    decision2 = parse_llm_response(malformed)
    assert decision2.recommended_action == "proceed"  # "continue" maps to "proceed"
    print(f"  ✅ Malformed response repaired: action={decision2.recommended_action}")
    
    # Test output schema
    schema = AgentDecision.output_schema()
    assert "success_condition_met" in schema
    print(f"  ✅ Output schema has {len(schema)} fields")
    
    return decision


def test_prompt_builder(manifest, param_ctx):
    """Test prompt building."""
    print("\nTesting PromptBuilder...")
    
    from agents.prompt_builder import build_evaluation_prompt, get_thresholds_for_step
    
    results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "reverse_count": 156,
        "best_window_size": 512
    }
    
    prompt = build_evaluation_prompt(
        manifest_context=manifest.to_context_dict(),
        parameter_context=param_ctx.to_context_dict(),
        results=results,
        thresholds=get_thresholds_for_step(1)
    )
    
    # Verify structure
    assert "MANIFEST:" in prompt
    assert "RESULTS:" in prompt
    assert "TASK:" in prompt
    assert "OUTPUT FORMAT:" in prompt
    print(f"  ✅ Prompt generated: {len(prompt)} chars")
    
    # Check it's not prose-heavy
    assert "═══" not in prompt  # No decorative lines
    assert "AGENT MANIFEST CONTRACT" not in prompt  # No prose headers
    print(f"  ✅ Prompt is clean JSON (no prose decorators)")
    
    return prompt


def main():
    """Run all tests."""
    print("=" * 60)
    print("SUB-PHASE 1 VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        manifest = test_manifest_loading()
        test_parameter_registry()
        param_ctx = test_parameter_context(manifest)
        test_agent_decision()
        prompt = test_prompt_builder(manifest, param_ctx)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
        print("\nSample prompt (first 500 chars):")
        print("-" * 40)
        print(prompt[:500])
        print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
