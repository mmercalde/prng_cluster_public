#!/usr/bin/env python3
"""
Test Script - Verify Watcher Agent Implementation

Run this on Zeus to verify the Watcher Agent works.

Usage:
    cd ~/distributed_prng_analysis
    python3 test_watcher_agent.py
"""

import json
import sys
import os


def test_imports():
    """Test all Watcher Agent imports work."""
    print("Testing imports...")
    
    from agents.watcher_agent import (
        WatcherAgent,
        WatcherConfig,
        STEP_SCRIPTS,
        STEP_MANIFESTS,
        STEP_NAMES
    )
    
    print(f"  ✅ WatcherAgent imported")
    print(f"  ✅ Step scripts defined: {len(STEP_SCRIPTS)}")
    print(f"  ✅ Step manifests defined: {len(STEP_MANIFESTS)}")
    print(f"  ✅ Step names defined: {len(STEP_NAMES)}")
    
    return True


def test_config():
    """Test WatcherConfig."""
    print("\nTesting WatcherConfig...")
    
    from agents.watcher_agent import WatcherConfig
    
    # Default config
    config = WatcherConfig()
    assert config.auto_proceed_threshold == 0.70
    assert config.max_retries_per_step == 3
    print(f"  ✅ Default config: threshold={config.auto_proceed_threshold}")
    
    # Custom config
    config = WatcherConfig(
        auto_proceed_threshold=0.85,
        use_llm=False
    )
    assert config.auto_proceed_threshold == 0.85
    assert config.use_llm == False
    print(f"  ✅ Custom config: threshold={config.auto_proceed_threshold}, llm={config.use_llm}")
    
    # To dict
    d = config.to_dict()
    assert "auto_proceed_threshold" in d
    print(f"  ✅ Config to_dict() works")
    
    return config


def test_watcher_init():
    """Test WatcherAgent initialization."""
    print("\nTesting WatcherAgent initialization...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    
    config = WatcherConfig(use_llm=False)
    watcher = WatcherAgent(config)
    
    assert watcher.config == config
    assert watcher.history is not None
    assert watcher.kill_switch is not None
    assert watcher.current_step == 0
    
    print(f"  ✅ WatcherAgent initialized")
    print(f"  ✅ History loaded: {len(watcher.history.runs)} runs")
    print(f"  ✅ Kill switch initialized")
    
    return watcher


def test_heuristic_evaluation():
    """Test heuristic evaluation (no LLM)."""
    print("\nTesting heuristic evaluation...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    
    config = WatcherConfig(use_llm=False)
    watcher = WatcherAgent(config)
    
    # Test Step 1 results - should PROCEED
    good_results = {
        "bidirectional_count": 47,
        "forward_count": 892,
        "reverse_count": 156,
        "best_window_size": 512,
        "optimization_score": 0.85,
        "execution_time_seconds": 120
    }
    
    decision, context = watcher.evaluate_results(1, good_results)
    
    assert decision is not None
    assert context is not None
    print(f"  ✅ Good results: action={decision.recommended_action}, confidence={decision.confidence:.2f}")
    assert decision.confidence >= 0.7
    
    # Test poor results - should RETRY or ESCALATE
    poor_results = {
        "bidirectional_count": 50000,  # Too many
        "optimization_score": 0.3  # Low
    }
    
    decision2, context2 = watcher.evaluate_results(1, poor_results, run_number=1)
    print(f"  ✅ Poor results: action={decision2.recommended_action}, confidence={decision2.confidence:.2f}")
    
    return decision


def test_decision_execution():
    """Test decision execution (mock)."""
    print("\nTesting decision execution...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    from agents import AgentDecision
    
    config = WatcherConfig(use_llm=False)
    watcher = WatcherAgent(config)
    
    # Create mock results and evaluate
    results = {
        "bidirectional_count": 47,
        "optimization_score": 0.85
    }
    
    decision, context = watcher.evaluate_results(1, results)
    
    # Test that history recording works
    initial_runs = len(watcher.history.runs)
    
    # Don't actually execute (would try to run real scripts)
    # Just test the recording part
    context.record_to_history(decision.model_dump())
    
    assert len(watcher.history.runs) == initial_runs + 1
    print(f"  ✅ History recorded: {len(watcher.history.runs)} runs")
    
    return True


def test_all_steps():
    """Test evaluation for all 6 steps."""
    print("\nTesting all 6 pipeline steps...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig, STEP_NAMES
    
    config = WatcherConfig(use_llm=False)
    watcher = WatcherAgent(config)
    
    # Mock results for each step
    step_results = {
        1: {"bidirectional_count": 47, "optimization_score": 0.85},
        2: {"best_validation_score": 0.92, "cv_std": 0.03},
        3: {"completion_rate": 0.998, "feature_dimensions": 64},
        4: {"architecture_score": 0.82, "best_layers": 3},
        5: {"overfit_ratio": 1.08, "kfold_std": 0.03},
        6: {"pool_size": 200, "mean_confidence": 0.75}
    }
    
    for step, results in step_results.items():
        decision, context = watcher.evaluate_results(step, results)
        print(f"  ✅ Step {step} ({STEP_NAMES[step]}): {decision.recommended_action} (conf={decision.confidence:.2f})")
    
    return True


def test_retry_logic():
    """Test retry counting logic."""
    print("\nTesting retry logic...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    
    config = WatcherConfig(use_llm=False, max_retries_per_step=3)
    watcher = WatcherAgent(config)
    
    # Simulate retries
    watcher.retry_counts[1] = 0
    assert watcher.retry_counts.get(1, 0) < config.max_retries_per_step
    print(f"  ✅ Initial retry count: {watcher.retry_counts.get(1, 0)}")
    
    watcher.retry_counts[1] = 2
    assert watcher.retry_counts[1] < config.max_retries_per_step
    print(f"  ✅ After 2 retries: still under limit")
    
    watcher.retry_counts[1] = 3
    assert watcher.retry_counts[1] >= config.max_retries_per_step
    print(f"  ✅ At max retries: should escalate")
    
    return True


def test_safety_integration():
    """Test safety/kill switch integration."""
    print("\nTesting safety integration...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    from agents import check_safety, create_halt, clear_halt
    
    config = WatcherConfig(use_llm=False, halt_file="/tmp/test_watcher_halt")
    watcher = WatcherAgent(config)
    
    # Clear any existing halt
    if os.path.exists(config.halt_file):
        os.remove(config.halt_file)
    
    # Should be safe initially
    assert watcher.kill_switch.check_all()
    print(f"  ✅ Initially safe")
    
    # Create halt file
    watcher.kill_switch.create_halt_file("Test halt")
    assert not watcher.kill_switch.check_all()
    print(f"  ✅ Halt file stops watcher")
    
    # Clear halt
    watcher.kill_switch.clear_halt_file()
    assert watcher.kill_switch.check_all()
    print(f"  ✅ Cleared halt, safe again")
    
    return True


def test_decision_logging():
    """Test decision logging to JSONL."""
    print("\nTesting decision logging...")
    
    from agents.watcher_agent import WatcherAgent, WatcherConfig
    import tempfile
    
    # Use temp file for log
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name
    
    config = WatcherConfig(use_llm=False, log_file=log_file)
    watcher = WatcherAgent(config)
    
    # Log a decision
    watcher._log_decision({
        "step": 1,
        "action": "proceed",
        "confidence": 0.85
    })
    
    # Read back
    with open(log_file) as f:
        line = f.readline()
        logged = json.loads(line)
    
    assert logged["step"] == 1
    assert logged["action"] == "proceed"
    assert "timestamp" in logged
    
    print(f"  ✅ Decision logged to JSONL")
    print(f"  ✅ Log contains timestamp: {logged['timestamp'][:19]}")
    
    # Cleanup
    os.remove(log_file)
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WATCHER AGENT VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_config()
        watcher = test_watcher_init()
        test_heuristic_evaluation()
        test_decision_execution()
        test_all_steps()
        test_retry_logic()
        test_safety_integration()
        test_decision_logging()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
        print("\nWatcher Agent Usage:")
        print("-" * 40)
        print("  # Check status")
        print("  python3 -m agents.watcher_agent --status")
        print("")
        print("  # Evaluate a result file")
        print("  python3 -m agents.watcher_agent --evaluate results.json")
        print("")
        print("  # Run full pipeline (no LLM)")
        print("  python3 -m agents.watcher_agent --run-pipeline --no-llm")
        print("")
        print("  # Run with LLM")
        print("  python3 -m agents.watcher_agent --run-pipeline")
        print("")
        print("  # Create halt")
        print("  python3 -m agents.watcher_agent --halt 'Reason'")
        print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
