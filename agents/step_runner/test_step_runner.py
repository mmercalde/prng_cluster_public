#!/usr/bin/env python3
"""
Step Runner Test - Verify imports and basic functionality.
============================================================

Run this after deploying step_runner to agents/ directory.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from step_runner import (
            load_manifest,
            build_command,
            build_command_for_action,
            execute_step,
            execute_multi_action_step,
            validate_outputs,
            extract_metrics,
            print_result_summary,
            StepManifest,
            ActionConfig,
            ActionResult,
            StepResult,
            RunMode,
            STEP_DISPLAY_NAMES
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_manifest_loading():
    """Test manifest loading for Step 1."""
    print("\nTesting manifest loading...")
    
    try:
        from step_runner import load_manifest, list_available_manifests
        
        # List available manifests
        available = list_available_manifests("agent_manifests")
        print(f"  Available manifests: {list(available.keys())}")
        
        # Load Step 1 manifest
        manifest = load_manifest(1, "agent_manifests")
        print(f"  ✅ Loaded manifest for Step {manifest.pipeline_step}")
        print(f"     Agent: {manifest.agent_name}")
        print(f"     Script: {manifest.script}")
        print(f"     Actions: {manifest.action_count}")
        print(f"     Multi-action: {manifest.is_multi_action}")
        return True
    except Exception as e:
        print(f"  ❌ Manifest loading failed: {e}")
        return False


def test_multi_action_manifest():
    """Test manifest loading for Step 2.5 (multi-action)."""
    print("\nTesting multi-action manifest (Step 2.5)...")
    
    try:
        from step_runner import load_manifest, build_command_for_action, merge_params
        
        manifest = load_manifest(2, "agent_manifests")
        print(f"  ✅ Loaded manifest for Step {manifest.pipeline_step}")
        print(f"     Agent: {manifest.agent_name}")
        print(f"     Actions: {manifest.action_count}")
        print(f"     Multi-action: {manifest.is_multi_action}")
        
        if manifest.is_multi_action:
            for i, action in enumerate(manifest.actions):
                dist_tag = " [DISTRIBUTED]" if action.distributed else ""
                print(f"     [{i+1}] {action.script}{dist_tag} (timeout: {action.timeout_minutes}m)")
            
            # Verify distributed detection
            has_distributed = any(a.distributed for a in manifest.actions)
            print(f"  ✅ Distributed action detected: {has_distributed}")
            
            # Test building command for first action (non-distributed)
            params = merge_params(manifest, {
                'study_name': 'test_study',
                'scorer_trials': 10
            })
            cmd = build_command_for_action(manifest.actions[0], params)
            print(f"  ✅ Built command for action 1: {' '.join(cmd[:4])}...")
            
        return True
    except Exception as e:
        print(f"  ❌ Multi-action manifest test failed: {e}")
        return False


def test_command_building():
    """Test command building."""
    print("\nTesting command building...")
    
    try:
        from step_runner import load_manifest, build_command, merge_params
        
        manifest = load_manifest(1, "agent_manifests")
        
        # Merge params
        runtime_params = {
            "lottery_file": "synthetic_lottery.json",
            "prng_type": "java_lcg",
            "window_trials": 10,
            "search_strategy": "bayesian"
        }
        params = merge_params(manifest, runtime_params)
        
        # Build command
        command = build_command(manifest, params)
        print(f"  ✅ Built command: {' '.join(command)}")
        return True
    except Exception as e:
        print(f"  ❌ Command building failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("STEP RUNNER MODULE TEST v1.1")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Manifest Loading", test_manifest_loading()))
    results.append(("Multi-Action Manifest", test_multi_action_manifest()))
    results.append(("Command Building", test_command_building()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
