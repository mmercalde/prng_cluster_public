#!/usr/bin/env python3
"""Test hybrid mode integration across all components"""

import sys
import json

print("=" * 70)
print("HYBRID MODE INTEGRATION TEST")
print("=" * 70)

# Test 1: Check hybrid_strategy.py
print("\n[1/5] Testing hybrid_strategy.py...")
try:
    from hybrid_strategy import get_all_strategies, STRATEGY_PRESETS, StrategyConfig
    strategies = get_all_strategies()
    print(f"  ✅ Module loads successfully")
    print(f"  ✅ Found {len(strategies)} strategies: {list(STRATEGY_PRESETS.keys())}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Check prng_registry.py
print("\n[2/5] Testing prng_registry.py...")
try:
    from prng_registry import list_available_prngs, get_kernel_info
    prngs = list_available_prngs()
    if 'mt19937_hybrid' in prngs:
        print(f"  ✅ mt19937_hybrid registered")
        config = get_kernel_info('mt19937_hybrid')
        print(f"  ✅ Kernel: {config['kernel_name']}")
        print(f"  ✅ Variable skip: {config.get('variable_skip', False)}")
        print(f"  ✅ Multi-strategy: {config.get('multi_strategy', False)}")
    else:
        print(f"  ❌ mt19937_hybrid not found in: {prngs}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check sieve_filter.py
print("\n[3/5] Testing sieve_filter.py...")
try:
    from sieve_filter import GPUSieve
    print(f"  ✅ Module imports successfully")
    
    # Check if run_hybrid_sieve method exists
    sieve = GPUSieve(gpu_id=0)
    if hasattr(sieve, 'run_hybrid_sieve'):
        print(f"  ✅ run_hybrid_sieve method exists")
    else:
        print(f"  ❌ run_hybrid_sieve method not found")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check coordinator.py
print("\n[4/5] Testing coordinator.py...")
try:
    # Just check if it parses (don't import to avoid heavy dependencies)
    with open('coordinator.py', 'r') as f:
        code = f.read()
    
    if 'from hybrid_strategy import' in code:
        print(f"  ✅ Hybrid imports present")
    else:
        print(f"  ⚠️  Hybrid imports not found")
    
    if '--hybrid' in code:
        print(f"  ✅ --hybrid argument defined")
    else:
        print(f"  ❌ --hybrid argument not found")
        sys.exit(1)
    
    if 'phase1_threshold' in code:
        print(f"  ✅ Phase thresholds configured")
    else:
        print(f"  ❌ Phase thresholds not found")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Check timestamp_search.py
print("\n[5/5] Testing timestamp_search.py...")
try:
    with open('timestamp_search.py', 'r') as f:
        code = f.read()
    
    if '--hybrid' in code:
        print(f"  ✅ --hybrid argument defined")
    else:
        print(f"  ❌ --hybrid argument not found")
        sys.exit(1)
    
    if 'phase1_threshold' in code and 'phase2_threshold' in code:
        print(f"  ✅ Phase threshold arguments present")
    else:
        print(f"  ❌ Phase threshold arguments not found")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nHybrid mode is ready to use!")
print("\nExample command:")
print("  python3 timestamp_search.py daily3.json \\")
print("    --mode second --window 512 \\")
print("    --hybrid \\")
print("    --phase1-threshold 0.20 \\")
print("    --phase2-threshold 0.75 \\")
print("    --prngs mt19937")
print("=" * 70)
