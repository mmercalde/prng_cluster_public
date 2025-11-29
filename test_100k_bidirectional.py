#!/usr/bin/env python3
"""
Quick 100K bidirectional sieve test with new results system
"""

import sys
from window_optimizer_integration_final import run_bidirectional_test
from coordinator import MultiGPUCoordinator

print("="*80)
print("100K BIDIRECTIONAL SIEVE TEST")
print("="*80)
print()

# Initialize coordinator
print("Initializing coordinator...")
coordinator = MultiGPUCoordinator(
    dataset_path='daily3.json',
    num_gpus=1,  # Use 1 GPU for quick test
    results_dir='results'
)

# Test configuration
config = {
    'window_size': 244,
    'offset': 139,
    'sessions': ['evening'],
    'skip_min': 3,
    'skip_max': 29
}

print(f"Configuration:")
print(f"  Window size: {config['window_size']}")
print(f"  Offset: {config['offset']}")
print(f"  Sessions: {config['sessions']}")
print(f"  Skip range: [{config['skip_min']}, {config['skip_max']}]")
print(f"  Seed range: 0 → 100,000")
print()

# Run bidirectional test
print("Running bidirectional sieve...")
print()

try:
    result = run_bidirectional_test(
        coordinator=coordinator,
        config=config,
        dataset_path='daily3.json',
        seed_start=0,
        seed_count=100000,
        prng_base='java_lcg',
        threshold=0.012
    )
    
    print()
    print("="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print()
    print(f"Forward survivors: {result.get('forward_count', 'N/A'):,}")
    print(f"Reverse survivors: {result.get('reverse_count', 'N/A'):,}")
    print(f"Bidirectional survivors: {result.get('bidirectional_count', 'N/A'):,}")
    print()
    print("Check your results:")
    print("  results/summaries/")
    print("  results/csv/")
    print("  results/json/")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*80)
