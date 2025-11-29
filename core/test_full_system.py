#!/usr/bin/env python3
"""
Test the complete results system with sample data
"""

import sys
sys.path.insert(0, '..')

from core.results_manager import ResultsManager
from datetime import datetime

# Create sample analysis results
sample_data = {
    'run_metadata': {
        'timestamp_start': datetime.now().isoformat(),
        'execution_time_seconds': 3600.5
    },
    'analysis_parameters': {
        'prng_type': 'java_lcg',
        'seed_start': 0,
        'seed_end': 1000000000,
        'total_seeds_tested': 1000000000,
        'window_size': 244,
        'window_offset': 139,
        'skip_min': 3,
        'skip_max': 29,
        'threshold': 0.012,
        'dataset_file': 'daily3.json',
        'sessions_analyzed': ['evening']
    },
    'results_summary': {
        'total_survivors': 278,
        'survival_rate': 0.000000278,
        'forward_survivors': 52341,
        'reverse_survivors': 51892,
        'bidirectional_survivors': 278,
        'best_match_count': 10,
        'best_match_rate': 0.041,
        'best_seed': 47774590,
        'analysis_complete': True
    },
    'performance_metrics': {
        'seeds_per_second': 277777.8,
        'gpu_utilization_avg': 87.3,
        'memory_peak_gb': 14.2,
        'temperature_max_celsius': 78.5,
        'batches_processed': 100,
        'batches_failed': 0
    },
    'survivors': [
        {
            'seed': 47774590,
            'matches': 10,
            'total_draws': 244,
            'match_rate': 0.041,
            'skip_length': 7,
            'direction': 'bidirectional'
        },
        {
            'seed': 83888441,
            'matches': 9,
            'total_draws': 244,
            'match_rate': 0.037,
            'skip_length': 14,
            'direction': 'bidirectional'
        },
        {
            'seed': 12345678,
            'matches': 8,
            'total_draws': 244,
            'match_rate': 0.033,
            'skip_length': 5,
            'direction': 'bidirectional'
        }
    ]
}

print("=" * 80)
print("TESTING RESULTS SYSTEM WITH SAMPLE DATA")
print("=" * 80)
print()

# Initialize ResultsManager
rm = ResultsManager(schema_dir='../schemas', results_dir='../results')

print()
print("Saving test results...")
print()

# Save results
try:
    output_paths = rm.save_results(
        analysis_type='bidirectional_sieve',
        run_id='test_1B_sample',
        data=sample_data
    )
    
    print()
    print("=" * 80)
    print("SUCCESS! All output files created:")
    print("=" * 80)
    
    for format_type, path in output_paths.items():
        print(f"\n{format_type.upper()}:")
        print(f"  Location: {path}")
        print(f"  Size: {path.stat().st_size:,} bytes")
        
        if format_type == 'summary':
            print("\n  Preview (first 20 lines):")
            with open(path, 'r') as f:
                lines = f.readlines()[:20]
                for line in lines:
                    print(f"    {line.rstrip()}")
    
    print()
    print("=" * 80)
    print("✅ COMPLETE SYSTEM TEST PASSED!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Check the output files in results/")
    print("  2. Integrate into window_optimizer_integration_final.py")
    print("  3. Test with real 1B seed run")
    
except Exception as e:
    print()
    print("=" * 80)
    print(f"❌ TEST FAILED: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)

