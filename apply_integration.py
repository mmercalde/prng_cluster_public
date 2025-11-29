#!/usr/bin/env python3
"""
Apply integration changes to the 3 main analysis files.
Creates new versions with _INTEGRATED suffix for review.
"""

import sys
from pathlib import Path

# Integration code to add
SIEVE_INTEGRATION = '''
    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_forward_sieve_results
        save_forward_sieve_results(
            survivors=result.get('survivors', []),
            config={
                'prng_type': result.get('prng_families', ['unknown'])[0] if result.get('prng_families') else 'unknown',
                'seed_start': job.get('seed_start', 0),
                'seed_end': job.get('seed_end', 0),
                'total_seeds': result.get('stats', {}).get('total_candidates', 0),
                'window_size': result.get('window_size', 0),
                'offset': 0,
                'skip_min': result.get('skip_range', [0, 0])[0],
                'skip_max': result.get('skip_range', [0, 0])[1],
                'threshold': job.get('min_match_threshold', 0),
                'dataset': job.get('dataset_path', 'unknown'),
                'sessions': job.get('sessions', [])
            },
            execution_time=result.get('stats', {}).get('duration_ms', 0) / 1000.0
        )
    except Exception as e:
        print(f"Note: New results format unavailable: {e}")
'''

REVERSE_INTEGRATION = '''
    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_reverse_sieve_results
        save_reverse_sieve_results(
            survivors=result.get('survivors', []),
            config={
                'prng_type': result.get('prng_families', ['unknown'])[0] if result.get('prng_families') else 'unknown',
                'seed_start': 0,
                'seed_end': result.get('stats', {}).get('total_candidates', 0),
                'total_seeds': result.get('candidates_tested', 0),
                'window_size': result.get('window_size', 0),
                'offset': 0,
                'skip_min': 0,
                'skip_max': 0,
                'threshold': job.get('min_match_threshold', 0),
                'dataset': job.get('dataset_path', 'unknown'),
                'sessions': job.get('sessions', [])
            },
            execution_time=result.get('stats', {}).get('duration_ms', 0) / 1000.0
        )
    except Exception as e:
        print(f"Note: New results format unavailable: {e}")
'''

WINDOW_OPT_INTEGRATION = '''
    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_bidirectional_sieve_results
        save_bidirectional_sieve_results(
            forward_survivors=[],  # Not available in this context
            reverse_survivors=[],  # Not available in this context
            intersection=[],  # Would need to pass bidirectional list here
            config={
                'prng_type': prng_base,
                'seed_start': seed_start,
                'seed_end': seed_start + seed_count,
                'total_seeds': seed_count,
                'window_size': best.get('window_size', 0),
                'offset': best.get('offset', 0),
                'skip_min': best.get('skip_min', 0),
                'skip_max': best.get('skip_max', 0),
                'threshold': 0.01,
                'dataset': dataset_path,
                'sessions': best.get('sessions', [])
            },
            run_id=f"window_opt_{prng_base}_{strategy_name}"
        )
    except Exception as e:
        print(f"Note: New results format unavailable: {e}")
'''

def modify_sieve_filter():
    """Modify sieve_filter.py"""
    print("📝 Modifying sieve_filter.py...")
    
    with open('sieve_filter.py', 'r') as f:
        content = f.read()
    
    # Find the spot: right after json.dump(result, f, indent=2)
    # and before print(json.dumps(result))
    target = '    print(json.dumps(result))'
    
    if target in content:
        modified = content.replace(target, SIEVE_INTEGRATION + '\n' + target)
        
        with open('sieve_filter_INTEGRATED.py', 'w') as f:
            f.write(modified)
        
        print("✅ Created: sieve_filter_INTEGRATED.py")
        return True
    else:
        print("❌ Could not find insertion point in sieve_filter.py")
        return False

def modify_reverse_sieve():
    """Modify reverse_sieve_filter.py"""
    print("📝 Modifying reverse_sieve_filter.py...")
    
    with open('reverse_sieve_filter.py', 'r') as f:
        content = f.read()
    
    # Find the spot: right after json.dump(result, f, indent=2)
    target = '    print(json.dumps(result))'
    
    if target in content:
        modified = content.replace(target, REVERSE_INTEGRATION + '\n' + target)
        
        with open('reverse_sieve_filter_INTEGRATED.py', 'w') as f:
            f.write(modified)
        
        print("✅ Created: reverse_sieve_filter_INTEGRATED.py")
        return True
    else:
        print("❌ Could not find insertion point in reverse_sieve_filter.py")
        return False

def modify_window_optimizer():
    """Modify window_optimizer_integration_final.py"""
    print("📝 Modifying window_optimizer_integration_final.py...")
    
    with open('window_optimizer_integration_final.py', 'r') as f:
        content = f.read()
    
    # Find the spot: at the end of optimize_window function, before return results
    target = '        return results'
    
    if target in content:
        modified = content.replace(target, WINDOW_OPT_INTEGRATION + '\n' + target)
        
        with open('window_optimizer_integration_final_INTEGRATED.py', 'w') as f:
            f.write(modified)
        
        print("✅ Created: window_optimizer_integration_final_INTEGRATED.py")
        return True
    else:
        print("❌ Could not find insertion point in window_optimizer_integration_final.py")
        return False

def main():
    print("="*80)
    print("INTEGRATION APPLICATION SCRIPT")
    print("="*80)
    print()
    
    print("This will create 3 new files with _INTEGRATED suffix:")
    print("  - sieve_filter_INTEGRATED.py")
    print("  - reverse_sieve_filter_INTEGRATED.py")
    print("  - window_optimizer_integration_final_INTEGRATED.py")
    print()
    print("Original files will NOT be modified.")
    print("You can review the _INTEGRATED files before using them.")
    print()
    
    # Check files exist
    required = ['sieve_filter.py', 'reverse_sieve_filter.py', 'window_optimizer_integration_final.py']
    for fname in required:
        if not Path(fname).exists():
            print(f"❌ Error: {fname} not found!")
            return 1
    
    # Apply modifications
    results = []
    results.append(modify_sieve_filter())
    results.append(modify_reverse_sieve())
    results.append(modify_window_optimizer())
    
    print()
    print("="*80)
    if all(results):
        print("✅ SUCCESS! All 3 files created!")
        print()
        print("Next steps:")
        print("  1. Review the _INTEGRATED files")
        print("  2. Test with: python3 sieve_filter_INTEGRATED.py --job-file test.json --gpu-id 0")
        print("  3. If satisfied, replace originals:")
        print("     cp sieve_filter_INTEGRATED.py sieve_filter.py")
        print("     cp reverse_sieve_filter_INTEGRATED.py reverse_sieve_filter.py")
        print("     cp window_optimizer_integration_final_INTEGRATED.py window_optimizer_integration_final.py")
    else:
        print("❌ Some modifications failed - check messages above")
        return 1
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
