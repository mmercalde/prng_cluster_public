#!/usr/bin/env python3
"""
Test that sieve_filter.py works with ALL configurable PRNGs
Now includes: xorshift32/64, pcg32, lcg32, mt19937, java_lcg, minstd, xorshift128
"""
import json
import subprocess
import sys

# Get all available PRNGs
sys.path.insert(0, '.')
from prng_registry import list_available_prngs

available_prngs = list_available_prngs()
print(f"Testing {len(available_prngs)} PRNGs: {available_prngs}")
print("="*70)

results = {}

for prng in available_prngs:
    # Skip hybrid and reverse variants for now - those need special handling
    if 'hybrid' in prng or 'reverse' in prng:
        print(f"⏭️  Skipping {prng} (needs special handling)")
        continue
    
    print(f"\n🔬 Testing {prng}...")
    
    # Create job for this PRNG
    test_job = {
        "job_id": f"test_{prng}",
        "dataset_path": "test_26gpu_large.json",
        "seed_start": 0,
        "seed_end": 1000,  # Small range for quick test
        "window_size": 512,
        "min_match_threshold": 0.5,
        "skip_range": [5, 5],  # Fixed skip for consistency
        "prng_families": [prng],  # Test this specific PRNG
        "sessions": ["midday"]
    }
    
    job_file = f"test_{prng}_job.json"
    with open(job_file, 'w') as f:
        json.dump(test_job, f, indent=2)
    
    # Run sieve
    try:
        result = subprocess.run(
            ['python3', 'sieve_filter.py', '--job-file', job_file, '--gpu-id', '0'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse JSON output
        try:
            output = json.loads(result.stdout)
            
            # Check for success
            if output.get('success') == True:
                # Verify correct PRNG was used
                if prng in str(output.get('prng_families', [])):
                    print(f"   ✅ {prng} - PASS (tested {output['stats']['total_seeds_tested']} seeds)")
                    results[prng] = 'PASS'
                else:
                    print(f"   ⚠️  {prng} - ran but used wrong PRNG")
                    results[prng] = 'WRONG_PRNG'
            else:
                error = output.get('error', 'Unknown error')
                print(f"   ❌ {prng} - FAIL: {error}")
                results[prng] = 'FAIL'
        
        except json.JSONDecodeError:
            # Fallback to stderr checking
            if prng in result.stderr and result.returncode == 0:
                print(f"   ✅ {prng} - PASS (stderr check)")
                results[prng] = 'PASS'
            else:
                print(f"   ❌ {prng} - invalid output")
                print(f"      stdout: {result.stdout[:200]}")
                print(f"      stderr: {result.stderr[:200]}")
                results[prng] = 'INVALID_OUTPUT'
    
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  {prng} - timeout (still running after 30s)")
        results[prng] = 'TIMEOUT'
    
    except Exception as e:
        print(f"   ❌ {prng} - EXCEPTION: {e}")
        results[prng] = 'EXCEPTION'

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

for prng, status in results.items():
    emoji = "✅" if status == "PASS" else "⚠️" if status == "UNCLEAR" else "❌"
    print(f"{emoji} {prng:20} {status}")

passed = sum(1 for s in results.values() if s == 'PASS')
total = len(results)
print(f"\nPassed: {passed}/{total}")

# Expected PRNGs
expected = ['xorshift32', 'xorshift64', 'pcg32', 'lcg32', 'mt19937', 
            'java_lcg', 'minstd', 'xorshift128']
tested_base = [p for p in results.keys() if '_hybrid' not in p and '_reverse' not in p]

print(f"\n📊 Base PRNGs tested: {len(tested_base)}")
print(f"   Expected: {expected}")
print(f"   Tested: {sorted(tested_base)}")

if passed == total and total >= 8:
    print(f"\n🎉 SUCCESS! All {total} base PRNGs working!")
    sys.exit(0)
else:
    print(f"\n⚠️  {total - passed} PRNGs failed")
    sys.exit(1)
