#!/usr/bin/env python3
"""
Analyze existing test data and create PROPER test files with skip=5
"""
import json
from prng_registry import get_cpu_reference

print("=" * 70)
print("ANALYZING EXISTING TEST DATA")
print("=" * 70)

# Check what's actually in the test files
for prng_name, test_file in [
    ('xoshiro256pp_reverse', 'test_multi_prng_xoshiro256pp.json'),
    ('sfc64_reverse', 'test_multi_prng_sfc64.json')
]:
    print(f"\n{prng_name}: {test_file}")
    
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        print(f"  Draws in file: {len(data)}")
        print(f"  First 10 draws: {[d['draw'] for d in data[:10]]}")
        
        # Try to reverse-engineer what seed/skip was used
        cpu_ref = get_cpu_reference(prng_name)
        
        # Try common seeds
        found = False
        for test_seed in [42, 1234, 12345, 0, 1, 100, 999]:
            for test_skip in range(0, 50):
                outputs = cpu_ref(test_seed, len(data), skip=test_skip, offset=0)
                generated_draws = [x % 1000 for x in outputs]
                file_draws = [d['draw'] for d in data]
                
                if generated_draws[:10] == file_draws[:10]:
                    print(f"  ✅ FOUND: seed={test_seed}, skip={test_skip}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"  ❌ Could not determine seed/skip in range tested")
    
    except FileNotFoundError:
        print(f"  ❌ File not found!")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 70)
print("CREATING NEW PROPER TEST DATA (seed=1234, skip=5)")
print("=" * 70)

# Create proper test data with known seed and skip=5
for prng_name, output_file in [
    ('xoshiro256pp_reverse', 'test_multi_prng_xoshiro256pp.json'),
    ('sfc64_reverse', 'test_multi_prng_sfc64.json')
]:
    print(f"\nCreating: {output_file}")
    
    cpu_ref = get_cpu_reference(prng_name)
    
    # Generate with seed=1234, skip=5, 512 draws
    SEED = 1234
    SKIP = 5
    NUM_DRAWS = 512
    
    outputs = cpu_ref(SEED, NUM_DRAWS, skip=SKIP, offset=0)
    
    # Create JSON data
    test_data = []
    for i, output in enumerate(outputs):
        test_data.append({
            'draw': int(output % 1000),
            'session': 'midday',
            'timestamp': 5000000 + i
        })
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"  ✅ Created {output_file}")
    print(f"     Seed: {SEED}")
    print(f"     Skip: {SKIP}")
    print(f"     Draws: {NUM_DRAWS}")
    print(f"     First 5: {[d['draw'] for d in test_data[:5]]}")
    
    # Verify it works
    print(f"  🔍 Verifying...")
    test_outputs = cpu_ref(SEED, 5, skip=SKIP, offset=0)
    test_draws = [x % 1000 for x in test_outputs]
    file_draws = [d['draw'] for d in test_data[:5]]
    if test_draws == file_draws:
        print(f"  ✅ VERIFIED: First 5 match!")
    else:
        print(f"  ❌ MISMATCH: {test_draws} vs {file_draws}")

print("\n" + "=" * 70)
print("DONE! Test data files updated.")
print("Now rerun: bash test_last_four_prngs.sh")
print("=" * 70)
