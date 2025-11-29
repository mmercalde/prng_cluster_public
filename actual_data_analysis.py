import json
from collections import Counter

def analyze_actual_survivor_structure():
    print("=== ANALYZING ACTUAL SURVIVOR DATA STRUCTURE ===")
    
    forward_file = 'results/window_opt_forward_244_139.json'
    
    # First, let's see the exact structure of your survivors
    with open(forward_file, 'r') as f:
        data = json.load(f)
    
    print("Top-level keys:", list(data.keys()))
    
    if 'results' in data and data['results']:
        first_result = data['results'][0]
        print("\nFirst result keys:", list(first_result.keys()))
        
        if 'survivors' in first_result and first_result['survivors']:
            first_survivor = first_result['survivors'][0]
            print("\nFirst survivor structure:")
            print(f"  Type: {type(first_survivor)}")
            print(f"  Keys: {list(first_survivor.keys()) if isinstance(first_survivor, dict) else 'N/A'}")
            print(f"  Value: {first_survivor}")
            
            # Show a few more survivors to see if structure is consistent
            print(f"\nSample of 5 survivors:")
            for i, survivor in enumerate(first_result['survivors'][:5]):
                print(f"  Survivor {i}: {survivor}")

def load_bidirectional_seeds():
    print("\n" + "="*60)
    print("LOADING BIDIRECTIONAL SEEDS FOR PREDICTION TESTING")
    print("="*60)
    
    forward_file = 'results/window_opt_forward_244_139.json'
    reverse_file = 'results/window_opt_reverse_244_139.json'
    
    def extract_seeds(data):
        """Extract seed values from survivor data"""
        seeds = set()
        if 'results' in data:
            for result in data['results']:
                if 'survivors' in result and isinstance(result['survivors'], list):
                    for survivor in result['survivors']:
                        if isinstance(survivor, dict) and 'seed' in survivor:
                            seeds.add(survivor['seed'])
                        elif isinstance(survivor, (int, float)):
                            seeds.add(int(survivor))
        return seeds
    
    # Load seeds from both directions
    with open(forward_file, 'r') as f:
        forward_data = json.load(f)
        forward_seeds = extract_seeds(forward_data)
        print(f"Forward seeds: {len(forward_seeds):,}")
    
    with open(reverse_file, 'r') as f:
        reverse_data = json.load(f)
        reverse_seeds = extract_seeds(reverse_data)
        print(f"Reverse seeds: {len(reverse_seeds):,}")
    
    # Get bidirectional intersection
    bidirectional_seeds = forward_seeds & reverse_seeds
    print(f"Bidirectional seeds: {len(bidirectional_seeds):,}")
    print(f"Sample seeds: {list(bidirectional_seeds)[:10]}")
    
    return list(bidirectional_seeds)

def test_prediction_with_multiple_prngs(seeds):
    print("\n" + "="*60)
    print("TESTING PREDICTION WITH YOUR 44 PRNGS")
    print("="*60)
    
    # Load lottery data
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    print(f"Recent draws: {recent_draws}")
    print(f"Testing {len(seeds):,} bidirectional seeds")
    
    # Since we don't know which of your 44 PRNGs is correct,
    # let's test the seeds with different generation approaches
    
    results = {
        'java_lcg_48bit': {'tested': 0, 'matches': 0, 'matching_seeds': []},
        'java_lcg_32bit': {'tested': 0, 'matches': 0, 'matching_seeds': []},
        'standard_lcg': {'tested': 0, 'matches': 0, 'matching_seeds': []},
        'simple_xorshift': {'tested': 0, 'matches': 0, 'matching_seeds': []}
    }
    
    # Test a subset of seeds
    test_seeds = seeds[:200]  # Test first 200 seeds
    
    for seed in test_seeds:
        # Test 1: Java LCG 48-bit (standard Java Random)
        sequence = []
        current = seed
        for _ in range(50):
            current = (current * 25214903917 + 11) & 0x7fffffffffffffff
            sequence.append(current % 1000)
        if check_sequence_matches(sequence, recent_draws):
            results['java_lcg_48bit']['matches'] += 1
            results['java_lcg_48bit']['matching_seeds'].append(seed)
        results['java_lcg_48bit']['tested'] += 1
        
        # Test 2: Java LCG 32-bit variant
        sequence = []
        current = seed
        for _ in range(50):
            current = (current * 25214903917 + 11) & 0x7fffffff
            sequence.append(current % 1000)
        if check_sequence_matches(sequence, recent_draws):
            results['java_lcg_32bit']['matches'] += 1
            results['java_lcg_32bit']['matching_seeds'].append(seed)
        results['java_lcg_32bit']['tested'] += 1
        
        # Test 3: Standard LCG
        sequence = []
        current = seed
        for _ in range(50):
            current = (current * 1103515245 + 12345) & 0x7fffffff
            sequence.append(current % 1000)
        if check_sequence_matches(sequence, recent_draws):
            results['standard_lcg']['matches'] += 1
            results['standard_lcg']['matching_seeds'].append(seed)
        results['standard_lcg']['tested'] += 1
        
        # Test 4: Simple xorshift
        sequence = []
        current = seed
        for _ in range(50):
            current ^= (current << 13) & 0xffffffff
            current ^= (current >> 17) & 0xffffffff
            current ^= (current << 5) & 0xffffffff
            sequence.append(current % 1000)
        if check_sequence_matches(sequence, recent_draws):
            results['simple_xorshift']['matches'] += 1
            results['simple_xorshift']['matching_seeds'].append(seed)
        results['simple_xorshift']['tested'] += 1
    
    print(f"\n📊 PREDICTION RESULTS:")
    for method, result in results.items():
        accuracy = (result['matches'] / result['tested'] * 100) if result['tested'] > 0 else 0
        print(f"  {method}: {accuracy:.1f}% ({result['matches']}/{result['tested']})")
        if result['matches'] > 0:
            print(f"    Matching seeds: {result['matching_seeds'][:3]}")
    
    return results

def check_sequence_matches(sequence, recent_draws):
    """Check if sequence contains any 3-number pattern from recent draws"""
    for i in range(len(sequence) - 2):
        test_window = sequence[i:i+3]
        for j in range(len(recent_draws) - 2):
            if test_window == recent_draws[j:j+3]:
                return True
    return False

if __name__ == "__main__":
    # First, understand the data structure
    analyze_actual_survivor_structure()
    
    # Then load the bidirectional seeds
    seeds = load_bidirectional_seeds()
    
    if seeds:
        # Test prediction with different PRNG approaches
        results = test_prediction_with_multiple_prngs(seeds)
        
        # Find the best approach
        best_method = None
        best_accuracy = 0
        for method, result in results.items():
            accuracy = (result['matches'] / result['tested'] * 100) if result['tested'] > 0 else 0
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
        
        if best_method and best_accuracy > 0:
            print(f"\n🎯 BEST APPROACH: {best_method} ({best_accuracy:.1f}% accuracy)")
            print("Next: Use this PRNG type with your matching seeds")
        else:
            print(f"\n🔍 No strong matches found with basic PRNGs")
            print("The lottery PRNG might be:")
            print("  1. A custom implementation")
            print("  2. Using different parameters")
            print("  3. More complex than basic LCG/Xorshift")
