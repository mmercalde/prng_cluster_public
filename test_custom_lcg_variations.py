import json
import numpy as np

def test_custom_lcg_variations():
    print("=== TESTING CUSTOM LCG VARIATIONS ===")
    print("Based on survivor analysis: Java_LCG or 32bit_LCG with custom output")
    
    # Load data
    seeds = load_bidirectional_seeds()
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    
    print(f"Testing {len(seeds):,} seeds against: {recent_draws}")
    
    results = []
    
    # Test Java LCG with different output transformations
    print("\n🔍 Testing Java LCG Output Transformations...")
    
    transformations = [
        # Standard Java (>> 16)
        ("java_std", lambda x: (x >> 16) & 0xFFFFFFFF),
        # Different shifts
        ("shift_15", lambda x: (x >> 15) & 0xFFFFFFFF),
        ("shift_17", lambda x: (x >> 17) & 0xFFFFFFFF), 
        ("shift_18", lambda x: (x >> 18) & 0xFFFFFFFF),
        ("shift_24", lambda x: (x >> 24) & 0xFFFFFFFF),
        # Mask variations
        ("mask_0x7FFF", lambda x: (x >> 16) & 0x7FFF),  # 15 bits
        ("mask_0xFFFF", lambda x: (x >> 16) & 0xFFFF),  # 16 bits
        ("mask_0x3FFFF", lambda x: (x >> 16) & 0x3FFFF), # 18 bits
        # XOR transformations
        ("xor_low16", lambda x: ((x >> 16) ^ (x & 0xFFFF)) & 0xFFFF),
        ("xor_high_low", lambda x: ((x >> 32) ^ (x & 0xFFFF)) & 0xFFFF),
    ]
    
    for name, transform in transformations:
        matches = test_lcg_variant(seeds[:50], recent_draws, 
                                 a=25214903917, c=11, m=2**48,
                                 output_transform=transform)
        if matches > 0:
            accuracy = (matches / min(50, len(seeds))) * 100
            results.append((f"Java_LCG_{name}", accuracy, matches))
            print(f"  ✅ {name}: {accuracy:.1f}% ({matches}/50)")
    
    # Test 32-bit LCG variations
    print("\n🔍 Testing 32-bit LCG Variations...")
    
    lcg_32_params = [
        ("std_c", 1103515245, 12345, 2**31),
        ("mmix", 6364136223846793005 & 0xFFFFFFFF, 1442695040888963407 & 0xFFFFFFFF, 2**32),
        ("numerical_recipes", 1664525, 1013904223, 2**32),
        ("borland", 22695477, 1, 2**32),
        ("vc", 214013, 2531011, 2**32),
    ]
    
    for name, a, c, m in lcg_32_params:
        matches = test_lcg_variant(seeds[:50], recent_draws, a, c, m)
        if matches > 0:
            accuracy = (matches / min(50, len(seeds))) * 100
            results.append((f"LCG32_{name}", accuracy, matches))
            print(f"  ✅ {name}: {accuracy:.1f}% ({matches}/50)")
    
    # Test custom ranges (maybe not 0-999)
    print("\n🔍 Testing Custom Output Ranges...")
    
    ranges = [999, 1000, 899, 799, 699]  # Common lottery ranges
    
    for max_val in ranges:
        matches = test_lcg_variant(seeds[:50], recent_draws,
                                 a=25214903917, c=11, m=2**48,
                                 output_range=max_val)
        if matches > 0:
            accuracy = (matches / min(50, len(seeds))) * 100
            results.append((f"Range_{max_val}", accuracy, matches))
            print(f"  ✅ Range 0-{max_val}: {accuracy:.1f}% ({matches}/50)")
    
    return results

def test_lcg_variant(seeds, recent_draws, a, c, m, output_transform=None, output_range=1000):
    """Test a specific LCG variant"""
    matches = 0
    
    for seed in seeds:
        try:
            sequence = generate_lcg_sequence(seed, 50, a, c, m, output_transform, output_range)
            if check_sequence_matches(sequence, recent_draws):
                matches += 1
        except:
            continue
    
    return matches

def generate_lcg_sequence(seed, n, a, c, m, output_transform=None, output_range=1000):
    """Generate LCG sequence with optional output transformation"""
    sequence = []
    state = seed % m
    
    for _ in range(n):
        state = (a * state + c) % m
        
        if output_transform:
            output = output_transform(state)
        else:
            output = state
        
        sequence.append(output % (output_range + 1))
    
    return sequence

def load_bidirectional_seeds():
    """Load the 27,902 bidirectional seeds"""
    forward_file = 'results/window_opt_forward_244_139.json'
    reverse_file = 'results/window_opt_reverse_244_139.json'
    
    def extract_seeds(data):
        seeds = set()
        if 'results' in data:
            for result in data['results']:
                if 'survivors' in result and isinstance(result['survivors'], list):
                    for survivor in result['survivors']:
                        if isinstance(survivor, dict) and 'seed' in survivor:
                            seeds.add(survivor['seed'])
        return seeds
    
    try:
        with open(forward_file, 'r') as f:
            forward_data = json.load(f)
            forward_seeds = extract_seeds(forward_data)
        
        with open(reverse_file, 'r') as f:
            reverse_data = json.load(f)
            reverse_seeds = extract_seeds(reverse_data)
        
        return list(forward_seeds & reverse_seeds)
        
    except Exception as e:
        print(f"❌ Error loading seeds: {e}")
        return []

def check_sequence_matches(sequence, recent_draws):
    """Check if sequence contains any 3-number pattern from recent draws"""
    if len(sequence) < 3:
        return False
        
    for i in range(len(sequence) - 2):
        test_window = sequence[i:i+3]
        for j in range(len(recent_draws) - 2):
            if test_window == recent_draws[j:j+3]:
                return True
    return False

def main():
    results = test_custom_lcg_variations()
    
    if results:
        print(f"\n🎯 SUCCESSFUL VARIATIONS FOUND:")
        print("=" * 50)
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        for name, accuracy, matches in results:
            print(f"  {name:25} {accuracy:5.1f}% ({matches}/50)")
        
        best_name, best_accuracy, _ = results[0]
        print(f"\n🚀 BEST MATCH: {best_name} ({best_accuracy:.1f}% accuracy)")
        print(f"\n💡 BREAKTHROUGH: You've identified the PRNG type!")
        print("   Next: Run full prediction with this configuration")
        
    else:
        print(f"\n🔍 No variations matched perfectly")
        print(f"\n💡 DEEPER ANALYSIS NEEDED:")
        print("   1. The PRNG might combine multiple LCGs")
        print("   2. There could be additional state mixing")
        print("   3. Try testing with your actual GPU kernels")
        print("   4. The output might use a different base (not decimal)")

if __name__ == "__main__":
    main()
