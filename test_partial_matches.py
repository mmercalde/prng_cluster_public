import json
import numpy as np
from collections import Counter

def test_partial_matches():
    print("=== FINDING PARTIAL MATCHES ===")
    print("Looking for patterns that get CLOSE to matching")
    
    # Load data
    seeds = load_bidirectional_seeds()
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    
    print(f"Testing {len(seeds):,} seeds against: {recent_draws}")
    
    results = []
    
    # Test Java LCG with different output transformations
    print("\n🔍 Testing Java LCG Output Transformations (showing partial matches)...")
    
    transformations = [
        ("java_std", lambda x: (x >> 16) & 0xFFFFFFFF),
        ("shift_15", lambda x: (x >> 15) & 0xFFFFFFFF),
        ("shift_17", lambda x: (x >> 17) & 0xFFFFFFFF),
        ("shift_18", lambda x: (x >> 18) & 0xFFFFFFFF),
        ("shift_24", lambda x: (x >> 24) & 0xFFFFFFFF),
        ("mask_0x7FFF", lambda x: (x >> 16) & 0x7FFF),
        ("mask_0xFFFF", lambda x: (x >> 16) & 0xFFFF),
        ("mask_0x3FFFF", lambda x: (x >> 16) & 0x3FFFF),
        ("xor_low16", lambda x: ((x >> 16) ^ (x & 0xFFFF)) & 0xFFFF),
        ("xor_high_low", lambda x: ((x >> 32) ^ (x & 0xFFFF)) & 0xFFFF),
    ]
    
    for name, transform in transformations:
        best_match_rate = test_lcg_partial_matches(seeds[:100], recent_draws, 
                                                  a=25214903917, c=11, m=2**48,
                                                  output_transform=transform)
        if best_match_rate > 0:
            results.append((f"Java_LCG_{name}", best_match_rate))
            print(f"  {name:15} Best match rate: {best_match_rate:.1f}%")
    
    # Test 32-bit LCG variations
    print("\n🔍 Testing 32-bit LCG Variations (showing partial matches)...")
    
    lcg_32_params = [
        ("std_c", 1103515245, 12345, 2**31),
        ("mmix", 6364136223846793005 & 0xFFFFFFFF, 1442695040888963407 & 0xFFFFFFFF, 2**32),
        ("numerical_recipes", 1664525, 1013904223, 2**32),
        ("borland", 22695477, 1, 2**32),
        ("vc", 214013, 2531011, 2**32),
    ]
    
    for name, a, c, m in lcg_32_params:
        best_match_rate = test_lcg_partial_matches(seeds[:100], recent_draws, a, c, m)
        if best_match_rate > 0:
            results.append((f"LCG32_{name}", best_match_rate))
            print(f"  {name:20} Best match rate: {best_match_rate:.1f}%")
    
    # Test custom ranges
    print("\n🔍 Testing Custom Output Ranges (showing partial matches)...")
    
    ranges = [999, 1000, 899, 799, 699, 500, 400, 300]
    
    for max_val in ranges:
        best_match_rate = test_lcg_partial_matches(seeds[:100], recent_draws,
                                                  a=25214903917, c=11, m=2**48,
                                                  output_range=max_val)
        if best_match_rate > 0:
            results.append((f"Range_{max_val}", best_match_rate))
            print(f"  Range 0-{max_val:3} Best match rate: {best_match_rate:.1f}%")
    
    # Show what patterns ARE matching
    print("\n🔍 Analyzing WHAT patterns are matching...")
    analyze_matching_patterns(seeds[:20], recent_draws)
    
    return results

def test_lcg_partial_matches(seeds, recent_draws, a, c, m, output_transform=None, output_range=1000):
    """Test LCG variant and return best partial match rate"""
    best_match_rate = 0
    
    for seed in seeds:
        try:
            sequence = generate_lcg_sequence(seed, 50, a, c, m, output_transform, output_range)
            match_rate = calculate_partial_match_rate(sequence, recent_draws)
            best_match_rate = max(best_match_rate, match_rate)
        except:
            continue
    
    return best_match_rate

def calculate_partial_match_rate(sequence, recent_draws):
    """Calculate how well the sequence matches recent patterns"""
    max_matches = 0
    
    # Check different pattern lengths
    for pattern_len in [2, 3, 4]:  # 2-number, 3-number, 4-number patterns
        matches = 0
        for i in range(len(sequence) - pattern_len + 1):
            test_window = sequence[i:i+pattern_len]
            for j in range(len(recent_draws) - pattern_len + 1):
                if test_window == recent_draws[j:j+pattern_len]:
                    matches += 1
                    break
        
        match_rate = (matches / len(sequence)) * 100
        max_matches = max(max_matches, match_rate)
    
    return max_matches

def analyze_matching_patterns(seeds, recent_draws):
    """Analyze what specific patterns are matching"""
    print("\n  Analyzing matching patterns for Java LCG (standard)...")
    
    pattern_counts = Counter()
    
    for seed in seeds:
        try:
            sequence = generate_lcg_sequence(seed, 100, 25214903917, 11, 2**48)
            
            # Check for 2-number patterns
            for i in range(len(sequence) - 1):
                pair = tuple(sequence[i:i+2])
                if pair in [tuple(recent_draws[j:j+2]) for j in range(len(recent_draws)-1)]:
                    pattern_counts[pair] += 1
            
            # Check for single number matches
            for num in sequence:
                if num in recent_draws:
                    pattern_counts[(num,)] += 1
                    
        except:
            continue
    
    # Show most common matches
    print("  Most common matching patterns:")
    for pattern, count in pattern_counts.most_common(10):
        if count > 0:
            print(f"    {pattern}: {count} seeds")
    
    # Analyze number distribution
    print("  Number distribution in generated sequences:")
    all_numbers = []
    for seed in seeds[:10]:
        sequence = generate_lcg_sequence(seed, 50, 25214903917, 11, 2**48)
        all_numbers.extend(sequence)
    
    number_freq = Counter(all_numbers)
    print(f"    Most common numbers: {number_freq.most_common(10)}")
    print(f"    Recent draws: {recent_draws}")

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

def main():
    results = test_partial_matches()
    
    if results:
        print(f"\n📊 PARTIAL MATCH RESULTS:")
        print("=" * 50)
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        for name, match_rate in results:
            if match_rate > 10:  # Only show meaningful matches
                print(f"  {name:25} {match_rate:5.1f}% match rate")
        
        # Analyze the best performers
        best_name, best_rate = results[0]
        print(f"\n🎯 BEST PERFORMER: {best_name} ({best_rate:.1f}% match rate)")
        
        if best_rate > 30:
            print("🚀 STRONG PARTIAL MATCH - This is likely the right direction!")
        elif best_rate > 15:
            print("💡 MODERATE MATCH - Getting warm, needs refinement")
        else:
            print("🔍 WEAK MATCH - May need completely different approach")
            
    else:
        print(f"\n❌ No meaningful partial matches found")
        print("This suggests the PRNG might be fundamentally different than LCG")

if __name__ == "__main__":
    main()
