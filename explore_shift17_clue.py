import json
import numpy as np
from collections import Counter

def explore_shift17_clue():
    print("=== EXPLORING SHIFT_17 CLUE ===")
    print("2% match rate with shift_17 is a BREAKTHROUGH!")
    print("This suggests Java LCG with custom output transformation")
    
    # Load data
    seeds = load_bidirectional_seeds()
    with open('daily3.json', 'r') as f:
        lottery_data = json.load(f)
    recent_draws = [d['draw'] for d in lottery_data[-10:]]
    
    print(f"Testing {len(seeds):,} seeds against: {recent_draws}")
    
    # Focus on shift_17 variations
    print("\n🔍 DEEPER ANALYSIS OF SHIFT_17...")
    
    # Test different combinations with shift_17
    variations = [
        # Basic shift variations
        ("shift_17_basic", lambda x: (x >> 17) & 0xFFFFFFFF),
        ("shift_17_mask_7FFF", lambda x: (x >> 17) & 0x7FFF),      # 15 bits
        ("shift_17_mask_FFFF", lambda x: (x >> 17) & 0xFFFF),      # 16 bits  
        ("shift_17_mask_1FFFF", lambda x: (x >> 17) & 0x1FFFF),    # 17 bits
        ("shift_17_mask_3FFFF", lambda x: (x >> 17) & 0x3FFFF),    # 18 bits
        
        # XOR combinations
        ("shift_17_xor_low16", lambda x: ((x >> 17) ^ (x & 0xFFFF)) & 0xFFFF),
        ("shift_17_xor_shift16", lambda x: ((x >> 17) ^ (x >> 16)) & 0xFFFF),
        ("shift_17_xor_low17", lambda x: ((x >> 17) ^ (x & 0x1FFFF)) & 0x1FFFF),
        
        # Addition combinations
        ("shift_17_plus_low16", lambda x: ((x >> 17) + (x & 0xFFFF)) & 0xFFFF),
        ("shift_17_plus_shift16", lambda x: ((x >> 17) + (x >> 16)) & 0xFFFF),
        
        # Multiplication combinations
        ("shift_17_times_low16", lambda x: ((x >> 17) * (x & 0xFF)) & 0xFFFF),
    ]
    
    results = []
    
    for name, transform in variations:
        match_details = test_transformation_detailed(seeds[:50], recent_draws, transform)
        if match_details['best_match_rate'] > 0:
            results.append((name, match_details))
            print(f"  {name:25} {match_details['best_match_rate']:5.1f}% match")
    
    # Analyze what's working
    print("\n🔍 ANALYZING SUCCESSFUL PATTERNS...")
    if results:
        best_name, best_details = max(results, key=lambda x: x[1]['best_match_rate'])
        analyze_successful_patterns(seeds[:10], recent_draws, best_details['best_seed'], best_name)
    
    return results

def test_transformation_detailed(seeds, recent_draws, transform):
    """Test transformation with detailed analysis"""
    best_match_rate = 0
    best_seed = None
    pattern_analysis = []
    
    for seed in seeds:
        try:
            sequence = generate_java_lcg_sequence(seed, 100, transform)
            match_rate, patterns = analyze_sequence_matches(sequence, recent_draws)
            
            if match_rate > best_match_rate:
                best_match_rate = match_rate
                best_seed = seed
                pattern_analysis = patterns
        except:
            continue
    
    return {
        'best_match_rate': best_match_rate,
        'best_seed': best_seed,
        'patterns': pattern_analysis
    }

def analyze_sequence_matches(sequence, recent_draws):
    """Analyze what patterns match between sequence and recent draws"""
    matches = 0
    matched_patterns = []
    
    # Check different pattern lengths
    for pattern_len in [2, 3]:
        for i in range(len(sequence) - pattern_len + 1):
            test_window = sequence[i:i+pattern_len]
            for j in range(len(recent_draws) - pattern_len + 1):
                if test_window == recent_draws[j:j+pattern_len]:
                    matches += 1
                    matched_patterns.append(test_window)
                    break
    
    match_rate = (matches / len(sequence)) * 100
    return match_rate, matched_patterns

def analyze_successful_patterns(seeds, recent_draws, best_seed, transform_name):
    """Analyze what makes the best seed work"""
    if best_seed is None:
        return
    
    print(f"\n  Analyzing best seed {best_seed} with {transform_name}:")
    
    # Generate sequence from best seed
    sequence = generate_java_lcg_sequence(best_seed, 50, 
        lambda x: (x >> 17) & 0xFFFF)  # Use the most promising transform
    
    print(f"  First 20 numbers: {sequence[:20]}")
    print(f"  Recent draws: {recent_draws}")
    
    # Find exact matches
    exact_matches = []
    for i in range(len(sequence) - 2):
        test_window = sequence[i:i+3]
        for j in range(len(recent_draws) - 2):
            if test_window == recent_draws[j:j+3]:
                exact_matches.append((i, test_window, j))
    
    if exact_matches:
        print(f"  🎯 EXACT 3-NUMBER MATCHES FOUND:")
        for match in exact_matches:
            print(f"    Position {match[0]}: {match[1]} = Recent[{match[2]}:{match[2]+3}]")
    else:
        print("  No exact 3-number matches")
        
        # Check 2-number matches
        two_number_matches = []
        for i in range(len(sequence) - 1):
            test_pair = sequence[i:i+2]
            for j in range(len(recent_draws) - 1):
                if test_pair == recent_draws[j:j+2]:
                    two_number_matches.append((i, test_pair, j))
        
        if two_number_matches:
            print(f"  🔍 2-NUMBER MATCHES:")
            for match in two_number_matches[:5]:  # Show first 5
                print(f"    Position {match[0]}: {match[1]} = Recent[{match[2]}:{match[2]+2}]")

def generate_java_lcg_sequence(seed, n, output_transform):
    """Generate Java LCG sequence with custom output transformation"""
    sequence = []
    state = seed & 0xFFFFFFFFFFFF  # 48-bit mask
    a = 25214903917
    c = 11
    
    for _ in range(n):
        state = (a * state + c) & 0xFFFFFFFFFFFF
        output = output_transform(state)
        sequence.append(output % 1000)
    
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
    results = explore_shift17_clue()
    
    if results:
        print(f"\n📊 SHIFT_17 VARIATION RESULTS:")
        print("=" * 50)
        
        results.sort(key=lambda x: x[1]['best_match_rate'], reverse=True)
        
        for name, details in results:
            if details['best_match_rate'] > 1:  # Only show meaningful matches
                print(f"  {name:30} {details['best_match_rate']:5.1f}%")
        
        best_name, best_details = results[0]
        best_rate = best_details['best_match_rate']
        
        print(f"\n🎯 BREAKTHROUGH ANALYSIS:")
        print(f"Best transformation: {best_name}")
        print(f"Match rate: {best_rate:.1f}%")
        
        if best_rate > 5:
            print("🚀 STRONG EVIDENCE - This is likely the right transformation!")
            print("💡 The lottery PRNG uses Java LCG with shift_17 output")
        elif best_rate > 2:
            print("💡 PROMISING CLUE - Getting very close!")
            print("   There might be one more small transformation needed")
        else:
            print("🔍 WEAK BUT MEANINGFUL - The direction is correct")
            
    else:
        print(f"\n❌ No variations improved on basic shift_17")

if __name__ == "__main__":
    main()
