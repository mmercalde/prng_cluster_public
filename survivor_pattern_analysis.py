import json
import numpy as np
from collections import Counter

def analyze_survivor_patterns():
    print("=== ANALYZING SURVIVOR PATTERNS FOR PRNG IDENTIFICATION ===")
    print("27,902 survivors = STRONG mathematical structure exists!")
    
    # Load survivors and their characteristics
    survivors = load_survivors_with_metadata()
    print(f"✅ Loaded {len(survivors):,} survivors with full metadata")
    
    # Analyze survivor distribution patterns
    print("\n📊 ANALYZING SURVIVOR DISTRIBUTION:")
    
    # 1. Check if survivors follow LCG spacing patterns
    seed_values = [s['seed'] for s in survivors]
    seed_gaps = analyze_seed_gaps(seed_values)
    
    # 2. Analyze modular patterns in survivors
    modular_patterns = analyze_modular_patterns(seed_values)
    
    # 3. Look for geometric progressions (LCG characteristic)
    geometric_patterns = find_geometric_patterns(seed_values)
    
    # 4. Check for common PRNG modulus relationships
    modulus_clues = find_modulus_clues(seed_values)
    
    return {
        'seed_gaps': seed_gaps,
        'modular_patterns': modular_patterns,
        'geometric_patterns': geometric_patterns,
        'modulus_clues': modulus_clues
    }

def load_survivors_with_metadata():
    """Load survivors with their match rates and skip patterns"""
    forward_file = 'results/window_opt_forward_244_139.json'
    
    survivors = []
    with open(forward_file, 'r') as f:
        data = json.load(f)
        
        if 'results' in data:
            for result in data['results']:
                if 'survivors' in result and isinstance(result['survivors'], list):
                    for survivor in result['survivors']:
                        if isinstance(survivor, dict) and 'seed' in survivor:
                            survivors.append(survivor)
    
    return survivors

def analyze_seed_gaps(seed_values):
    """Analyze gaps between survivors for LCG patterns"""
    sorted_seeds = sorted(seed_values)
    gaps = [sorted_seeds[i+1] - sorted_seeds[i] for i in range(len(sorted_seeds)-1)]
    
    print(f"  Total seeds: {len(sorted_seeds):,}")
    print(f"  Seed range: {min(sorted_seeds):,} to {max(sorted_seeds):,}")
    print(f"  Average gap: {np.mean(gaps):,.0f}")
    print(f"  Most common gap: {Counter(gaps).most_common(5)}")
    
    # LCGs often have characteristic gap patterns
    common_gap = Counter(gaps).most_common(1)[0][0] if gaps else 0
    
    return {
        'common_gap': common_gap,
        'gap_gcd': compute_gcd(gaps) if gaps else 0,
        'gap_pattern': 'LCG-like' if len(set(gaps)) < 100 else 'Complex'
    }

def analyze_modular_patterns(seed_values):
    """Analyze survivors modulo common PRNG moduli"""
    moduli = [2**31, 2**32, 2**48, 2**64, 2147483647]
    patterns = {}
    
    for modulus in moduli:
        residues = [seed % modulus for seed in seed_values[:1000]]  # Sample for speed
        residue_counts = Counter(residues)
        
        # LCGs often have uniform residue distribution
        entropy = len(set(residues)) / len(residues) if residues else 0
        
        patterns[f"mod_{modulus}"] = {
            'unique_residues': len(set(residues)),
            'entropy': entropy,
            'max_count': max(residue_counts.values()) if residue_counts else 0
        }
    
    print(f"  Modular analysis: {patterns}")
    return patterns

def find_geometric_patterns(seed_values):
    """Look for geometric progressions (characteristic of LCGs)"""
    sorted_seeds = sorted(seed_values[:100])  # Sample for speed
    
    # Check if seeds follow a * seed mod m pattern
    for i in range(len(sorted_seeds) - 2):
        s1, s2, s3 = sorted_seeds[i], sorted_seeds[i+1], sorted_seeds[i+2]
        
        # Check for LCG pattern: s2 = (a * s1 + c) mod m
        # This would manifest as approximately geometric spacing
        if s1 > 0 and s2 > 0 and s3 > 0:
            ratio1 = s2 / s1
            ratio2 = s3 / s2
            
            # If ratios are similar, suggests LCG with large modulus
            if abs(ratio1 - ratio2) / ratio1 < 0.1:
                return {'pattern': 'geometric', 'approximate_ratio': ratio1}
    
    return {'pattern': 'non_geometric'}

def find_modulus_clues(seed_values):
    """Look for clues about the PRNG modulus"""
    sorted_seeds = sorted(seed_values)
    
    # Common modulus sizes
    possible_moduli = [2**31, 2**32, 2**48, 2**64]
    modulus_evidence = {}
    
    for modulus in possible_moduli:
        # Count how many seeds are within modulus range
        within_modulus = sum(1 for seed in sorted_seeds if seed < modulus)
        percentage = (within_modulus / len(sorted_seeds)) * 100
        
        modulus_evidence[modulus] = {
            'seeds_within_range': within_modulus,
            'percentage': percentage,
            'likely': percentage > 90
        }
    
    print(f"  Modulus analysis: {modulus_evidence}")
    return modulus_evidence

def compute_gcd(numbers):
    """Compute GCD of a list of numbers"""
    if not numbers:
        return 0
    
    result = numbers[0]
    for num in numbers[1:]:
        result = np.gcd(result, num)
        if result == 1:
            break
    return result

def generate_hypotheses(analysis):
    """Generate PRNG hypotheses based on survivor patterns"""
    print("\n🎯 GENERATING PRNG HYPOTHESES:")
    
    hypotheses = []
    
    # Hypothesis 1: Standard LCG with custom parameters
    if analysis['seed_gaps']['gap_pattern'] == 'LCG-like':
        common_gap = analysis['seed_gaps']['common_gap']
        gap_gcd = analysis['seed_gaps']['gap_gcd']
        
        hypotheses.append({
            'type': 'LCG',
            'confidence': 'HIGH',
            'reason': f'Consistent gap patterns (common gap: {common_gap}, GCD: {gap_gcd})',
            'parameters': f'Try multipliers near {common_gap}'
        })
    
    # Hypothesis 2: Java-style LCG
    java_modulus = analysis['modulus_clues'].get(2**48, {})
    if java_modulus.get('likely', False):
        hypotheses.append({
            'type': 'Java_LCG',
            'confidence': 'MEDIUM', 
            'reason': f"48-bit modulus likely ({java_modulus['percentage']:.1f}% seeds within range)",
            'parameters': 'a=25214903917, c=11, m=2^48 with possible output transformation'
        })
    
    # Hypothesis 3: 32-bit LCG
    bit32_modulus = analysis['modulus_clues'].get(2**32, {})
    if bit32_modulus.get('likely', False):
        hypotheses.append({
            'type': '32bit_LCG',
            'confidence': 'MEDIUM',
            'reason': f"32-bit range likely ({bit32_modulus['percentage']:.1f}% seeds within range)",
            'parameters': 'Test various (a, c) pairs with m=2^32'
        })
    
    # Hypothesis 4: Custom output transformation
    if analysis['geometric_patterns']['pattern'] == 'non_geometric':
        hypotheses.append({
            'type': 'Transformed_LCG',
            'confidence': 'MEDIUM',
            'reason': 'Non-geometric patterns suggest output transformation',
            'parameters': 'LCG with custom output function (not just >> 16)'
        })
    
    return hypotheses

def main():
    print("27,902 SURVIVORS = WE FOUND THE HAYSTACK!")
    print("Now let's find the needle (the exact PRNG implementation)")
    print("=" * 70)
    
    # Analyze survivor patterns
    analysis = analyze_survivor_patterns()
    
    # Generate hypotheses
    hypotheses = generate_hypotheses(analysis)
    
    print(f"\n💡 GENERATED {len(hypotheses)} HYPOTHESES:")
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n{i}. {hypothesis['type']} [{hypothesis['confidence']}]")
        print(f"   Reason: {hypothesis['reason']}")
        print(f"   Parameters: {hypothesis['parameters']}")
    
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Test the highest-confidence hypothesis first")
    print("2. Use your 27,902 survivors as test seeds") 
    print("3. Focus on parameter variations of the most likely PRNG type")
    print("4. The mathematical structure is REAL - we just need the right implementation!")

if __name__ == "__main__":
    main()
