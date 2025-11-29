#!/usr/bin/env python3
"""
Validate Bidirectional Survivors Against Target Dataset
========================================================

Tests all 90 seeds to find which one(s) actually generated the lottery sequence.

Usage:
    python3 validate_survivors.py --target daily3.json --seeds bidirectional_seeds.txt

Features:
    - Tests each seed with multiple skip values (0-30)
    - Compares against actual lottery draws
    - Scores based on exact sequence matches
    - Outputs ranked list of best candidates
    - Extracts 3-lane coherence features for ML
"""

import json
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# =============================================================================
# PRNG IMPLEMENTATIONS (Must match your GPU kernels exactly!)
# =============================================================================

class JavaLCG:
    """Java's Linear Congruential Generator (java.util.Random)"""
    
    def __init__(self, seed: int):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    
    def next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return int(self.seed >> (48 - bits))
    
    def next_int(self, bound: int) -> int:
        """Generate random int in range [0, bound)"""
        if (bound & -bound) == bound:  # Power of 2
            return int((bound * self.next(31)) >> 31)
        
        bits = self.next(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self.next(31)
            val = bits % bound
        return val


class MT19937:
    """Mersenne Twister MT19937 (32-bit)"""
    
    def __init__(self, seed: int):
        self.MT = [0] * 624
        self.index = 624
        self.MT[0] = seed & 0xFFFFFFFF
        
        for i in range(1, 624):
            self.MT[i] = (0x6C078965 * (self.MT[i-1] ^ (self.MT[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    def extract_number(self) -> int:
        if self.index >= 624:
            self.twist()
        
        y = self.MT[self.index]
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        
        self.index += 1
        return y
    
    def twist(self):
        for i in range(624):
            x = (self.MT[i] & 0x80000000) + (self.MT[(i+1) % 624] & 0x7FFFFFFF)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= 0x9908B0DF
            self.MT[i] = self.MT[(i + 397) % 624] ^ xA
        self.index = 0
    
    def next_int(self, bound: int) -> int:
        """Generate random int in range [0, bound)"""
        return self.extract_number() % bound


class Xoshiro256PlusPlus:
    """xoshiro256++ (64-bit state, 64-bit output)"""
    
    def __init__(self, seed: int):
        # SplitMix64 initialization
        self.s = [0, 0, 0, 0]
        z = seed
        for i in range(4):
            z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            result = z
            result = ((result ^ (result >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            result = ((result ^ (result >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            self.s[i] = (result ^ (result >> 31)) & 0xFFFFFFFFFFFFFFFF
    
    @staticmethod
    def rotl(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF
    
    def next(self) -> int:
        result = (self.rotl(self.s[0] + self.s[3], 23) + self.s[0]) & 0xFFFFFFFFFFFFFFFF
        t = (self.s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        
        self.s[2] ^= self.s[0]
        self.s[3] ^= self.s[1]
        self.s[1] ^= self.s[2]
        self.s[0] ^= self.s[3]
        self.s[2] ^= t
        self.s[3] = self.rotl(self.s[3], 45)
        
        return result
    
    def next_int(self, bound: int) -> int:
        """Generate random int in range [0, bound)"""
        return int(self.next() % bound)


# =============================================================================
# PRNG REGISTRY
# =============================================================================

PRNG_REGISTRY = {
    'java_lcg': JavaLCG,
    'mt19937': MT19937,
    'xoshiro256pp': Xoshiro256PlusPlus,
}


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

@dataclass
class ValidationResult:
    """Results from validating a single seed"""
    seed: int
    prng_type: str
    best_skip: int
    exact_matches: int
    total_draws: int
    match_rate: float
    lane_coherence: Dict[str, float]
    consecutive_matches: int
    score: float


def load_target_dataset(filepath: str, session: Optional[str] = None) -> List[int]:
    """Load lottery draws from JSON dataset
    
    Args:
        filepath: Path to daily3.json
        session: 'midday', 'evening', or None (both)
    
    Returns:
        List of draw values (0-999) in chronological order
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    draws = []
    for entry in reversed(data):  # Oldest first
        if session is None or entry.get('session') == session:
            draw_value = int(entry['draw']) % 1000
            draws.append(draw_value)
    
    return draws


def generate_sequence(prng_type: str, seed: int, skip: int, length: int) -> List[int]:
    """Generate sequence from PRNG with skip pattern
    
    Args:
        prng_type: 'java_lcg', 'mt19937', 'xoshiro256pp'
        seed: Initial seed value
        skip: Number of values to skip between draws
        length: Number of draws to generate
    
    Returns:
        List of generated draw values (0-999)
    """
    prng_class = PRNG_REGISTRY.get(prng_type)
    if not prng_class:
        raise ValueError(f"Unknown PRNG type: {prng_type}")
    
    prng = prng_class(seed)
    sequence = []
    
    for _ in range(length):
        # Skip values
        for _ in range(skip):
            prng.next_int(1000)
        
        # Get the actual draw
        value = prng.next_int(1000)
        sequence.append(value)
    
    return sequence


def compute_3lane_coherence(generated: List[int], target: List[int]) -> Dict[str, float]:
    """Compute per-lane match coherence scores
    
    Returns dict with:
        - mod8: Percentage matching on lane 8
        - mod125: Percentage matching on lane 125
        - mod1000: Percentage matching on lane 1000
    """
    if len(generated) != len(target):
        raise ValueError("Sequences must be same length")
    
    total = len(generated)
    matches_8 = sum(1 for g, t in zip(generated, target) if (g % 8) == (t % 8))
    matches_125 = sum(1 for g, t in zip(generated, target) if (g % 125) == (t % 125))
    matches_1000 = sum(1 for g, t in zip(generated, target) if g == t)
    
    return {
        'mod8': matches_8 / total,
        'mod125': matches_125 / total,
        'mod1000': matches_1000 / total
    }


def find_longest_consecutive_matches(generated: List[int], target: List[int]) -> int:
    """Find longest streak of consecutive exact matches"""
    max_streak = 0
    current_streak = 0
    
    for g, t in zip(generated, target):
        if g == t:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def validate_seed(seed: int, prng_type: str, target: List[int], 
                  skip_range: Tuple[int, int] = (0, 30)) -> ValidationResult:
    """Validate a single seed against target dataset
    
    Tests all skip values in range and returns best result.
    
    Args:
        seed: Seed value to test
        prng_type: Type of PRNG
        target: Target lottery sequence
        skip_range: (min_skip, max_skip) to test
    
    Returns:
        ValidationResult with best skip configuration
    """
    best_result = None
    best_score = -1
    
    for skip in range(skip_range[0], skip_range[1] + 1):
        # Generate sequence with this skip
        generated = generate_sequence(prng_type, seed, skip, len(target))
        
        # Count exact matches
        exact_matches = sum(1 for g, t in zip(generated, target) if g == t)
        match_rate = exact_matches / len(target)
        
        # Compute 3-lane coherence
        lane_coherence = compute_3lane_coherence(generated, target)
        
        # Find consecutive matches
        consecutive = find_longest_consecutive_matches(generated, target)
        
        # Compute composite score
        # Weight: exact matches (50%), lane coherence (30%), consecutive (20%)
        score = (
            match_rate * 0.5 +
            (lane_coherence['mod8'] + lane_coherence['mod125'] + lane_coherence['mod1000']) / 3 * 0.3 +
            (consecutive / len(target)) * 0.2
        )
        
        if score > best_score:
            best_score = score
            best_result = ValidationResult(
                seed=seed,
                prng_type=prng_type,
                best_skip=skip,
                exact_matches=exact_matches,
                total_draws=len(target),
                match_rate=match_rate,
                lane_coherence=lane_coherence,
                consecutive_matches=consecutive,
                score=score
            )
    
    return best_result


def validate_all_seeds(seeds: List[int], prng_type: str, target: List[int],
                       skip_range: Tuple[int, int] = (0, 30)) -> List[ValidationResult]:
    """Validate all seeds and return sorted results
    
    Args:
        seeds: List of seed values to test
        prng_type: Type of PRNG
        target: Target lottery sequence
        skip_range: (min_skip, max_skip) to test
    
    Returns:
        List of ValidationResults sorted by score (best first)
    """
    results = []
    total = len(seeds)
    
    print(f"\n{'='*70}")
    print(f"VALIDATING {total} SEEDS")
    print(f"{'='*70}")
    print(f"PRNG Type: {prng_type}")
    print(f"Target draws: {len(target)}")
    print(f"Skip range: {skip_range[0]}-{skip_range[1]}")
    print(f"{'='*70}\n")
    
    for i, seed in enumerate(seeds, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{total} seeds validated...")
        
        result = validate_seed(seed, prng_type, target, skip_range)
        results.append(result)
    
    # Sort by score (descending)
    results.sort(key=lambda r: r.score, reverse=True)
    
    return results


def print_validation_report(results: List[ValidationResult], top_n: int = 10):
    """Print comprehensive validation report"""
    
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS - TOP {top_n} CANDIDATES")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results[:top_n], 1):
        print(f"Rank #{i}: Seed {r.seed}")
        print(f"  Score: {r.score:.4f}")
        print(f"  Skip: {r.best_skip}")
        print(f"  Exact Matches: {r.exact_matches}/{r.total_draws} ({r.match_rate*100:.2f}%)")
        print(f"  Lane Coherence:")
        print(f"    - mod 8:    {r.lane_coherence['mod8']*100:.2f}%")
        print(f"    - mod 125:  {r.lane_coherence['mod125']*100:.2f}%")
        print(f"    - mod 1000: {r.lane_coherence['mod1000']*100:.2f}%")
        print(f"  Longest consecutive matches: {r.consecutive_matches}")
        print()
    
    # Winner analysis
    winner = results[0]
    print(f"{'='*70}")
    print(f"🏆 WINNER: Seed {winner.seed}")
    print(f"{'='*70}")
    print(f"This seed achieved {winner.match_rate*100:.2f}% exact match rate")
    print(f"with skip={winner.best_skip} and scored {winner.score:.4f}")
    
    if winner.match_rate > 0.95:
        print(f"\n✅ EXTREMELY HIGH CONFIDENCE - This is likely THE seed!")
    elif winner.match_rate > 0.80:
        print(f"\n✅ HIGH CONFIDENCE - Strong candidate")
    elif winner.match_rate > 0.50:
        print(f"\n⚠️  MODERATE CONFIDENCE - Possible candidate")
    else:
        print(f"\n❌ LOW CONFIDENCE - May need different PRNG type or parameters")
    
    print(f"{'='*70}\n")


def save_results_json(results: List[ValidationResult], output_file: str):
    """Save validation results to JSON file"""
    
    data = {
        'timestamp': np.datetime64('now').astype(str),
        'total_seeds_tested': len(results),
        'winner': {
            'seed': results[0].seed,
            'score': results[0].score,
            'skip': results[0].best_skip,
            'match_rate': results[0].match_rate,
            'exact_matches': results[0].exact_matches,
            'total_draws': results[0].total_draws,
            'lane_coherence': results[0].lane_coherence,
            'consecutive_matches': results[0].consecutive_matches
        },
        'all_results': [
            {
                'seed': r.seed,
                'prng_type': r.prng_type,
                'skip': r.best_skip,
                'score': r.score,
                'match_rate': r.match_rate,
                'exact_matches': r.exact_matches,
                'lane_coherence': r.lane_coherence,
                'consecutive_matches': r.consecutive_matches
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate bidirectional survivors')
    parser.add_argument('--target', required=True, help='Path to daily3.json dataset')
    parser.add_argument('--seeds', required=True, help='File with seed list (one per line)')
    parser.add_argument('--prng', default='java_lcg', choices=['java_lcg', 'mt19937', 'xoshiro256pp'],
                       help='PRNG type (default: java_lcg)')
    parser.add_argument('--session', choices=['midday', 'evening'], help='Filter by session')
    parser.add_argument('--skip-min', type=int, default=0, help='Minimum skip value to test')
    parser.add_argument('--skip-max', type=int, default=30, help='Maximum skip value to test')
    parser.add_argument('--output', default='validation_results.json', help='Output JSON file')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to display')
    
    args = parser.parse_args()
    
    # Load target dataset
    print(f"\n📂 Loading target dataset: {args.target}")
    target = load_target_dataset(args.target, args.session)
    print(f"✅ Loaded {len(target)} draws")
    
    # Load seeds
    print(f"\n📂 Loading seeds from: {args.seeds}")
    with open(args.seeds, 'r') as f:
        seeds = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(f"✅ Loaded {len(seeds)} seeds")
    
    # Validate all seeds
    results = validate_all_seeds(
        seeds=seeds,
        prng_type=args.prng,
        target=target,
        skip_range=(args.skip_min, args.skip_max)
    )
    
    # Print report
    print_validation_report(results, top_n=args.top_n)
    
    # Save results
    save_results_json(results, args.output)
    
    print("🎯 Validation complete!")


if __name__ == '__main__':
    main()
