#!/usr/bin/env python3
"""
Enhanced Synthetic Pattern Generator - Java LCG + Statistical Biases
====================================================================
Generates lottery data with:
1. Java LCG mathematical patterns (constant + variable skip)
2. Subtle statistical biases (frequency, marker correlation, etc.)

This creates data that the bidirectional sieve can detect!
"""
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import Counter

class JavaLCG:
    """Java Linear Congruential Generator (java.util.Random)"""
    def __init__(self, seed: int):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        self.a = 25214903917  # Java LCG multiplier
        self.c = 11           # Java LCG addend
        self.m = 1 << 48      # Java LCG modulus (2^48)
    
    def next(self, bits: int = 32) -> int:
        """Generate next random number"""
        self.seed = (self.a * self.seed + self.c) & (self.m - 1)
        return int(self.seed >> (48 - bits))
    
    def next_int(self, bound: int) -> int:
        """Generate next int in range [0, bound)"""
        if (bound & -bound) == bound:  # Power of 2
            return int((bound * self.next(31)) >> 31)
        
        bits = self.next(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self.next(31)
            val = bits % bound
        return val

class EnhancedSyntheticGenerator:
    """Generate synthetic lottery data with Java LCG + statistical biases"""
    
    def __init__(self, seed: int = 42, mod: int = 1000):
        self.seed = seed
        self.mod = mod
        self.rng = np.random.RandomState(seed)
        
        # Java LCG instances for constant and variable skip
        self.lcg_constant = JavaLCG(seed)
        self.lcg_variable = JavaLCG(seed + 1000)
        
        # Skip pattern parameters
        self.constant_skip_values = [3, 5, 7]  # Mix of constant skips
        self.variable_skip_base = 2
        self.variable_skip_multiplier = 3
        
        # Mix ratio: 70% LCG-based, 30% pure random with biases
        self.lcg_mix_ratio = 0.7
        
        # Statistical bias parameters (from original)
        self.frequency_bias_number = 147
        self.frequency_bias_ratio = 1.8
        self.marker_numbers = [273, 581, 892]
        self.marker_correlation = 0.3
        self.regime_period = 200
        self.regime_strength = 0.3
        self.modulo_bias = 7
        self.modulo_strength = 0.25
        self.drift_rate = 0.0005
        
        # Skip mode tracking
        self.current_skip_mode = "constant"  # Start with constant
        self.skip_mode_change_interval = 500  # Change every 500 draws
        self.current_constant_skip = self.rng.choice(self.constant_skip_values)
    
    def _get_lcg_number_with_skip(self, draw_index: int, use_constant_skip: bool) -> int:
        """Generate number using Java LCG with skip pattern"""
        if use_constant_skip:
            # Constant skip: always skip same number of states
            for _ in range(self.current_constant_skip):
                self.lcg_constant.next()
            return self.lcg_constant.next_int(self.mod)
        else:
            # Variable skip: skip depends on previous output
            prev_output = self.lcg_variable.next_int(self.mod)
            skip_count = self.variable_skip_base + (prev_output % self.variable_skip_multiplier)
            for _ in range(skip_count):
                self.lcg_variable.next()
            return self.lcg_variable.next_int(self.mod)
    
    def _apply_frequency_bias(self, number: int) -> int:
        """Apply frequency bias to a number"""
        if self.rng.random() < (self.frequency_bias_ratio - 1.0) / self.frequency_bias_ratio:
            return self.frequency_bias_number
        return number
    
    def _apply_marker_correlation(self, number: int, has_marker: bool) -> int:
        """Apply marker correlation"""
        if has_marker and self.rng.random() < self.marker_correlation:
            return self.rng.choice(self.marker_numbers)
        return number
    
    def _apply_regime_effect(self, number: int, draw_index: int) -> int:
        """Apply regime changes"""
        regime = (draw_index // self.regime_period) % 3
        if self.rng.random() < self.regime_strength:
            if regime == 0:
                return int(number * 0.9) % self.mod
            elif regime == 1:
                return int(number * 1.1) % self.mod
        return number
    
    def _apply_modulo_bias(self, number: int) -> int:
        """Apply modulo bias"""
        if self.rng.random() < self.modulo_strength:
            return (number // self.modulo_bias * self.modulo_bias + 
                    self.rng.randint(0, self.modulo_bias)) % self.mod
        return number
    
    def _apply_drift(self, number: int, draw_index: int) -> int:
        """Apply slow drift"""
        drift = int(draw_index * self.drift_rate * self.mod)
        return (number + drift) % self.mod
    
    def generate_draw_number(self, draw_index: int) -> int:
        """Generate a single draw number with LCG + biases"""
        
        # Determine skip mode (changes periodically)
        if draw_index > 0 and draw_index % self.skip_mode_change_interval == 0:
            self.current_skip_mode = "variable" if self.current_skip_mode == "constant" else "constant"
            if self.current_skip_mode == "constant":
                self.current_constant_skip = self.rng.choice(self.constant_skip_values)
        
        # Mix LCG with pure random
        if self.rng.random() < self.lcg_mix_ratio:
            # Use LCG with skip pattern
            use_constant = (self.current_skip_mode == "constant")
            number = self._get_lcg_number_with_skip(draw_index, use_constant)
        else:
            # Use pure random
            number = self.rng.randint(0, self.mod)
        
        # Apply statistical biases (these add noise but maintain LCG core pattern)
        has_marker = (number in self.marker_numbers)
        
        # Apply biases with lower probability to preserve LCG pattern
        if self.rng.random() < 0.3:  # Only 30% of the time
            number = self._apply_frequency_bias(number)
        if self.rng.random() < 0.2:
            number = self._apply_marker_correlation(number, has_marker)
        if self.rng.random() < 0.2:
            number = self._apply_regime_effect(number, draw_index)
        if self.rng.random() < 0.2:
            number = self._apply_modulo_bias(number)
        
        # Always apply drift (very subtle)
        number = self._apply_drift(number, draw_index)
        
        return int(number) % self.mod
    
    def generate_dataset(self, num_draws: int = 5000) -> List[Dict]:
        """Generate complete synthetic dataset"""
        dataset = []
        start_date = datetime(2020, 1, 1)
        
        print(f"Generating {num_draws} draws with Java LCG + Statistical Biases")
        print(f"\nJava LCG Parameters:")
        print(f" - Multiplier (a): 25214903917")
        print(f" - Addend (c): 11")
        print(f" - Modulus (m): 2^48")
        print(f" - Constant skip values: {self.constant_skip_values}")
        print(f" - Variable skip: base={self.variable_skip_base}, multiplier={self.variable_skip_multiplier}")
        print(f" - Skip mode changes every {self.skip_mode_change_interval} draws")
        print(f" - LCG mix ratio: {self.lcg_mix_ratio*100:.0f}%")
        print(f"\nStatistical Biases (applied to {(1-self.lcg_mix_ratio)*100:.0f}% + noise):")
        print(f" - Frequency bias: {self.frequency_bias_number} @ {self.frequency_bias_ratio}x")
        print(f" - Markers: {self.marker_numbers} (correlation: {self.marker_correlation})")
        print(f" - Regime changes: every {self.regime_period} draws")
        print(f" - Modulo bias: %{self.modulo_bias}")
        print()
        
        for i in range(num_draws):
            draw_date = start_date + timedelta(days=i)
            session = "midday" if i % 2 == 0 else "evening"
            draw_number = self.generate_draw_number(i)
            
            dataset.append({
                'draw_id': int(i + 1),
                'date': draw_date.strftime('%Y-%m-%d'),
                'session': session,
                'draw': draw_number
            })
        
        return dataset
    
    def analyze_patterns(self, dataset: List[Dict]) -> Dict:
        """Analyze the generated dataset"""
        all_numbers = [draw['draw'] for draw in dataset]
        freq = Counter(all_numbers)
        
        # Frequency analysis
        bias_count = freq[self.frequency_bias_number]
        avg_count = np.mean(list(freq.values()))
        actual_bias_ratio = bias_count / avg_count if avg_count > 0 else 0
        
        # Modulo analysis
        modulo_matches = sum(1 for n in all_numbers if n % self.modulo_bias == 0)
        modulo_rate = modulo_matches / len(all_numbers)
        expected_modulo_rate = 1.0 / self.modulo_bias
        
        # LCG pattern detection (simple autocorrelation test)
        diffs = [all_numbers[i+1] - all_numbers[i] for i in range(len(all_numbers)-1)]
        diff_variance = np.var(diffs)
        
        return {
            'total_draws': len(dataset),
            'unique_numbers': len(freq),
            'frequency_bias_ratio': actual_bias_ratio,
            'modulo_bias_ratio': modulo_rate / expected_modulo_rate,
            'diff_variance': diff_variance,
            'has_lcg_pattern': diff_variance < np.var(all_numbers) * 0.5  # Heuristic
        }

def main():
    parser = argparse.ArgumentParser(description='Generate Java LCG + biased lottery data')
    parser.add_argument('--draws', type=int, default=5000, help='Number of draws')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mod', type=int, default=1000, help='Modulus for draw numbers')
    parser.add_argument('--output', type=str, default='synthetic_lottery.json', help='Output file')
    parser.add_argument('--analyze', action='store_true', help='Analyze generated patterns')
    args = parser.parse_args()
    
    # Generate data
    generator = EnhancedSyntheticGenerator(seed=args.seed, mod=args.mod)
    dataset = generator.generate_dataset(num_draws=args.draws)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Generated {len(dataset)} draws")
    print(f"✅ Saved to: {args.output}")
    
    # Analyze if requested
    if args.analyze:
        print("\n" + "="*70)
        print("PATTERN ANALYSIS")
        print("="*70)
        analysis = generator.analyze_patterns(dataset)
        print(f"\nTotal draws: {analysis['total_draws']}")
        print(f"Unique numbers: {analysis['unique_numbers']}")
        print(f"Frequency bias ratio: {analysis['frequency_bias_ratio']:.2f}x")
        print(f"Modulo bias ratio: {analysis['modulo_bias_ratio']:.2f}x")
        print(f"Diff variance: {analysis['diff_variance']:.2f}")
        print(f"Has LCG pattern: {analysis['has_lcg_pattern']}")
        print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
