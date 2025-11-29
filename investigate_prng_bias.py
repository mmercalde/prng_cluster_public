#!/usr/bin/env python3
"""
FINAL FIXED — NO FLOAT TOLERANCE ERRORS
"""

import json
import numpy as np
from collections import Counter, defaultdict
from scipy import stats


def load_daily3_data(json_file='daily3.json'):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    numbers = []
    for entry in json_data:
        if 'numbers' in entry:
            numbers.append(int(entry['numbers']))
        elif 'draw' in entry:
            numbers.append(int(entry['draw']))
    return np.array(numbers)


def test_1_residue_class_analysis(numbers):
    print("\n" + "="*70)
    print("TEST 1: RESIDUE CLASS ANALYSIS")
    print("="*70)
    for mod in [2,4,8,16,32,64,128,256,512,1024]:
        c = defaultdict(int)
        for n in numbers: c[n%mod] += 1
        obs = [c[i] for i in range(mod)]
        exp = [len(numbers)/mod] * mod
        chi2, p = stats.chisquare(obs, exp)
        if p < 0.01:
            print(f"  mod {mod:4d}: χ²={chi2:8.2f}, p={p:.6f} ***")
        else:
            print(f"  mod {mod:4d}: χ²={chi2:8.2f}, p={p:.6f}")


def test_2_digit_position_bias(numbers):
    print("\n" + "="*70)
    print("TEST 2: DIGIT POSITION BIAS")
    print("="*70)
    h = [n//100 for n in numbers]
    t = [(n//10)%10 for n in numbers]
    o = [n%10 for n in numbers]
    def show(name, d):
        print(f"\n{name}:")
        c = Counter(d)
        e = len(numbers)/10
        for i in range(10):
            ex = (c[i]/e - 1)*100
            print(f"  {i}: {c[i]:5d} ({ex:+6.2f}%)")
        chi2, p = stats.chisquare([c[i] for i in range(10)], [e]*10)
        print(f"  χ²={chi2:.2f}, p={p:.6f}")
    show("Hundreds", h); show("Tens", t); show("Ones", o)


def test_3_range_bias_breakdown(numbers):
    print("\n" + "="*70)
    print("TEST 3: RANGE BIAS")
    print("="*70)
    ranges = [(0,99,"0-99"), (100,199,"100-199"), (200,295,"200-295"),
              (296,399,"296-399"), (400,499,"400-499"), (500,599,"500-599"),
              (600,699,"600-699"), (700,799,"700-799"), (800,899,"800-899"),
              (900,999,"900-999")]
    N = len(numbers)
    for s,e,l in ranges:
        c = sum(1 for x in numbers if s<=x<=e)
        exp = N*(e-s+1)/1000
        ex = (c/exp - 1)*100
        print(f"{l:<12} {c:6d} {exp:8.2f} {ex:+9.2f}%")
    c1 = sum(1 for x in numbers if x <= 295)
    e1 = N*296/1000
    print(f"{'0-295':<12} {c1:6d} {e1:8.2f} {(c1/e1-1)*100:+9.2f}%")


def test_4_lcg_parameter_detection(numbers):
    print("\n" + "="*70)
    print("TEST 4: LCG")
    print("="*70)
    params = [(1103515245,12345,2**31,"glibc"), (214013,2531011,2**31,"MSVC")]
    for a,c,m,nm in params:
        x = 1
        sim = [(x:=(a*x+c)%m)%1000 for _ in numbers]
        r = sum(1 for v in sim if v<=295) / (len(sim)*296/1000)
        print(f"  {nm}: ratio={r:.6f}")


def test_5_autocorrelation_and_periodicity(numbers):
    print("\n" + "="*70)
    print("TEST 5: AUTOCORR")
    print("="*70)
    if len(numbers)>1:
        r = np.corrcoef(numbers[:-1], numbers[1:])[0,1]
        print(f"  Lag-1: {r:.6f}")


# FINAL FIXED TEST 6 — INTEGER DISTRIBUTION, EXACT SUM
def test_6_bit_pattern_analysis(numbers):
    print("\n" + "="*70)
    print("TEST 6: BIT PATTERN ANALYSIS")
    print("="*70)
    
    N = len(numbers)
    bit_counts = [0] * 10
    for num in numbers:
        for b in range(10):
            if num & (1 << b):
                bit_counts[b] += 1
    
    total_ones = sum(bit_counts)
    exp_base = total_ones // 10
    rem = total_ones % 10
    expected = [exp_base] * 10
    for i in range(rem):
        expected[i] += 1

    print(f"\nBit balance (N={N}, total 1-bits={total_ones}):")
    print(f"{'Bit':<5} {'Ones':>6} {'Exp':>8} {'Excess %':>10}")
    print("-" * 35)
    for b in range(10):
        c, e = bit_counts[b], expected[b]
        excess = (c - e) / e * 100 if e > 0 else 0
        stars = "***" if abs(excess) > 2 else ""
        print(f"{b:<5} {c:6d} {e:8d} {excess:+9.2f}% {stars}")
    
    chi2, p = stats.chisquare(bit_counts, expected)
    print(f"\nχ² = {chi2:.2f}, p-value = {p:.6f}")


def main():
    print("="*70)
    print("FINAL PRNG BIAS TEST")
    print("="*70)
    numbers = load_daily3_data('daily3.json')
    print(f"Loaded {len(numbers)} numbers\n")
    test_1_residue_class_analysis(numbers)
    test_2_digit_position_bias(numbers)
    test_3_range_bias_breakdown(numbers)
    test_4_lcg_parameter_detection(numbers)
    test_5_autocorrelation_and_periodicity(numbers)
    test_6_bit_pattern_analysis(numbers)
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
