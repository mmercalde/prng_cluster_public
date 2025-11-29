#!/usr/bin/env python3
"""Retroactive Three-Lane Filter"""
import json
import argparse
import time

class JavaLCG:
    def __init__(self, seed):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    
    def next(self, bits=32):
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return int(self.state >> (48 - bits))
    
    def next_int(self, bound):
        if (bound & -bound) == bound:
            return int((bound * self.next(31)) >> 31)
        bits = self.next(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self.next(31)
            val = bits % bound
        return val

def load_survivors(filepath):
    with open(filepath) as f:
        data = json.load(f)
    survivors = []
    for result in data.get('results', []):
        for s in result.get('survivors', []):
            if isinstance(s, int):
                survivors.append(s)
            elif isinstance(s, dict) and 'seed' in s:
                survivors.append(s['seed'])
    print(f"Loaded {len(survivors):,} from {filepath}")
    return survivors

def generate_draws(seed, count, skip):
    prng = JavaLCG(seed)
    draws = []
    for _ in range(count):
        draws.append(prng.next_int(1000))
        for _ in range(skip):
            prng.next_int(1000)
    return draws

def test_seed(seed, targets, window, offset, skip_min, skip_max, threshold):
    target_window = targets[offset:offset+window]
    for skip in range(skip_min, skip_max + 1):
        gen = generate_draws(seed, window, skip)
        matches = sum(1 for g, t in zip(gen, target_window)
                     if g%1000==t%1000 and g%125==t%125 and g%8==t%8)
        if matches / window >= threshold:
            return {'seed': seed, 'skip': skip, 'matches': matches, 'rate': matches/window}
    return None

parser = argparse.ArgumentParser()
parser.add_argument('--forward', required=True)
parser.add_argument('--reverse', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('--output', default='filtered.json')
parser.add_argument('--window-size', type=int, default=244)
parser.add_argument('--offset', type=int, default=139)
parser.add_argument('--skip-min', type=int, default=3)
parser.add_argument('--skip-max', type=int, default=29)
parser.add_argument('--threshold', type=float, default=0.90)
args = parser.parse_args()

print("="*70)
print("THREE-LANE FILTER")
print("="*70)

fwd = load_survivors(args.forward)
rev = load_survivors(args.reverse)
with open(args.target) as f:
    draws = [d['draw'] for d in json.load(f)]

print(f"\nFiltering {len(fwd):,} forward survivors...")
start = time.time()
fwd_filtered = []
for i, seed in enumerate(fwd):
    if (i+1) % 10000 == 0:
        print(f"  {i+1:,}/{len(fwd):,}")
    r = test_seed(seed, draws, args.window_size, args.offset, args.skip_min, args.skip_max, args.threshold)
    if r:
        fwd_filtered.append(r)
print(f"✅ {len(fwd):,} → {len(fwd_filtered):,} ({time.time()-start:.1f}s)")

print(f"\nFiltering {len(rev):,} reverse survivors...")
start = time.time()
rev_filtered = []
for i, seed in enumerate(rev):
    if (i+1) % 10000 == 0:
        print(f"  {i+1:,}/{len(rev):,}")
    r = test_seed(seed, draws, args.window_size, args.offset, args.skip_min, args.skip_max, args.threshold)
    if r:
        rev_filtered.append(r)
print(f"✅ {len(rev):,} → {len(rev_filtered):,} ({time.time()-start:.1f}s)")

fwd_set = {r['seed'] for r in fwd_filtered}
rev_set = {r['seed'] for r in rev_filtered}
bi = fwd_set & rev_set

print(f"\nBIDIRECTIONAL: {len(bi):,}")
with open(args.output, 'w') as f:
    json.dump({'bidirectional_seeds': sorted(bi), 'forward': fwd_filtered, 'reverse': rev_filtered}, f, indent=2)
print(f"Saved to {args.output}")
