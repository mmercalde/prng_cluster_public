#!/usr/bin/env python3
import json
from collections import Counter

def check():
    print("CHECKING SURVIVORS...\n")
    
    # Load files
    fw = json.load(open('results/window_opt_forward_244_139.json'))
    rv = json.load(open('results/window_opt_reverse_244_139.json'))
    
    def extract(d):
        seeds = set()
        for r in d.get('results', []):
            for s in r.get('survivors', []):
                if isinstance(s, dict) and 'seed' in s:
                    seeds.add(s['seed'])
        return seeds
    
    fwd_seeds = extract(fw)
    rev_seeds = extract(rv)
    survivors = list(fwd_seeds & rev_seeds)
    
    print(f"Forward survivors: {len(fwd_seeds):,}")
    print(f"Reverse survivors: {len(rev_seeds):,}")
    print(f"Intersection (your 27,902): {len(survivors):,}")
    
    # Check for duplicates
    if len(survivors) == len(set(survivors)):
        print("No duplicate seeds")
    else:
        print("DUPLICATES FOUND")
    
    # Sample 5 seeds
    print("\nSample seeds:")
    for s in list(survivors)[:5]:
        print(f"  {s}")
    
    # Check if seeds are in range
    min_seed = min(survivors)
    max_seed = max(survivors)
    print(f"\nSeed range: {min_seed} → {max_seed}")
    
    # Check if forward/reverse files exist
    if len(fwd_seeds) == 0 or len(rev_seeds) == 0:
        print("\nONE FILE IS EMPTY — THIS IS THE PROBLEM")
    else:
        print("\nBoth files have data")
    
    # Check recent draws
    with open('daily3.json') as f:
        data = json.load(f)
    recent = [d['draw'] for d in data[-10:]]
    print(f"\nRecent 10 draws: {recent}")
    
    print("\nDIAGNOSIS COMPLETE")
    if len(survivors) > 1000:
        print("Survivors look valid")
    else:
        print("TOO FEW SURVIVORS — RE-RUN SIEVE")

check()
