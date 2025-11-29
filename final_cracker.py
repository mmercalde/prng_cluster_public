#!/usr/bin/env python3
"""
FINAL LCG CRACKER — 100% YOUR SYSTEM, NO CRASH
Reads: results/json/*_top100.json
Handles: nulls, strings, missing keys
November 5, 2025 — Production
"""

import json
import glob
from typing import List, Tuple, Optional

# CONFIG
DATA_FILE = 'daily3.json'
RESULTS_DIR = 'results/json/'
WINDOW_SIZE = 50
M_CANDIDATES = [2**16, 2**31-1, 2**31, 2**32, 10007, 32749, 65521]

def load_draws() -> List[int]:
    with open(DATA_FILE) as f:
        data = json.load(f)
    draws = []
    for i, e in enumerate(data):
        val = e.get('draw') or e.get('numbers')
        if val is not None:
            try:
                draws.append(int(val))
            except (ValueError, TypeError) as err:
                print(f"Warning: Bad draw at index {i}: {val} → {err}")
        else:
            print(f"Warning: Missing draw at index {i}: {e}")
    if not draws:
        raise ValueError("No valid draws in daily3.json")
    print(f"Loaded {len(draws)} valid draws")
    return draws[-WINDOW_SIZE:]

def load_survivors(min_rate: float = 0.95) -> List[int]:
    survivors = []
    for path in glob.glob(f"{RESULTS_DIR}*_top100.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            for run in data.get('results', []):
                for s in run.get('survivors', []):
                    seed = s.get('seed')
                    rate = s.get('match_rate', 0)
                    if seed is not None and rate >= min_rate:
                        survivors.append(int(seed))
        except Exception as e:
            print(f"Error reading {path}: {e}")
    unique = list(set(survivors))
    print(f"Found {len(unique)} survivors (match_rate >= {min_rate})")
    return unique

def expand_state(x: int, m: int) -> List[int]:
    return [x + k*1000 for k in range(m//1000 + 1) if x + k*1000 < m]

def crack_lcg(draws: List[int], seeds: List[int]) -> Optional[Tuple[int, int, int, int]]:
    for m in M_CANDIDATES:
        print(f"  Testing m = {m}")
        state_map = {i: expand_state(x, m) for i, x in enumerate(draws)}
        for seed in seeds:
            if seed >= m or seed % 1000 != draws[0]: continue
            chain = [seed]
            for i in range(1, len(draws)):
                next_s = [s for s in state_map[i] if s > chain[-1]]
                if not next_s: break
                chain.append(next_s[0])
            else:
                s0, s1, s2 = chain[:3]
                d1 = (s1 - s0) % m
                d2 = (s2 - s1) % m
                if d1 == 0: continue
                try:
                    inv = pow(d1, -1, m)
                    a = (d2 * inv) % m
                    c = (s1 - a * s0) % m
                except: continue
                state = seed
                if all((state := (a * state + c) % m) % 1000 == draws[i] for i in range(len(draws))):
                    return a, c, m, seed
    return None

def main():
    print("LCG CRACKER — YOUR SYSTEM v3")
    draws = load_draws()
    survivors = load_survivors(0.95)
    if not survivors:
        print("Run your sieve first!")
        return
    result = crack_lcg(draws, survivors)
    if result:
        a, c, m, seed = result
        state = seed
        for _ in draws: state = (a * state + c) % m
        pred = state % 1000
        print(f"\nCRACKED!")
        print(f"a = {a}, c = {c}, m = {m}, seed = {seed}")
        print(f"NEXT DRAW: {pred}")
    else:
        print("No LCG. Try --prng-type lcg32")

if __name__ == "__main__":
    main()
