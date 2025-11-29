#!/usr/bin/env python3
"""
FINAL test_100k_bidirectional_CORRECT.py — BULLETPROOF
Handles: nulls, missing keys, strings
Matches your production system
"""

import json
import os
from datetime import datetime

# CONFIG
DATA_FILE = 'daily3.json'
SEED_COUNT = 100_000
WINDOW = 30
A, C, M = 1103515245, 12345, 1 << 31
OUTPUT_DIR = 'results'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(f"{OUTPUT_DIR}/json", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/summaries", exist_ok=True)

def load_draws():
    with open(DATA_FILE) as f:
        data = json.load(f)
    draws = []
    for i, e in enumerate(data):
        val = e.get('draw') or e.get('numbers')
        if val is not None:
            try:
                draws.append(int(val))
            except (ValueError, TypeError):
                print(f"Warning: Invalid draw at index {i}: {val}")
        else:
            print(f"Warning: Missing draw at index {i}: {e}")
    if not draws:
        raise ValueError("No valid draws found in daily3.json")
    print(f"Loaded {len(draws)} valid draws")
    return draws

def forward_match(seed, recent):
    state = seed
    for x in recent:
        state = (A * state + C) % M
        if state % 1000 != x:
            return False
    return True

def reverse_match(seed, recent_rev):
    # Brute-force all possible states for each draw
    current_states = {seed}
    for x in recent_rev:
        next_states = set()
        for state in current_states:
            for k in range(M // 1000 + 1):
                full = x + k * 1000
                if full >= M:
                    break
                prev = (full - C) * pow(A, -1, M) % M
                if prev == state:
                    next_states.add(full)
                    break
        if not next_states:
            return False
        current_states = next_states
    return bool(current_states)

def main():
    print("Starting 100k bidirectional sieve...")
    draws = load_draws()
    start = len(draws) - WINDOW
    recent = draws[start:start + WINDOW]
    recent_rev = recent[::-1]

    print(f"Testing {SEED_COUNT} seeds on draws {start} to {start+WINDOW-1}")

    forward = []
    reverse = []
    for seed in range(SEED_COUNT):
        if seed % 10000 == 0:
            print(f"  Tested {seed:,} seeds...")
        if forward_match(seed, recent):
            forward.append(seed)
        if reverse_match(seed, recent_rev):
            reverse.append(seed)

    intersection = set(forward) & set(reverse)
    print(f"Forward: {len(forward)}, Reverse: {len(reverse)}, Intersection: {len(intersection)}")

    # Save JSON
    result = {
        "metadata": {"prng": "java_lcg", "seeds": SEED_COUNT, "window": WINDOW},
        "results": [{
            "survivors": [{"seed": s, "match_rate": 1.0} for s in list(intersection)[:100]],
            "bidirectional_count": len(intersection)
        }]
    }
    path = f"{OUTPUT_DIR}/json/bidirectional_java_lcg_{TIMESTAMP}_top100.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {path}")

if __name__ == "__main__":
    main()
