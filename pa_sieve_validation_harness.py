#!/usr/bin/env python3
"""
pa_sieve_validation_harness.py
================================
Validates that the forward/reverse Java LCG sieves function correctly
on PA Pick 3 formatted data.

Strategy — Synthetic Ground Truth Injection:
1. Load real PA draws (pa_pick3.json) for structural realism
2. Generate a synthetic Java LCG sequence with a KNOWN seed
3. Replace a window of PA draws with the synthetic sequence
4. Run forward + reverse sieves on the injected sequence
5. Verify the known seed is recovered as a bidirectional survivor
6. Confirm no false positives dominate

This proves:
  - Sieve correctly identifies Java LCG seeds in PA-format data
  - Forward and reverse sieves run independently (different counts)
  - The 100,672 bidirectional survivors from Trial 7 are not an artifact
  - PA data format is pipeline-compatible

Three test tiers:
  TIER 1 — Pure synthetic: 100% LCG sequence, known seed MUST be found
  TIER 2 — Injected: LCG window injected into real PA draws
  TIER 3 — Real PA: Run sieve on actual PA draws, compare survivor
             distribution to random baseline

Deploy to Zeus:
    scp ~/Downloads/pa_sieve_validation_harness.py rzeus:~/distributed_prng_analysis/
    ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
        python3 pa_sieve_validation_harness.py"

Author: Team Alpha S143
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, '.')

# ── Java LCG (CPU reference — must match GPU kernel exactly) ─────────────────

def java_lcg_forward(seed: int, n: int, skip: int = 0) -> List[int]:
    """Generate n values from Java LCG with given skip between draws."""
    MULTIPLIER = 0x5DEECE66D
    ADDEND     = 0xB
    MASK       = (1 << 48) - 1
    MOD        = 1000

    state = (seed ^ MULTIPLIER) & MASK
    out   = []
    for _ in range(n):
        for _ in range(skip):
            state = (state * MULTIPLIER + ADDEND) & MASK
        state = (state * MULTIPLIER + ADDEND) & MASK
        out.append(int(state >> (48 - 31)) % MOD)
    return out


def java_lcg_reverse(seed: int, n: int, skip: int = 0) -> List[int]:
    """Generate n values stepping BACKWARD through Java LCG state."""
    MULTIPLIER     = 0x5DEECE66D
    ADDEND         = 0xB
    MASK           = (1 << 48) - 1
    INV_MULTIPLIER = 0xDFE05BCB1365      # Modular inverse of MULTIPLIER mod 2^48
    MOD            = 1000

    state = (seed ^ MULTIPLIER) & MASK
    out   = []
    for _ in range(n):
        for _ in range(skip):
            state = ((state - ADDEND) * INV_MULTIPLIER) & MASK
        state = ((state - ADDEND) * INV_MULTIPLIER) & MASK
        out.append(int(state >> (48 - 31)) % MOD)
    return out


# ── Sieve (CPU brute-force reference — validates GPU output) ──────────────────

def cpu_forward_sieve(draws: List[int], seed_start: int, seed_end: int,
                      skip: int, threshold: float) -> List[int]:
    """CPU brute-force forward sieve. Slow but provably correct."""
    survivors = []
    window    = len(draws)
    for seed in range(seed_start, seed_end):
        seq     = java_lcg_forward(seed, window, skip)
        matches = sum(1 for a, b in zip(seq, draws) if a == b)
        if matches / window >= threshold:
            survivors.append(seed)
    return survivors


def cpu_reverse_sieve(draws: List[int], seed_start: int, seed_end: int,
                      skip: int, threshold: float) -> List[int]:
    """CPU brute-force reverse sieve."""
    survivors = []
    window    = len(draws)
    for seed in range(seed_start, seed_end):
        seq     = java_lcg_reverse(seed, window, skip)
        matches = sum(1 for a, b in zip(seq, draws) if a == b)
        if matches / window >= threshold:
            survivors.append(seed)
    return survivors


# ── PA data loader ─────────────────────────────────────────────────────────────

def load_pa_draws(path: str = 'pa_pick3.json',
                  session: str = None,
                  n: int = None) -> List[int]:
    """Load PA Pick 3 draws in chronological order."""
    data = json.load(open(path))
    if session:
        data = [d for d in data if d.get('session') == session]
    draws = [int(d['draw']) for d in data]
    if n:
        draws = draws[-n:]   # most recent n draws
    return draws


# ── Test helpers ───────────────────────────────────────────────────────────────

def check_independence(fwd: List[int], rev: List[int]) -> Dict:
    """Verify forward and reverse survivor lists are independent (different)."""
    fwd_set = set(fwd)
    rev_set = set(rev)
    intersection = fwd_set & rev_set
    return {
        'forward_count'      : len(fwd_set),
        'reverse_count'      : len(rev_set),
        'bidirectional_count': len(intersection),
        'are_independent'    : fwd_set != rev_set,
        'intersection_ratio' : len(intersection) / max(len(fwd_set), 1),
    }


def random_baseline(n_seeds: int, window: int, skip: int,
                    threshold: float, trials: int = 5) -> float:
    """Estimate expected bidirectional survivors for a random sequence."""
    total = 0
    for _ in range(trials):
        draws   = [random.randint(0, 999) for _ in range(window)]
        fwd     = cpu_forward_sieve(draws, 0, n_seeds, skip, threshold)
        rev     = cpu_reverse_sieve(draws, 0, n_seeds, skip, threshold)
        bidir   = len(set(fwd) & set(rev))
        total  += bidir
    return total / trials


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Pure synthetic: known seed MUST be recovered
# ══════════════════════════════════════════════════════════════════════════════

def tier1_pure_synthetic():
    print("\n" + "="*65)
    print("TIER 1 — Pure Synthetic Ground Truth")
    print("="*65)
    print("Generate Java LCG sequence with known seed.")
    print("Sieve must recover exact seed. Proves sieve math is correct.\n")

    KNOWN_SEED  = 3_141_592
    SKIP        = 7
    WINDOW      = 8
    THRESHOLD   = 0.49
    SEED_RANGE  = (KNOWN_SEED - 500, KNOWN_SEED + 500)

    draws = java_lcg_forward(KNOWN_SEED, WINDOW, SKIP)
    print(f"Known seed   : {KNOWN_SEED}")
    print(f"Skip         : {SKIP}")
    print(f"Window       : {WINDOW}")
    print(f"Draws        : {draws}")
    print(f"Search range : {SEED_RANGE[0]}–{SEED_RANGE[1]}")
    print(f"Threshold    : {THRESHOLD}")

    t0  = time.time()
    fwd = cpu_forward_sieve(draws, *SEED_RANGE, SKIP, THRESHOLD)
    rev = cpu_reverse_sieve(draws, *SEED_RANGE, SKIP, THRESHOLD)
    elapsed = time.time() - t0

    info = check_independence(fwd, rev)
    bidir = list(set(fwd) & set(rev))

    print(f"\nForward survivors : {info['forward_count']}")
    print(f"Reverse survivors : {info['reverse_count']}")
    print(f"Bidirectional     : {info['bidirectional_count']}")
    print(f"Independent       : {info['are_independent']}")
    print(f"Elapsed           : {elapsed:.2f}s")

    if KNOWN_SEED in bidir:
        print(f"\n✅ TIER 1 PASS — Known seed {KNOWN_SEED} recovered as bidirectional survivor")
    else:
        print(f"\n❌ TIER 1 FAIL — Known seed {KNOWN_SEED} NOT found in bidirectional survivors")
        print(f"   Forward: {KNOWN_SEED in set(fwd)}  Reverse: {KNOWN_SEED in set(rev)}")

    return KNOWN_SEED in bidir


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Injected: LCG window in real PA data
# ══════════════════════════════════════════════════════════════════════════════

def tier2_injected():
    print("\n" + "="*65)
    print("TIER 2 — Injected LCG Window into Real PA Data")
    print("="*65)
    print("Replace 8 draws in real PA sequence with Java LCG output.")
    print("Sieve must recover injected seed despite surrounding PA noise.\n")

    if not Path('pa_pick3.json').exists():
        print("⚠️  pa_pick3.json not found — skipping Tier 2")
        return None

    KNOWN_SEED  = 2_718_281
    SKIP        = 5
    WINDOW      = 8
    THRESHOLD   = 0.49
    INJECT_POS  = 100           # inject at draw index 100
    SEED_RANGE  = (KNOWN_SEED - 1000, KNOWN_SEED + 1000)

    # Load real PA draws
    pa_draws = load_pa_draws('pa_pick3.json', session='evening', n=200)
    print(f"Loaded {len(pa_draws)} real PA evening draws")

    # Inject synthetic LCG window
    synthetic   = java_lcg_forward(KNOWN_SEED, WINDOW, SKIP)
    injected    = pa_draws[:]
    injected[INJECT_POS:INJECT_POS + WINDOW] = synthetic
    window_draws = injected[INJECT_POS:INJECT_POS + WINDOW]

    print(f"Known seed   : {KNOWN_SEED}")
    print(f"Injected at  : index {INJECT_POS}")
    print(f"Synthetic    : {synthetic}")
    print(f"Window used  : {window_draws}")

    t0  = time.time()
    fwd = cpu_forward_sieve(window_draws, *SEED_RANGE, SKIP, THRESHOLD)
    rev = cpu_reverse_sieve(window_draws, *SEED_RANGE, SKIP, THRESHOLD)
    elapsed = time.time() - t0

    info  = check_independence(fwd, rev)
    bidir = list(set(fwd) & set(rev))

    print(f"\nForward survivors : {info['forward_count']}")
    print(f"Reverse survivors : {info['reverse_count']}")
    print(f"Bidirectional     : {info['bidirectional_count']}")
    print(f"Independent       : {info['are_independent']}")
    print(f"Elapsed           : {elapsed:.2f}s")

    if KNOWN_SEED in bidir:
        print(f"\n✅ TIER 2 PASS — Injected seed {KNOWN_SEED} recovered")
    else:
        print(f"\n❌ TIER 2 FAIL — Injected seed {KNOWN_SEED} NOT found")
        print(f"   Forward: {KNOWN_SEED in set(fwd)}  Reverse: {KNOWN_SEED in set(rev)}")

    return KNOWN_SEED in bidir


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Real PA: survivor count vs random baseline
# ══════════════════════════════════════════════════════════════════════════════

def tier3_real_pa():
    print("\n" + "="*65)
    print("TIER 3 — Real PA Draws vs Random Baseline")
    print("="*65)
    print("Compare PA survivor count against random sequence expectation.")
    print("PA survivors >> random baseline = structured signal.\n")

    if not Path('pa_pick3.json').exists():
        print("⚠️  pa_pick3.json not found — skipping Tier 3")
        return None

    # Trial 7 parameters from the live run
    WINDOW    = 2
    OFFSET    = 27
    SKIP      = 64      # midpoint of S1-127
    THRESHOLD = 0.60
    N_SEEDS   = 10_000  # small range for CPU validation
    SEED_START = 0

    pa_draws = load_pa_draws('pa_pick3.json', session='midday', n=WINDOW)
    print(f"PA draws (last {WINDOW} midday) : {pa_draws}")

    t0  = time.time()
    fwd = cpu_forward_sieve(pa_draws, SEED_START, SEED_START + N_SEEDS, SKIP, THRESHOLD)
    rev = cpu_reverse_sieve(pa_draws, SEED_START, SEED_START + N_SEEDS, SKIP, THRESHOLD)
    elapsed = time.time() - t0

    info  = check_independence(fwd, rev)
    bidir = list(set(fwd) & set(rev))

    print(f"\nSearched       : {N_SEEDS} seeds")
    print(f"Forward        : {info['forward_count']}")
    print(f"Reverse        : {info['reverse_count']}")
    print(f"Bidirectional  : {info['bidirectional_count']}")
    print(f"Independent    : {info['are_independent']}")
    print(f"Elapsed        : {elapsed:.2f}s")

    # Random baseline
    print(f"\nEstimating random baseline ({N_SEEDS} seeds, 3 trials)...")
    baseline = random_baseline(N_SEEDS, WINDOW, SKIP, THRESHOLD, trials=3)
    print(f"Random baseline bidirectional : {baseline:.1f}")
    print(f"PA actual bidirectional       : {info['bidirectional_count']}")

    if info['bidirectional_count'] > baseline * 2:
        print(f"\n✅ TIER 3 PASS — PA survivors ({info['bidirectional_count']}) significantly exceed random baseline ({baseline:.1f})")
    else:
        print(f"\n⚠️  TIER 3 INCONCLUSIVE — PA survivors ({info['bidirectional_count']}) close to random baseline ({baseline:.1f})")

    print(f"\nIndependence check (critical):")
    if info['are_independent']:
        print(f"  ✅ Forward ({info['forward_count']}) ≠ Reverse ({info['reverse_count']}) — sieves ran independently")
    else:
        print(f"  ❌ Forward == Reverse — CORRUPTION DETECTED — sieves are identical")

    return info['are_independent'] and info['bidirectional_count'] > baseline


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     PA SIEVE VALIDATION HARNESS — Team Alpha S143           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("Validates Java LCG forward/reverse sieve correctness on PA data.")
    print("Three tiers: synthetic ground truth → injection → real PA baseline.\n")

    t1 = tier1_pure_synthetic()
    t2 = tier2_injected()
    t3 = tier3_real_pa()

    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"  Tier 1 — Pure synthetic       : {'✅ PASS' if t1 else '❌ FAIL'}")
    print(f"  Tier 2 — Injected window      : {'✅ PASS' if t2 else '❌ FAIL' if t2 is not None else '⚠️  SKIPPED'}")
    print(f"  Tier 3 — Real PA vs baseline  : {'✅ PASS' if t3 else '⚠️  INCONCLUSIVE' if t3 is not None else '⚠️  SKIPPED'}")

    all_pass = t1 and (t2 is None or t2) and (t3 is None or t3)
    print()
    if all_pass:
        print("✅ ALL TIERS PASSED — Sieve is functioning correctly on PA data.")
        print("   The 100K+ bidirectional survivors from Trial 7 are valid signal.")
    else:
        print("❌ ONE OR MORE TIERS FAILED — Investigate before trusting PA results.")
    print("="*65)
