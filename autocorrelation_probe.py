#!/usr/bin/env python3
"""
Digit Autocorrelation Probe — S119
Validates Team Beta's claim that decorrelation horizon ≈ 8 draws,
explaining why the optimizer discovered window_size=8.

TB's claim:
    - Digit match probability at lag 1-8 > 0.10 (random baseline)
    - Drops to ~0.10 by lag 9+
    - This decorrelation horizon = why sieve window_size=8 is optimal

Usage:
    ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && python3 autocorrelation_probe.py"
"""

import json
import numpy as np
from pathlib import Path

RANDOM_BASELINE = 0.10  # 1/10 for Z10 uniform digits

# ── Load data ──────────────────────────────────────────────────────────────
data_path = Path("daily3.json")
raw = json.load(open(data_path))
if isinstance(raw[0], dict):
    draws = [tuple(map(int, str(r['draw']).zfill(3))) for r in raw]
else:
    draws = [tuple(map(int, str(x).zfill(3))) for x in raw]

print(f"Loaded {len(draws)} draws")
print(f"Random baseline (Z10): {RANDOM_BASELINE:.3f}")
print()

# ── Autocorrelation per digit position and combined ────────────────────────
MAX_LAG = 20

print(f"{'lag':<5} {'hundreds':<12} {'tens':<12} {'ones':<12} {'combined':<12} {'above_baseline'}")
print("-" * 70)

results = []
for lag in range(1, MAX_LAG + 1):
    pos_matches = [0, 0, 0]
    total = 0
    for i in range(len(draws) - lag):
        for p in range(3):
            pos_matches[p] += (draws[i][p] == draws[i + lag][p])
        total += 1

    ac = [m / total for m in pos_matches]
    combined = sum(pos_matches) / (total * 3)
    above = combined - RANDOM_BASELINE
    marker = " <<<" if combined > RANDOM_BASELINE + 0.002 else ""

    print(f"{lag:<5} {ac[0]:<12.4f} {ac[1]:<12.4f} {ac[2]:<12.4f} {combined:<12.4f} {above:+.4f}{marker}")
    results.append((lag, ac[0], ac[1], ac[2], combined))

# ── Summary analysis ───────────────────────────────────────────────────────
print()
combined_vals = [r[4] for r in results]
lags = [r[0] for r in results]

above_baseline = [(l, v) for l, v in zip(lags, combined_vals) if v > RANDOM_BASELINE + 0.002]
peak_lag, peak_val = max(zip(lags, combined_vals), key=lambda x: x[1])

print(f"Peak autocorrelation: lag={peak_lag}, combined={peak_val:.4f}")
print(f"Lags above baseline (+0.002): {[l for l,v in above_baseline]}")
print()

# ── TB's claim validation ──────────────────────────────────────────────────
print("=== TB Claim Validation ===")
print()

# Claim 1: autocorrelation exists above baseline
max_ac = max(combined_vals)
if max_ac > RANDOM_BASELINE + 0.005:
    print(f"✓ CONFIRMED: autocorrelation signal exists (max={max_ac:.4f} vs baseline={RANDOM_BASELINE:.3f})")
else:
    print(f"✗ WEAK/NOT CONFIRMED: max combined={max_ac:.4f} barely above baseline={RANDOM_BASELINE:.3f}")

# Claim 2: decorrelation by lag 8-10
lag8_val = combined_vals[7]   # lag=8
lag9_val = combined_vals[8]   # lag=9
lag15_val = combined_vals[14] # lag=15

print(f"  lag=8  combined={lag8_val:.4f} (above baseline: {lag8_val - RANDOM_BASELINE:+.4f})")
print(f"  lag=9  combined={lag9_val:.4f} (above baseline: {lag9_val - RANDOM_BASELINE:+.4f})")
print(f"  lag=15 combined={lag15_val:.4f} (above baseline: {lag15_val - RANDOM_BASELINE:+.4f})")
print()

if lag8_val > RANDOM_BASELINE + 0.002 and lag15_val <= RANDOM_BASELINE + 0.002:
    print("✓ CONFIRMED: signal present at lag=8, decayed by lag=15")
    print("  → Supports TB's decorrelation horizon ≈ 8 draws")
    print("  → Explains optimizer discovering window_size=8")
elif lag8_val <= RANDOM_BASELINE + 0.002:
    print("? INCONCLUSIVE: signal already decayed by lag=8")
    print("  → Decorrelation horizon may be shorter than 8")
    print("  → window_size=8 may be explained by regime structure, not autocorrelation")
else:
    print("? PARTIAL: signal present at lag=8 but still elevated at lag=15")
    print("  → Decorrelation horizon may be longer than 8")
    print("  → window_size=8 captures partial signal only")

# Claim 3: decay is monotonic (smooth decay = real signal, noisy = artifact)
diffs = [combined_vals[i] - combined_vals[i+1] for i in range(len(combined_vals)-1)]
monotonic_count = sum(1 for d in diffs[:8] if d >= 0)
print()
if monotonic_count >= 6:
    print(f"✓ DECAY IS SMOOTH: {monotonic_count}/8 consecutive lags show decreasing ac")
    print("  → Real signal, not noise artifact")
else:
    print(f"✗ DECAY IS NOISY: only {monotonic_count}/8 consecutive lags decrease")
    print("  → May be noise artifact rather than true memory")

print()
print("=== Mid-day vs Evening Split Preview ===")
if isinstance(raw[0], dict) and 'session' in raw[0]:
    midday = [tuple(map(int, str(r['draw']).zfill(3))) for r in raw if r.get('session') == 'midday']
    evening = [tuple(map(int, str(r['draw']).zfill(3))) for r in raw if r.get('session') == 'evening']
    print(f"Midday draws: {len(midday)}")
    print(f"Evening draws: {len(evening)}")

    # Quick lag-1 comparison
    for name, subset in [("midday", midday), ("evening", evening)]:
        if len(subset) > 10:
            matches = sum(subset[i][p] == subset[i+1][p] for i in range(len(subset)-1) for p in range(3))
            total = (len(subset) - 1) * 3
            print(f"  {name} lag-1 combined ac: {matches/total:.4f}")
else:
    print("No session field in data — cannot split here")
