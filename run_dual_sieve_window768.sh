#!/bin/bash
# run_dual_sieve_window768.sh
# FIXED: Correct CLI for coordinator.py

WINDOW=768
PRNG="java_lcg_hybrid"
SEEDS=1000000000
THRESHOLD=0.01
SKIP_MIN=0
SKIP_MAX=30

echo "DUAL-SIEVE EXECUTION — WINDOW=$WINDOW"
echo "PRNG: $PRNG"
echo "Seed space: $SEEDS"
echo "Threshold: $THRESHOLD"
echo "Skip: $SKIP_MIN to $SKIP_MAX"
echo ""

# === FORWARD SIEVE ===
echo "FORWARD SIEVE..."
python3 coordinator.py \
    daily3.json \
    --prng-type "$PRNG" \
    --method residue_sieve \
    --window-size $WINDOW \
    --skip-min $SKIP_MIN \
    --skip-max $SKIP_MAX \
    --threshold $THRESHOLD \
    --seeds $SEEDS \
    --output "results/forward_window${WINDOW}.json"

# === REVERSE SIEVE (use reverse kernel) ===
echo "REVERSE SIEVE..."
python3 coordinator.py \
    daily3.json \
    --prng-type "${PRNG}_reverse" \
    --method residue_sieve \
    --window-size $WINDOW \
    --skip-min $SKIP_MIN \
    --skip-max $SKIP_MAX \
    --threshold $THRESHOLD \
    --seeds $SEEDS \
    --output "results/reverse_window${WINDOW}.json"

# === INTERSECTION ANALYSIS ===
echo "COMPUTING INTERSECTION..."
python3 -c "
import json, sys
from pathlib import Path

fwd_file = 'results/forward_window${WINDOW}.json'
rev_file = 'results/reverse_window${WINDOW}.json'

if not Path(fwd_file).exists() or not Path(rev_file).exists():
    print('ERROR: One or both sieve results missing.')
    sys.exit(1)

with open(fwd_file) as f:
    fwd = json.load(f)
with open(rev_file) as f:
    rev = json.load(f)

fwd_seeds = set(fwd.get('survivors', []))
rev_seeds = set(rev.get('survivors', []))
intersection = sorted(fwd_seeds & rev_seeds)

print(f'FORWARD survivors: {len(fwd_seeds)}')
print(f'REVERSE survivors: {len(rev_seeds)}')
print(f'INTERSECTION: {len(intersection)}')
if intersection:
    print(f'Survivors: {intersection[:10]}...')

with open('results/intersection_window${WINDOW}.json', 'w') as f:
    json.dump({'intersection': intersection}, f, indent=2)
print('Intersection saved.')
"
