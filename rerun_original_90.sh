#!/bin/bash
# rerun_90_original.sh
# EXACT REPRODUCTION — 0.01 THRESHOLD — CORRECT PARSING

echo "RERUNNING ORIGINAL TEST — 90 SURVIVORS"
echo "THRESHOLD: 0.01"
echo "PARSING: 'survivors' KEY"
echo ""

# === FORWARD SIEVE ===
echo "FORWARD SIEVE..."
python3 coordinator.py daily3.json \
    --prng-type java_lcg \
    --method residue_sieve \
    --window-size 768 \
    --skip-min 0 --skip-max 30 \
    --threshold 0.01 \
    --seeds 1000000000 \
    --output results/forward_90_orig.json

# === REVERSE SIEVE ===
echo "REVERSE SIEVE..."
python3 coordinator.py daily3.json \
    --prng-type java_lcg_reverse \
    --method residue_sieve \
    --window-size 768 \
    --skip-min 0 --skip-max 30 \
    --threshold 0.01 \
    --seeds 1000000000 \
    --output results/reverse_90_orig.json

# === INTERSECTION (PARSE 'survivors' KEY) ===
echo "COMPUTING INTERSECTION..."
python3 -c "
import json
from pathlib import Path

fwd_file = 'results/forward_90_orig.json'
rev_file = 'results/reverse_90_orig.json'

if not Path(fwd_file).exists() or not Path(rev_file).exists():
    print('ERROR: Missing results')
    exit(1)

fwd = json.load(open(fwd_file))
rev = json.load(open(rev_file))

# CORRECT KEY: 'survivors'
fwd_seeds = set(fwd.get('survivors', []))
rev_seeds = set(rev.get('survivors', []))

intersection = sorted(fwd_seeds & rev_seeds)

print(f'FORWARD survivors: {len(fwd_seeds)}')
print(f'REVERSE survivors: {len(rev_seeds)}')
print(f'INTERSECTION: {len(intersection)}')
if intersection:
    print(f'First 10: {intersection[:10]}')
    print(f'Last 10:  {intersection[-10:]}')

json.dump({'intersection': intersection}, open('results/intersection_90_orig.json', 'w'), indent=2)
print('Intersection saved.')
"
