#!/usr/bin/env python3
"""
Manually verify that forward and reverse produce different survivors
"""

import json
import subprocess
import glob
import os

print("="*80)
print("MANUAL VERIFICATION OF REVERSE SIEVE")
print("="*80)

# Run forward sieve
print("\n1. Running FORWARD sieve (java_lcg)...")
subprocess.run([
    'python3', 'coordinator.py', 'daily3.json',
    '--method', 'residue_sieve',
    '--prng-type', 'java_lcg',
    '--seeds', '50000',
    '--window-size', '512',
    '--skip-min', '0',
    '--skip-max', '10',
    '--session-filter', 'both',
    '--max-concurrent', '26'
], capture_output=True)

# Get the result
fwd_file = max(glob.glob('results/multi_gpu_analysis_*.json'), key=os.path.getctime)
with open(fwd_file) as f:
    fwd_data = json.load(f)

# Extract survivor seeds
fwd_survivors = []
for result in fwd_data.get('results', []):
    for survivor in result.get('survivors', []):
        fwd_survivors.append(survivor.get('seed'))

print(f"   Forward survivors: {len(fwd_survivors)}")
print(f"   First 10 seeds: {sorted(fwd_survivors)[:10]}")

# Run reverse sieve
print("\n2. Running REVERSE sieve (java_lcg_reverse)...")
subprocess.run([
    'python3', 'coordinator.py', 'daily3.json',
    '--method', 'residue_sieve',
    '--prng-type', 'java_lcg_reverse',
    '--seeds', '50000',
    '--window-size', '512',
    '--skip-min', '0',
    '--skip-max', '10',
    '--session-filter', 'both',
    '--max-concurrent', '26'
], capture_output=True)

rev_file = max(glob.glob('results/multi_gpu_analysis_*.json'), key=os.path.getctime)
with open(rev_file) as f:
    rev_data = json.load(f)

rev_survivors = []
for result in rev_data.get('results', []):
    for survivor in result.get('survivors', []):
        rev_survivors.append(survivor.get('seed'))

print(f"   Reverse survivors: {len(rev_survivors)}")
print(f"   First 10 seeds: {sorted(rev_survivors)[:10]}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

fwd_set = set(fwd_survivors)
rev_set = set(rev_survivors)
intersection = fwd_set & rev_set
only_fwd = fwd_set - rev_set
only_rev = rev_set - fwd_set

print(f"Forward only:       {len(fwd_set):6d} survivors")
print(f"Reverse only:       {len(rev_set):6d} survivors")
print(f"Intersection:       {len(intersection):6d} survivors (appear in both)")
print(f"Forward exclusive:  {len(only_fwd):6d} survivors (only in forward)")
print(f"Reverse exclusive:  {len(only_rev):6d} survivors (only in reverse)")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if len(fwd_set) == len(rev_set) == len(intersection):
    print("❌ BROKEN: All survivors are identical!")
    print("   Forward and reverse produce the SAME seeds")
elif len(only_fwd) > 0 or len(only_rev) > 0:
    print("✅ WORKING: Forward and reverse produce DIFFERENT survivors!")
    print(f"   Overlap: {len(intersection)/max(len(fwd_set),len(rev_set))*100:.1f}%")
    print(f"   This is expected - temporal reversal filters differently")
    
    # Show some examples
    if only_fwd:
        print(f"\n   Example forward-only seeds: {sorted(only_fwd)[:5]}")
    if only_rev:
        print(f"   Example reverse-only seeds: {sorted(only_rev)[:5]}")
else:
    print("⚠️  UNCLEAR: Need more data")

print("="*80)

