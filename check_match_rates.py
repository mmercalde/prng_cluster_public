#!/usr/bin/env python3
"""Check if survivors have different match rates"""

import json
import glob
import os

# Get latest two result files
files = sorted(glob.glob('results/multi_gpu_analysis_*.json'), key=os.path.getctime)[-2:]

print("Comparing last 2 test results:")
print(f"  File 1: {os.path.basename(files[0])}")
print(f"  File 2: {os.path.basename(files[1])}")

for i, file in enumerate(files, 1):
    with open(file) as f:
        data = json.load(f)
    
    print(f"\nFile {i} survivors:")
    for result in data.get('results', [])[:3]:  # First 3 jobs
        for survivor in result.get('survivors', [])[:3]:  # First 3 survivors per job
            print(f"  Seed {survivor.get('seed')}: {survivor.get('match_rate'):.4f} match rate, skip={survivor.get('best_skip')}")

