#!/usr/bin/env python3
"""
S139 Patch: Lower window_size max from 500 → 50 across all locations.

RATIONALE:
    167-trial Optuna run confirmed short-term temporal regime — W2 dominant,
    W3 next best. Window sizes >50 were never competitive. TRSE also confirms
    short_persistence (conf=0.8275) which caps window at 32 via Rule A.
    Setting max=50 as the config ceiling provides a safe buffer above TRSE's
    runtime cap while eliminating the vast wasteland of W50-500 from search.

CHANGES:
    1. distributed_config.json       — search_bounds window_size max: 500 → 50
    2. window_optimizer.py           — fallback dict window_size max: 500 → 50
    3. window_optimizer.py           — fallback dict skip_max max: 500 → 250
       (consistency with distributed_config.json which already has 250)
    4. window_optimizer.py           — SearchBounds dataclass default: 500 → 50
    5. agent_manifests/window_optimizer.json — informational max: 500 → 50
"""

import sys
import json
import shutil
from datetime import datetime

BASE = sys.argv[1] if len(sys.argv) > 1 else '/home/michael/distributed_prng_analysis'

files = {
    'config':    f'{BASE}/distributed_config.json',
    'optimizer': f'{BASE}/window_optimizer.py',
    'manifest':  f'{BASE}/agent_manifests/window_optimizer.json',
}

# Backup all files
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
for key, path in files.items():
    backup = f'{path}.bak_s139_{ts}'
    shutil.copy2(path, backup)
    print(f"✅ Backup: {backup}")

# ============================================================================
# CHANGE 1: distributed_config.json — window_size max 500 → 50
# ============================================================================
with open(files['config'], 'r') as f:
    cfg = json.load(f)

old_max = cfg['search_bounds']['window_size']['max']
assert old_max == 500, f"ABORT: expected 500, got {old_max}"
cfg['search_bounds']['window_size']['max'] = 50

with open(files['config'], 'w') as f:
    json.dump(cfg, f, indent=2)
print(f"✅ Change 1: distributed_config.json window_size max {old_max} → 50")

# ============================================================================
# CHANGE 2 & 3: window_optimizer.py — fallback dict
# ============================================================================
with open(files['optimizer'], 'r') as f:
    src = f.read()

OLD_FALLBACK = '''    defaults = {
        "window_size": {"min": 2, "max": 500},
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 500},
        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
    }'''

NEW_FALLBACK = '''    defaults = {
        "window_size": {"min": 2, "max": 50},   # S139: 500→50, short-term temporal confirmed
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 250},     # S139: 500→250, matches distributed_config.json
        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
    }'''

assert OLD_FALLBACK in src, "ABORT: fallback dict not found"
src = src.replace(OLD_FALLBACK, NEW_FALLBACK, 1)
print("✅ Change 2+3: window_optimizer.py fallback dict updated (window_size 500→50, skip_max 500→250)")

# ============================================================================
# CHANGE 4: window_optimizer.py — SearchBounds dataclass default
# ============================================================================
OLD_DATACLASS = '    max_window_size: int = 500'
NEW_DATACLASS = '    max_window_size: int = 50    # S139: 500→50, short-term temporal confirmed'

assert OLD_DATACLASS in src, "ABORT: SearchBounds dataclass default not found"
src = src.replace(OLD_DATACLASS, NEW_DATACLASS, 1)
print("✅ Change 4: window_optimizer.py SearchBounds dataclass max_window_size 500→50")

with open(files['optimizer'], 'w') as f:
    f.write(src)

# ============================================================================
# CHANGE 5: agent_manifests/window_optimizer.json — informational max
# ============================================================================
with open(files['manifest'], 'r') as f:
    manifest = json.load(f)

old_mmax = manifest['parameter_bounds']['window_size']['max']
# Manifest was stale (4096 from old informational values) — update regardless
manifest['parameter_bounds']['window_size']['max'] = 50
manifest['parameter_bounds']['window_size']['min'] = 2
manifest['parameter_bounds']['window_size']['default'] = 2
manifest['parameter_bounds']['window_size']['effect'] = (
    "Short-term temporal regime confirmed (167-trial run S138). "
    "TRSE Rule A caps at 32 when short_persistence active. "
    "Config ceiling 50 provides buffer. W2 dominant empirically."
)

with open(files['manifest'], 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Change 5: agent_manifests/window_optimizer.json window_size max {old_mmax} → 50")

# ============================================================================
# Verify
# ============================================================================
print("\nVerification:")
with open(files['config']) as f:
    cfg2 = json.load(f)
print(f"   distributed_config.json window_size max: {cfg2['search_bounds']['window_size']['max']}")

with open(files['optimizer']) as f:
    src2 = f.read()
print(f"   window_optimizer.py 'max_window_size: int = 50' present: {'max_window_size: int = 50' in src2}")
_count = src2.count('"max": 50')
print(f"   window_optimizer.py fallback 'max': 50 count: {_count}")

with open(files['manifest']) as f:
    m2 = json.load(f)
print(f"   manifest window_size max: {m2['parameter_bounds']['window_size']['max']}")

# Confirm no more 500 window references
remaining = [l for l in src2.split('\n') 
             if '500' in l and 'window' in l.lower() 
             and '5000' not in l and 'seed' not in l.lower()]
if remaining:
    print(f"\n⚠️  Remaining window/500 references to review:")
    for l in remaining:
        print(f"   {l.strip()}")
else:
    print("\n✅ No remaining window_size=500 references found")

print("\n✅ S139 patch complete")
