#!/usr/bin/env python3
"""
fix_s145r1_manifest.py
Restores window_optimizer.json from backup and re-applies
the four manifest patches WITHOUT invalid JSON comments.
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')
MANIFEST = PROJECT_ROOT / 'agent_manifests/window_optimizer.json'
BACKUP   = Path(str(MANIFEST) + '.s145r1_backup')

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c): Path(p).write_text(c, encoding='utf-8')

print("fix_s145r1_manifest.py — restore + clean re-patch")
print("=" * 55)

# Step 1 — restore from backup
if not BACKUP.exists():
    print(f"❌ Backup not found: {BACKUP}")
    raise SystemExit(1)

shutil.copy2(BACKUP, MANIFEST)
print(f"✅ Restored from backup: {BACKUP.name}")

content = read(MANIFEST)
original_lines = len(content.splitlines())
print(f"   Restored file: {original_lines} lines")

# Step 2 — re-apply all four patches WITHOUT comments (valid JSON)
patches = [
    (
        '"timeout_minutes": 240',
        '"timeout_minutes": 900',
        "action.timeout_minutes 240 → 900"
    ),
    (
        '"max_seeds": 10000000,',
        '"max_seeds": 1073741824,',
        "default_params.max_seeds 10000000 → 1073741824"
    ),
    (
        '"enable_pruning": false,',
        '"enable_pruning": true,',
        "default_params.enable_pruning false → true"
    ),
    (
        '"window_trials": 100,',
        '"window_trials": 50,',
        "default_params.window_trials 100 → 50"
    ),
]

all_ok = True
for old, new, desc in patches:
    if old not in content:
        print(f"❌ ANCHOR NOT FOUND: {desc}")
        all_ok = False
        continue
    content = content.replace(old, new, 1)
    print(f"✅ Patched: {desc}")

if not all_ok:
    print("\n❌ Some patches failed — not writing file")
    raise SystemExit(1)

# Step 3 — validate JSON before writing
import json
try:
    json.loads(content)
    print("✅ JSON valid")
except json.JSONDecodeError as e:
    print(f"❌ JSON invalid after patch: {e}")
    raise SystemExit(1)

write(MANIFEST, content)
final_lines = len(read(MANIFEST).splitlines())
print(f"✅ Written: {final_lines} lines (same as original — value replacements only)")

print("\n✅ Manifest fixed. Re-run smoke tests.")
