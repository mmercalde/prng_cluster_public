#!/usr/bin/env python3
"""Check which backups have FULL vs SIMPLIFIED MT19937"""

import os
import glob

backups = glob.glob('prng_registry*.py')
backups.sort(key=os.path.getmtime, reverse=True)

print("=" * 70)
print("BACKUP ANALYSIS")
print("=" * 70)

working = []
broken = []

for backup in backups:
    # Skip the current working file
    if backup == 'prng_registry.py':
        continue
    
    with open(backup, 'r') as f:
        content = f.read()
    
    # Check for FULL MT19937 indicators
    has_624_array = 'unsigned int mt[624];' in content
    has_twist = 'for (int i = 0; i < N; i++)' in content and 'UPPER_MASK' in content
    
    # Check for simplified version indicators
    is_simplified = 'simplified - full version needs' in content or \
                    ('state = 1812433253U' in content and 'unsigned int mt[624];' not in content)
    
    size_kb = os.path.getsize(backup) / 1024
    mod_time = os.path.getmtime(backup)
    
    if has_624_array and has_twist:
        status = "✅ WORKING (Full MT19937)"
        working.append(backup)
    elif is_simplified:
        status = "❌ BROKEN (Simplified)"
        broken.append(backup)
    else:
        status = "⚠️  UNKNOWN"
    
    print(f"\n{backup}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Status: {status}")

print("\n" + "=" * 70)
print(f"Working backups: {len(working)}")
print(f"Broken backups: {len(broken)}")
print("=" * 70)

if broken:
    print("\n🗑️  Files to DELETE (broken/simplified):")
    for f in broken:
        print(f"   {f}")
    
    print("\n⚠️  Run this to delete them:")
    print(f"   rm {' '.join(broken)}")

if working:
    print("\n✅ Files to KEEP (working full MT19937):")
    for f in working:
        print(f"   {f}")

