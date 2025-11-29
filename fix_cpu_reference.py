#!/usr/bin/env python3
"""
Fix the CPU reference overflow bug in xoshiro256pp_reverse
"""

import shutil
from datetime import datetime

# Backup first
backup_file = f"prng_registry.py.backup_cpu_bug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2('prng_registry.py', backup_file)
print(f"✅ Backup created: {backup_file}")

# Read the file
with open('prng_registry.py', 'r') as f:
    content = f.read()

# Fix the bug
old_line = "        result = (rotl(s0 + s3, 23) + s0) & 0xFFFFFFFFFFFFFFFF"
new_line = "        result = (rotl((s0 + s3) & 0xFFFFFFFFFFFFFFFF, 23) + s0) & 0xFFFFFFFFFFFFFFFF"

if old_line in content:
    content = content.replace(old_line, new_line)
    
    with open('prng_registry.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed xoshiro256pp_reverse CPU reference")
    print(f"   Changed: s0 + s3")
    print(f"   To:      (s0 + s3) & 0xFFFFFFFFFFFFFFFF")
else:
    print("⚠️  Could not find the line to fix!")
    print("   The code may have already been fixed or changed.")

print("\n" + "="*70)
print("DONE! Now test:")
print("="*70)
print("python3 << 'EOF'")
print("from prng_registry import get_cpu_reference")
print("cpu_ref = get_cpu_reference('xoshiro256pp_reverse')")
print("result = cpu_ref(1234, 3, skip=5, offset=0)")
print("print(f'CPU reference: {[x % 1000 for x in result]}')")
print("print('Should be: [808, 187, 219]')")
print("EOF")
