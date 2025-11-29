#!/usr/bin/env python3
"""Auto-detect hybrid from PRNG name instead of requiring --hybrid flag"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE job - include hybrid parameters
                use_hybrid = getattr(args, 'hybrid', False)"""

new = """            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE job - include hybrid parameters
                # Auto-detect hybrid from PRNG name (matches old static behavior)
                use_hybrid = '_hybrid' in (args.prng_type if hasattr(args, 'prng_type') else '')"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed hybrid auto-detection from PRNG name!")
else:
    print("❌ Pattern not found")
    exit(1)
