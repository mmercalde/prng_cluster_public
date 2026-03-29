#!/usr/bin/env python3
"""
fix_s158_zero_survivor_npz.py

FIX: convert_survivors_to_binary.py crashes with ValueError when called
with an empty bidirectional_survivors.json (0 survivors). numpy .min()
on a zero-size array raises:
  ValueError: zero-size array to reduction operation minimum which has no identity

The fix: add an early exit after n = len(survivors) when n == 0.
This is a valid production case — some trial configs yield 0 bidirectional
survivors, and the pipeline should handle it gracefully.

Deploy:
  scp ~/Downloads/fix_s158_zero_survivor_npz.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 fix_s158_zero_survivor_npz.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/convert_survivors_to_binary.py")
BACKUP = TARGET.with_suffix(".py.bak_s158_zero_survivor")

OLD = '''    n = len(survivors)
    print(f"Loaded {n:,} survivors")

    if n > 0:'''

NEW = '''    n = len(survivors)
    print(f"Loaded {n:,} survivors")

    if n == 0:
        print("⚠️  No survivors — skipping NPZ conversion (empty input is valid)")
        # Write empty NPZ so downstream steps don't fail on missing file
        import numpy as np
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), seeds=np.array([], dtype=np.uint32))
        print(f"✅ Empty NPZ written to {output_file}")
        return
    if n > 0:'''

def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    if OLD not in content:
        if "No survivors — skipping NPZ" in content:
            print("Zero-survivor guard already present — skipping")
            return True
        print("ERROR: anchor not found")
        return False

    new_content = content.replace(OLD, NEW)

    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"ERROR: Syntax error: {e}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(new_content)

    try:
        ast.parse(TARGET.read_text())
        print("✅ Zero-survivor guard applied and verified")
        print("\nNext steps:")
        print("  git add convert_survivors_to_binary.py fix_s158_zero_survivor_npz.py")
        print("  git commit -m 'fix(s158): graceful empty NPZ on zero survivors'")
        print("  git push origin main && git push public main")
        return True
    except SyntaxError as e:
        print(f"ERROR: Post-write syntax error: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("Restored backup")
        return False

if __name__ == "__main__":
    print("Applying zero-survivor NPZ guard...")
    apply()
