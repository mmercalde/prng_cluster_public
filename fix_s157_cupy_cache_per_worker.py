#!/usr/bin/env python3
"""
fix_s157_cupy_cache_per_worker.py

ROOT CAUSE: 8 workers simultaneously compile cp.RawKernel() and race on the
shared CUPY_CACHE_DIR=${HOME}/.cache/cupy → kernel crash / hang.

S152 identified this fix but it was never applied to the live code.
The fix: add per-worker CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N in _spawn_worker().

Deploy:
  scp ~/Downloads/fix_s157_cupy_cache_per_worker.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 fix_s157_cupy_cache_per_worker.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s157_cupy_cache")

OLD = '        rocm_env = " ".join(ROCM_ENV_VARS + [\n            f"ROCR_VISIBLE_DEVICES={gpu_id}",\n        ])'

NEW = '        rocm_env = " ".join(ROCM_ENV_VARS + [\n            f"ROCR_VISIBLE_DEVICES={gpu_id}",\n            f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",  # [S157] per-worker isolated cache\n        ])'

def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    if OLD not in content:
        if "cupy_cache_gpu_" in content:
            print("Per-worker CUPY_CACHE_DIR already applied — skipping")
            return True
        print("ERROR: anchor not found")
        print("Looking for:")
        print(repr(OLD))
        return False

    new_content = content.replace(OLD, NEW)

    # Validate syntax
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"ERROR: Syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    TARGET.write_text(new_content)

    try:
        ast.parse(TARGET.read_text())
        print("✅ Per-worker CUPY_CACHE_DIR applied and verified")
        print("\nNext steps:")
        print("  git add persistent_worker_coordinator.py fix_s157_cupy_cache_per_worker.py")
        print("  git commit -m 'fix(s157): per-worker CUPY_CACHE_DIR — prevents 8-worker cache race'")
        print("  git push origin main && git push public main")
        return True
    except SyntaxError as e:
        print(f"ERROR: Post-write syntax error: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("Restored backup")
        return False

if __name__ == "__main__":
    print("Applying per-worker CUPY_CACHE_DIR fix...")
    apply()
