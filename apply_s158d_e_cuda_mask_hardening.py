#!/usr/bin/env python3
"""
apply_s158d_e_cuda_mask_hardening.py

Team Beta S158D-E: Harden Zeus CUDA mask contract in zmq_sqlite_worker.py.

Rule: launcher sets CUDA_VISIBLE_DEVICES at process spawn.
      Worker uses setdefault so launcher mask wins if already set.
      execute_sieve_job always called with logical device 0 after masking.

Also adds startup log showing CUDA_VISIBLE_DEVICES per worker.

Deploy:
  scp ~/Downloads/apply_s158d_e_cuda_mask_hardening.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s158d_e_cuda_mask_hardening.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/zmq_sqlite_worker.py")
BACKUP = TARGET.with_suffix(".py.bak_s158d_e")

# ── Patch 1: setdefault for CUDA (launcher mask wins) ─────────────────────
OLD_CUDA = '        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)'
NEW_CUDA = '        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))  # S158D-E: launcher mask wins'

# ── Patch 2: add startup log after worker connects ────────────────────────
OLD_READY = '''    log.info(
        f"Worker ready — id={worker_id} gpu={gpu_id} cuda={use_cuda} "
        f"zeus={args.zeus_host}:{args.job_port}"
    )'''

NEW_READY = '''    log.info(
        f"Worker ready — id={worker_id} gpu={gpu_id} cuda={use_cuda} "
        f"zeus={args.zeus_host}:{args.job_port} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES','unset')}"
    )'''


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    # Patch 1: CUDA setdefault
    if OLD_CUDA not in content:
        print("ERROR: CUDA_VISIBLE_DEVICES anchor not found")
        return False
    content = content.replace(OLD_CUDA, NEW_CUDA, 1)
    print("  ✅ Patch 1: CUDA setdefault applied")

    # Patch 2: startup log
    if OLD_READY not in content:
        print("WARNING: startup log anchor not found — skipping")
    else:
        content = content.replace(OLD_READY, NEW_READY, 1)
        print("  ✅ Patch 2: startup log with device env added")

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"ERROR: syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(content)
    ast.parse(TARGET.read_text())

    print("✅ S158D-E: CUDA mask hardening applied")
    print()
    print("Git commit:")
    print("  git add zmq_sqlite_worker.py apply_s158d_e_cuda_mask_hardening.py")
    print("  git commit -m 'fix(s158d-e): harden Zeus CUDA mask — setdefault + startup log'")
    print("  git push origin main && git push public main")
    return True


if __name__ == "__main__":
    print("Applying S158D-E: CUDA mask hardening...")
    apply()
