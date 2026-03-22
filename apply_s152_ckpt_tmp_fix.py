#!/usr/bin/env python3
"""
apply_s152_ckpt_tmp_fix.py
===========================
Fix: numpy.savez_compressed auto-appends .npz to filename if not present.

ROOT CAUSE
----------
In window_optimizer_bayesian.py, the S149-CKPT block does:

    _tmp = _accum_npz + '.ckpt.tmp'
    # _tmp = 'bidirectional_survivors_all.npz.ckpt.tmp'
    _np_ckpt.savez_compressed(_tmp, ...)
    _os_ckpt.replace(_tmp, _accum_npz)

numpy.savez_compressed() appends '.npz' when the filename doesn't end in '.npz'.
So the actual file written is:
    'bidirectional_survivors_all.npz.ckpt.tmp.npz'

But os.replace() tries to rename:
    'bidirectional_survivors_all.npz.ckpt.tmp'  ← does not exist → FileNotFoundError

Fix: use a tmp filename that already ends in '.npz' so numpy doesn't append it:
    _tmp = 'bidirectional_survivors_all.ckpt.tmp.npz'

Same fix applied to the binary NPZ tmp file.

Files patched
-------------
  window_optimizer_bayesian.py

Backup: window_optimizer_bayesian.py.bak_s152_ckpt_tmp
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("window_optimizer_bayesian.py")
BACKUP = Path("window_optimizer_bayesian.py.bak_s152_ckpt_tmp")

# Fix 1: accum NPZ tmp file
OLD_TMP1 = '''\
                        # Atomic write
                        _tmp = _accum_npz + '.ckpt.tmp'
                        _np_ckpt.savez_compressed(_tmp, seeds=_seeds, score=_scores)
                        _os_ckpt.replace(_tmp, _accum_npz)'''

NEW_TMP1 = '''\
                        # Atomic write
                        # [S152] numpy.savez_compressed appends .npz if not present
                        # so tmp must already end in .npz or rename will fail
                        _tmp = _accum_npz.replace('.npz', '.ckpt.tmp.npz')
                        _np_ckpt.savez_compressed(_tmp, seeds=_seeds, score=_scores)
                        _os_ckpt.replace(_tmp, _accum_npz)'''

# Fix 2: binary NPZ tmp file
OLD_TMP2 = '''\
                        _tmp_bin = _binary_npz + '.ckpt.tmp'
                        _np_ckpt.savez_compressed(_tmp_bin, seeds=_seeds,
                                                  forward_match_rate=_fwd_mr,
                                                  reverse_match_rate=_rev_mr,
                                                  score=_scores)
                        _os_ckpt.replace(_tmp_bin, _binary_npz)'''

NEW_TMP2 = '''\
                        _tmp_bin = _binary_npz.replace('.npz', '.ckpt.tmp.npz')
                        _np_ckpt.savez_compressed(_tmp_bin, seeds=_seeds,
                                                  forward_match_rate=_fwd_mr,
                                                  reverse_match_rate=_rev_mr,
                                                  score=_scores)
                        _os_ckpt.replace(_tmp_bin, _binary_npz)'''


def apply():
    src = TARGET.read_text()

    if "[S152] numpy.savez_compressed" in src:
        print("⚠️  Already patched — aborting.")
        return

    missing = []
    if OLD_TMP1 not in src:
        missing.append("accum NPZ tmp anchor")
    if OLD_TMP2 not in src:
        missing.append("binary NPZ tmp anchor")
    if missing:
        print(f"❌ Anchors not found: {missing}")
        return

    patched = src.replace(OLD_TMP1, NEW_TMP1, 1)
    patched = patched.replace(OLD_TMP2, NEW_TMP2, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  [S152] marker present: {'[S152] numpy.savez_compressed' in patched}")
        print(f"  .ckpt.tmp.npz present: {'.ckpt.tmp.npz' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Fix: tmp files now end in .npz so numpy doesn't append it")
    print("  Before: bidirectional_survivors_all.npz.ckpt.tmp      ← numpy writes .tmp.npz")
    print("  After:  bidirectional_survivors_all.ckpt.tmp.npz      ← numpy writes exactly this")
    print()
    print("[S149-CKPT] will now write NPZ successfully after each trial with survivors.")


if __name__ == "__main__":
    apply()
