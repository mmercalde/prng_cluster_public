#!/usr/bin/env python3
"""
S139B Patch: Apply TRSE Rule A in n_parallel partition worker path.

ROOT CAUSE:
    When n_parallel > 1, each partition worker builds its own _local_bounds
    via SearchBounds.from_config() and calls _pstudy.optimize(_worker_obj)
    directly — bypassing BayesianOptimization.search() and
    OptunaBayesianSearch.search() entirely. TRSE Rule A bounds narrowing
    lives inside OptunaBayesianSearch.search() and was therefore never
    applied in any parallel run.

FIX:
    After _local_bounds = SearchBounds.from_config() in _partition_worker,
    apply TRSE Rule A directly using the same logic as
    window_optimizer_bayesian.py lines 380-406.

    Passive: if trse_context.json absent, version < 1.15, confidence < 0.70,
    or regime not short_persistence — bounds unchanged, no error raised.
"""

import sys
import shutil
from datetime import datetime

TARGET = sys.argv[1] if len(sys.argv) > 1 else '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'

BACKUP = TARGET + f'.bak_s139b_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy2(TARGET, BACKUP)
print(f"✅ Backup saved: {BACKUP}")

with open(TARGET, 'r') as f:
    src = f.read()

# ============================================================================
# CHANGE: Apply TRSE Rule A after _local_bounds = SearchBounds.from_config()
# ============================================================================
OLD = """                    _local_acc = {'forward': [], 'reverse': [], 'bidirectional': []}
                    _local_bounds = SearchBounds.from_config()
                    _tctr = {'n': 0}"""

NEW = """                    _local_acc = {'forward': [], 'reverse': [], 'bidirectional': []}
                    _local_bounds = SearchBounds.from_config()

                    # S139B: Apply TRSE Rule A in partition worker path
                    # Mirrors OptunaBayesianSearch.search() lines 380-406
                    # Passive: no-op if context absent, stale, or confidence low
                    try:
                        import json as _trse_json
                        import os as _trse_os
                        _trse_path = trse_context_file if trse_context_file else 'trse_context.json'
                        if _trse_os.path.exists(_trse_path):
                            with open(_trse_path) as _tf:
                                _trse_ctx = _trse_json.load(_tf)
                            _trse_ver = _trse_ctx.get('trse_version', '0.0.0')
                            _vmaj, _vmin = int(_trse_ver.split('.')[0]), int(_trse_ver.split('.')[1])
                            if (_vmaj, _vmin) >= (1, 15):
                                _regime_type   = _trse_ctx.get('regime_type', 'unknown')
                                _type_conf     = _trse_ctx.get('regime_type_confidence', 0.0)
                                _regime_stable = _trse_ctx.get('regime_stable', False)
                                _w3_w8_ratio   = _trse_ctx.get('w3_w8_ratio', None)
                                print(f"\\n[TRSE][P{partition_idx}] Context loaded — "
                                      f"regime_type={_regime_type} "
                                      f"type_conf={_type_conf:.3f} "
                                      f"stable={_regime_stable} "
                                      f"w3_w8_ratio={_w3_w8_ratio}")
                                if (_regime_type == 'short_persistence'
                                        and _type_conf >= 0.70
                                        and _regime_stable):
                                    _old_max = _local_bounds.max_window_size
                                    _new_max = max(_local_bounds.min_window_size + 1,
                                                   min(32, _local_bounds.max_window_size))
                                    _local_bounds.max_window_size = _new_max
                                    print(f"[TRSE][P{partition_idx}] Rule A ACTIVE: "
                                          f"short_persistence (conf={_type_conf:.3f}) → "
                                          f"window_size ceiling {_old_max} → {_new_max}")
                                else:
                                    print(f"[TRSE][P{partition_idx}] Rule A SKIPPED: "
                                          f"type={_regime_type} conf={_type_conf:.3f} "
                                          f"stable={_regime_stable}")
                            else:
                                print(f"[TRSE][P{partition_idx}] Context version "
                                      f"{_trse_ver} < 1.15 — skipping bounds narrowing")
                        else:
                            print(f"[TRSE][P{partition_idx}] No context found — "
                                  f"running with default bounds")
                    except Exception as _trse_e:
                        print(f"[TRSE][P{partition_idx}] Context load failed "
                              f"(non-fatal): {_trse_e}")

                    _tctr = {'n': 0}"""

assert OLD in src, "ABORT: _local_bounds block not found"
src = src.replace(OLD, NEW, 1)
print("✅ Change 1: TRSE Rule A applied in partition worker path")

with open(TARGET, 'w') as f:
    f.write(src)

print("\n✅ All changes applied successfully")
print(f"   Target: {TARGET}")
print(f"\nVerification:")
print(f"   'S139B' occurrences: {src.count('S139B')}")
print(f"   'Rule A ACTIVE' occurrences: {src.count('Rule A ACTIVE')}")
print(f"   'TRSE][P' occurrences: {src.count('TRSE][P')}")
