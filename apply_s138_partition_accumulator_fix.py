#!/usr/bin/env python3
"""
S138 Patch: Fix partition worker accumulator via temp files instead of Queue.

ROOT CAUSE:
    Each partition worker builds _local_acc with millions of survivor dicts.
    Pickling this (~2.4 GB) into a multiprocessing.Queue pipe (64KB buffer)
    causes the worker's result_queue.put() to block indefinitely.
    The parent's _rq.get(timeout=7200) then raises queue.Empty immediately
    (pipe deadlock, not a true timeout), breaking out of the collection loop
    with zero survivors merged.

FIX:
    1. Pass a temp_file path to each partition worker instead of a queue
    2. Worker writes _local_acc to temp JSON file after optimize() completes
    3. Parent reads temp files after proc.join(), merges into survivor_accumulator
    4. Temp files deleted after merge
    5. result_queue retained for status/error signaling only (tiny payload)

CHANGES:
    window_optimizer_integration_final.py:
    - _partition_worker: add temp_file param, write _local_acc to JSON file,
      put status-only dict to queue (no accumulator)
    - Parent collection loop: read temp files after join, merge survivors
"""

import re
import sys
import shutil
from datetime import datetime

TARGET = sys.argv[1] if len(sys.argv) > 1 else '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'

# Backup before patching
BACKUP = TARGET + f'.bak_s138_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy2(TARGET, BACKUP)
print(f"✅ Backup saved: {BACKUP}")

with open(TARGET, 'r') as f:
    src = f.read()

# ============================================================================
# CHANGE 1: Update _partition_worker signature to accept temp_file param
# OLD: result_queue):
# NEW: result_queue, temp_file):
# ============================================================================
OLD_SIG = """            def _partition_worker(partition_idx, allowlist, config_file_w,
                                   dataset_path_w, seed_start_w, seed_count_w,
                                   prng_base_w, test_both_modes_w,
                                   storage_url, study_name_w, trials_for_worker,
                                   result_queue):"""

NEW_SIG = """            def _partition_worker(partition_idx, allowlist, config_file_w,
                                   dataset_path_w, seed_start_w, seed_count_w,
                                   prng_base_w, test_both_modes_w,
                                   storage_url, study_name_w, trials_for_worker,
                                   result_queue, temp_file):"""

assert OLD_SIG in src, "ABORT: signature not found"
src = src.replace(OLD_SIG, NEW_SIG, 1)
print("✅ Change 1: _partition_worker signature updated")

# ============================================================================
# CHANGE 2: Replace result_queue.put (ok path) — write to temp file instead
# OLD: result_queue.put({'partition': ..., 'accumulator': _local_acc, 'status': 'ok'})
# NEW: write _local_acc to temp_file, put status-only dict to queue
# ============================================================================
OLD_PUT_OK = """                    result_queue.put({
                        'partition': partition_idx,
                        'accumulator': _local_acc,
                        'status': 'ok',
                    })"""

NEW_PUT_OK = """                    # S138: Write accumulator to temp file (avoids 2.4GB pipe deadlock)
                    import json as _json
                    with open(temp_file, 'w') as _tf:
                        _json.dump(_local_acc, _tf)
                    result_queue.put({
                        'partition': partition_idx,
                        'status': 'ok',
                        'temp_file': temp_file,
                    })"""

assert OLD_PUT_OK in src, "ABORT: ok put not found"
src = src.replace(OLD_PUT_OK, NEW_PUT_OK, 1)
print("✅ Change 2: ok queue put replaced with temp file write")

# ============================================================================
# CHANGE 3: Replace result_queue.put (error path) — no accumulator needed
# OLD: result_queue.put({'partition': ..., 'accumulator': {...}, 'status': 'error', ...})
# NEW: result_queue.put({'partition': ..., 'status': 'error', ...}) — no accumulator
# ============================================================================
OLD_PUT_ERR = """                    result_queue.put({
                        'partition': partition_idx,
                        'accumulator': {'forward': [], 'reverse': [], 'bidirectional': []},
                        'status': 'error',
                        'error': _tb.format_exc(),
                    })"""

NEW_PUT_ERR = """                    result_queue.put({
                        'partition': partition_idx,
                        'status': 'error',
                        'error': _tb.format_exc(),
                    })"""

assert OLD_PUT_ERR in src, "ABORT: error put not found"
src = src.replace(OLD_PUT_ERR, NEW_PUT_ERR, 1)
print("✅ Change 3: error queue put cleaned up (no accumulator)")

# ============================================================================
# CHANGE 4: Update Process args to pass temp_file path
# OLD: args=(..., _trials_per_worker[_pi], _rq,)
# NEW: args=(..., _trials_per_worker[_pi], _rq, f'/tmp/partition_{_pi}_survivors.json',)
# ============================================================================
OLD_PROC_ARGS = """                _proc = _mp.Process(
                    target=_partition_worker,
                    args=(
                        _pi,
                        _PARALLEL_PARTITIONS[_pi],
                        getattr(self, 'config_file', 'distributed_config.json'),
                        dataset_path, seed_start, seed_count,
                        prng_base, test_both_modes,
                        _mp_storage_url, _mp_study_name,
                        _trials_per_worker[_pi],
                        _rq,
                    ),
                    daemon=False,
                )"""

NEW_PROC_ARGS = """                _proc = _mp.Process(
                    target=_partition_worker,
                    args=(
                        _pi,
                        _PARALLEL_PARTITIONS[_pi],
                        getattr(self, 'config_file', 'distributed_config.json'),
                        dataset_path, seed_start, seed_count,
                        prng_base, test_both_modes,
                        _mp_storage_url, _mp_study_name,
                        _trials_per_worker[_pi],
                        _rq,
                        f'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json',
                    ),
                    daemon=False,
                )"""

assert OLD_PROC_ARGS in src, "ABORT: Process args not found"
src = src.replace(OLD_PROC_ARGS, NEW_PROC_ARGS, 1)
print("✅ Change 4: Process args updated with temp_file path")

# ============================================================================
# CHANGE 5: Replace parent collection loop — read temp files instead of accumulator from queue
# OLD: survivor_accumulator[_k].extend(_res['accumulator'][_k])
# NEW: after join, read temp files and merge
# ============================================================================
OLD_COLLECT = """            # Collect results from both worker processes
            _collected = 0
            while _collected < n_parallel:
                try:
                    _res = _rq.get(timeout=7200)  # 2-hour hard timeout per worker
                    _pi = _res['partition']
                    if _res['status'] == 'ok':
                        print(f\"\\n   Process-{_pi} complete -- merging survivors\")
                        for _k in ('forward', 'reverse', 'bidirectional'):
                            survivor_accumulator[_k].extend(_res['accumulator'][_k])
                    else:
                        print(f\"\\n   Process-{_pi} ERROR:\")
                        print(_res.get('error', 'unknown error'))
                    _collected += 1
                except Exception as _qe:
                    print(f\"   Queue timeout/error: {_qe}\")
                    break

            for _proc in _procs:
                _proc.join(timeout=60)
                if _proc.is_alive():
                    print(f\"   Process {_proc.pid} still alive -- terminating\")
                    _proc.terminate()"""

NEW_COLLECT = """            # Collect status from queue (lightweight — no accumulator payload)
            _collected = 0
            _partition_status = {}
            while _collected < n_parallel:
                try:
                    _res = _rq.get(timeout=7200)
                    _pi = _res['partition']
                    _partition_status[_pi] = _res
                    if _res['status'] == 'ok':
                        print(f\"\\n   Process-{_pi} signaled OK (survivors in temp file)\")
                    else:
                        print(f\"\\n   Process-{_pi} ERROR:\")
                        print(_res.get('error', 'unknown error'))
                    _collected += 1
                except Exception as _qe:
                    print(f\"   Queue timeout/error: {_qe}\")
                    break

            for _proc in _procs:
                _proc.join(timeout=60)
                if _proc.is_alive():
                    print(f\"   Process {_proc.pid} still alive -- terminating\")
                    _proc.terminate()

            # S138: Read temp files and merge survivors into accumulator
            import json as _json2
            import os as _os2
            for _pi in range(n_parallel):
                _tf_path = f'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json'
                if _os2.path.exists(_tf_path):
                    print(f\"   Process-{_pi} complete -- merging survivors from temp file\")
                    try:
                        with open(_tf_path, 'r') as _tf:
                            _res_acc = _json2.load(_tf)
                        for _k in ('forward', 'reverse', 'bidirectional'):
                            survivor_accumulator[_k].extend(_res_acc.get(_k, []))
                        print(f\"      Merged: fwd={len(_res_acc.get('forward',[]))} \"\
                              f\"rev={len(_res_acc.get('reverse',[]))} \"\
                              f\"bid={len(_res_acc.get('bidirectional',[]))}\")
                        _os2.remove(_tf_path)
                    except Exception as _tfe:
                        print(f\"   ⚠️  Failed to read temp file {_tf_path}: {_tfe}\")
                else:
                    print(f\"   ⚠️  Process-{_pi} temp file missing: {_tf_path}\")"""

assert OLD_COLLECT in src, "ABORT: collection loop not found"
src = src.replace(OLD_COLLECT, NEW_COLLECT, 1)
print("✅ Change 5: collection loop replaced with temp file merge")

# ============================================================================
# Write patched file
# ============================================================================
with open(TARGET, 'w') as f:
    f.write(src)

print("\n✅ All 5 changes applied successfully")
print(f"   Target: {TARGET}")
print("\nVerification counts:")
print(f"   'temp_file' occurrences: {src.count('temp_file')}")
print(f"   'accumulator' in queue put: {'accumulator' in src[src.find('result_queue.put'):src.find('result_queue.put')+200]}")
print(f"   'partition_survivors' occurrences: {src.count('partition_survivors')}")
