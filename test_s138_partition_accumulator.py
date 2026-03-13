#!/usr/bin/env python3
"""
S138 Smoke Test: Partition accumulator temp file fix.

Tests the Queue→tempfile plumbing WITHOUT running any GPU work.
Spawns two mock partition workers that:
  - Build a fake _local_acc with N synthetic survivor dicts
  - Write it to a temp file (patched behavior)
  - Put a status-only dict to the queue

Parent then:
  - Collects status from queue
  - Reads temp files after join
  - Merges into survivor_accumulator
  - Verifies counts match expected
  - Verifies temp files are cleaned up

Pass criteria:
  ✅ Both workers signal OK via queue
  ✅ Temp files exist after workers complete
  ✅ survivor_accumulator counts match sum of both partitions
  ✅ Temp files deleted after merge
  ✅ No queue deadlock (completes within timeout)

Usage:
    python3 test_s138_partition_accumulator.py
"""

import multiprocessing as mp
import json
import os
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
N_PARALLEL      = 2
SURVIVORS_P0    = 50_000    # partition 0 fake survivor count
SURVIVORS_P1    = 75_000    # partition 1 fake survivor count
STUDY_NAME      = 'test_study_s138'
TIMEOUT_SECS    = 60        # should complete in seconds, not hours
PASS  = "✅"
FAIL  = "❌"

# ── Mock survivor dict (mirrors real structure) ───────────────────────────────
def make_survivor(seed, partition_idx, trial_number):
    return {
        'seed':                   seed,
        'forward_match_rate':     0.75,
        'reverse_match_rate':     0.50,
        'score':                  0.625,
        'window_size':            2,
        'offset':                 14,
        'skip_min':               7,
        'skip_max':               63,
        'trial_number':           trial_number,
        'prng_type':              'java_lcg',
        'prng_base':              'java_lcg',
        'skip_mode':              'variable',
        'forward_count':          1_000_000,
        'reverse_count':          900_000,
        'bidirectional_count':    750_000,
        'forward_only_count':     250_000,
        'reverse_only_count':     150_000,
        'intersection_count':     600_000,
        'intersection_ratio':     0.80,
        'survivor_overlap_ratio': 0.75,
        'intersection_weight':    0.40,
        'sessions':               'evening',
        'skip_range':             '7-63',
    }

# ── Mock partition worker (mirrors patched _partition_worker) ─────────────────
def mock_partition_worker(partition_idx, n_survivors, result_queue, temp_file):
    """
    Simulates patched _partition_worker:
      - Builds _local_acc with n_survivors fake entries
      - Writes to temp_file
      - Puts status-only dict to queue
    """
    try:
        print(f"   [P{partition_idx}] Building {n_survivors:,} fake survivors...")
        _local_acc = {
            'forward':       [make_survivor(i, partition_idx, 1) for i in range(n_survivors)],
            'reverse':       [make_survivor(i, partition_idx, 1) for i in range(n_survivors)],
            'bidirectional': [make_survivor(i, partition_idx, 1) for i in range(n_survivors)],
        }

        # S138: Write accumulator to temp file (avoids 2.4GB pipe deadlock)
        import json as _json
        with open(temp_file, 'w') as _tf:
            _json.dump(_local_acc, _tf)
        print(f"   [P{partition_idx}] Wrote temp file: {temp_file}")

        result_queue.put({
            'partition': partition_idx,
            'status':    'ok',
            'temp_file': temp_file,
        })
        print(f"   [P{partition_idx}] Status OK sent to queue")

    except Exception:
        import traceback as _tb
        result_queue.put({
            'partition': partition_idx,
            'status':    'error',
            'error':     _tb.format_exc(),
        })

# ── Main test ─────────────────────────────────────────────────────────────────
def run_test():
    print("=" * 60)
    print("S138 Smoke Test: Partition accumulator temp file fix")
    print("=" * 60)
    print(f"   P0 survivors: {SURVIVORS_P0:,}")
    print(f"   P1 survivors: {SURVIVORS_P1:,}")
    print(f"   Study name:   {STUDY_NAME}")
    print()

    results = []
    start = time.time()

    # ── Setup ─────────────────────────────────────────────────────────────────
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
    _rq    = mp.Queue()
    _procs = []
    n_survivors_per_partition = [SURVIVORS_P0, SURVIVORS_P1]

    # ── Spawn workers ─────────────────────────────────────────────────────────
    for _pi in range(N_PARALLEL):
        _tf_path = f'/tmp/partition_{_pi}_survivors_{STUDY_NAME}.json'
        _proc = mp.Process(
            target=mock_partition_worker,
            args=(_pi, n_survivors_per_partition[_pi], _rq, _tf_path),
            daemon=False,
        )
        _proc.start()
        _procs.append(_proc)
        print(f"   Started Process-{_pi} (pid={_proc.pid})")

    # ── Collect status from queue ─────────────────────────────────────────────
    print("\n   Waiting for queue signals...")
    _collected = 0
    _partition_status = {}
    while _collected < N_PARALLEL:
        try:
            _res = _rq.get(timeout=TIMEOUT_SECS)
            _pi  = _res['partition']
            _partition_status[_pi] = _res
            if _res['status'] == 'ok':
                print(f"   Process-{_pi} signaled OK (survivors in temp file)")
            else:
                print(f"   Process-{_pi} ERROR: {_res.get('error','unknown')}")
            _collected += 1
        except Exception as _qe:
            print(f"   Queue timeout/error: {_qe}")
            break

    # ── Join workers ──────────────────────────────────────────────────────────
    for _proc in _procs:
        _proc.join(timeout=60)
        if _proc.is_alive():
            print(f"   Process {_proc.pid} still alive -- terminating")
            _proc.terminate()

    # ── TEST 1: Both workers signaled OK ──────────────────────────────────────
    t1 = all(_partition_status.get(i, {}).get('status') == 'ok' for i in range(N_PARALLEL))
    results.append((f"Both workers signaled OK via queue", t1))

    # ── Read temp files and merge ─────────────────────────────────────────────
    print("\n   Merging temp files into survivor_accumulator...")
    import json as _json2
    import os as _os2

    temp_files_found = []
    for _pi in range(N_PARALLEL):
        _tf_path = f'/tmp/partition_{_pi}_survivors_{STUDY_NAME}.json'

        # TEST 2: Temp file exists
        _exists = _os2.path.exists(_tf_path)
        temp_files_found.append(_exists)

        if _exists:
            try:
                with open(_tf_path, 'r') as _tf:
                    _res_acc = _json2.load(_tf)
                for _k in ('forward', 'reverse', 'bidirectional'):
                    survivor_accumulator[_k].extend(_res_acc.get(_k, []))
                print(f"   Process-{_pi} merged: "
                      f"fwd={len(_res_acc.get('forward',[]))} "
                      f"rev={len(_res_acc.get('reverse',[]))} "
                      f"bid={len(_res_acc.get('bidirectional',[]))}")
                _os2.remove(_tf_path)
            except Exception as _tfe:
                print(f"   ⚠️  Failed to read temp file {_tf_path}: {_tfe}")
        else:
            print(f"   ⚠️  Temp file missing: {_tf_path}")

    t2 = all(temp_files_found)
    results.append(("Temp files existed after worker join", t2))

    # ── TEST 3: Survivor counts match expected ────────────────────────────────
    expected_total = SURVIVORS_P0 + SURVIVORS_P1
    actual_bid     = len(survivor_accumulator['bidirectional'])
    actual_fwd     = len(survivor_accumulator['forward'])
    actual_rev     = len(survivor_accumulator['reverse'])
    t3 = (actual_bid == expected_total and
          actual_fwd == expected_total and
          actual_rev == expected_total)
    results.append((
        f"survivor_accumulator counts correct "
        f"(expected {expected_total:,}, got bid={actual_bid:,} fwd={actual_fwd:,} rev={actual_rev:,})",
        t3
    ))

    # ── TEST 4: Temp files cleaned up ────────────────────────────────────────
    still_exist = [
        os.path.exists(f'/tmp/partition_{_pi}_survivors_{STUDY_NAME}.json')
        for _pi in range(N_PARALLEL)
    ]
    t4 = not any(still_exist)
    results.append(("Temp files deleted after merge", t4))

    # ── TEST 5: Completed within timeout (no deadlock) ────────────────────────
    elapsed = time.time() - start
    t5 = elapsed < TIMEOUT_SECS
    results.append((f"Completed without deadlock ({elapsed:.1f}s < {TIMEOUT_SECS}s)", t5))

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_pass = True
    for desc, passed in results:
        icon = PASS if passed else FAIL
        print(f"   {icon} {desc}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print(f"{PASS} ALL TESTS PASSED — patch is working correctly")
        sys.exit(0)
    else:
        print(f"{FAIL} SOME TESTS FAILED — investigate before running full study")
        sys.exit(1)

if __name__ == '__main__':
    run_test()
