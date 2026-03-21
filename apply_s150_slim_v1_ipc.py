#!/usr/bin/env python3
"""
apply_s150_slim_v1_ipc.py
===========================
Implement slim_v1 IPC serialization — TB approved Option A (S149/S150).

Three fixes from TB review incorporated:
  1. nested result bug: run_sieve_job returns flat dict (not nested under 'result')
     run_worker() wraps as {"status":"ok","result": <return>}
     coordinator reads inner = result.get("result", {}) then inner.get("format")
     So "format" must be at top level of run_sieve_job() return dict.
  2. verify() is mode-aware: --coordinator-only / --worker-only check only relevant file
  3. compact separators: (',', ':') not (',', ': ')

Deploy order (CRITICAL):
  Step 1: python3 apply_s150_slim_v1_ipc.py --coordinator-only
          → Deploy coordinator to Zeus, verify old workers still work
  Step 2: python3 apply_s150_slim_v1_ipc.py --worker-only
          → Commit + git pull on all 3 rigs
          Workers respawn per trial — new code on next trial spawn
"""

import argparse, shutil, os, sys

DRY_RUN = False

def log(msg): print(msg)

def backup(path, tag="s150_slim_v1"):
    if DRY_RUN:
        log(f"  [DRY-RUN] would backup {path}.bak_{tag}")
        return
    bak = f"{path}.bak_{tag}"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")

def replace_exact(content, old, new, label):
    if old not in content:
        log(f"  ERROR: anchor not found — {label}")
        return content, False
    if content.count(old) != 1:
        log(f"  WARNING: multiple occurrences of {label}")
    log(f"  patched: {label}")
    return content.replace(old, new, 1), True


def patch_coordinator(path):
    log(f"\n[FILE 1] {path}")
    backup(path)
    with open(path) as f:
        content = f.read()

    # Check if already patched
    if 'slim_v1' in content and 'fast path' in content:
        log("  already patched — skipping coordinator")
        log("\n  persistent_worker_coordinator.py: already patched")
        if not DRY_RUN:
            pass  # no write needed
        return True

    old = (
        '                    inner = result.get("result", {})\n'
        '                    raw_survivors = inner.get("survivors", [])\n'
        '                    survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]\n'
        '                    match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]\n'
        '                    skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]\n'
        '                    strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]'
    )
    new = (
        '                    inner = result.get("result", {})\n'
        '                    # [S150-slim_v1] Accept both slim parallel-array and legacy dict-list\n'
        '                    if inner.get("format") == "slim_v1":\n'
        '                        # Fast path — parallel arrays (TB approved Option A)\n'
        '                        survivors   = [int(s) for s in inner.get("seeds", [])]\n'
        '                        match_rates = list(inner.get("match_rates", []))\n'
        '                        n = len(survivors)\n'
        '                        # TB ruling: strategy_ids+skip_sequences required for hybrid\n'
        '                        _is_hybrid_job = (\n'
        '                            "hybrid" in str(job.get("prng_type", "")).lower()\n'
        '                            or str(job.get("skip_mode", "")).lower() == "hybrid"\n'
        '                        )\n'
        '                        if _is_hybrid_job:\n'
        '                            if "strategy_ids" not in inner or "skip_sequences" not in inner:\n'
        '                                handle.alive = False\n'
        '                                return {"status": "error", "message": "slim_v1 hybrid payload missing strategy_ids/skip_sequences"}\n'
        '                            strat_ids = list(inner["strategy_ids"])\n'
        '                            skip_seqs = list(inner["skip_sequences"])\n'
        '                        else:\n'
        '                            strat_ids = [0] * n\n'
        '                            skip_seqs = [[] for _ in survivors]\n'
        '                        # TB guardrail: all arrays must match len(seeds)\n'
        '                        if len(match_rates) != n or len(strat_ids) != n or len(skip_seqs) != n:\n'
        '                            handle.alive = False\n'
        '                            return {"status": "error", "message": f"slim_v1 length mismatch: seeds={n} match_rates={len(match_rates)} strat_ids={len(strat_ids)} skip_seqs={len(skip_seqs)}"}\n'
        '                    else:\n'
        '                        # Legacy path — list of dicts (kept for rollout safety)\n'
        '                        raw_survivors = inner.get("survivors", [])\n'
        '                        survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]\n'
        '                        match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]\n'
        '                        skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]\n'
        '                        strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]'
    )

    content, ok = replace_exact(content, old, new,
        "_dispatch_to_worker: slim_v1 fast path + legacy fallback")

    log(f"\n  persistent_worker_coordinator.py: {1 if ok else 0}/1 patches applied")
    if not DRY_RUN:
        with open(path, "w") as f: f.write(content)
        log(f"  wrote {path}")
    return ok


def patch_worker(path):
    # Pre-check: warn if coordinator slim_v1 parser not yet deployed
    coord_path = os.path.join(os.path.dirname(os.path.abspath(path)), "persistent_worker_coordinator.py")
    if os.path.exists(coord_path):
        coord_src = open(coord_path).read()
        if "slim_v1" not in coord_src:
            log("  WARNING: coordinator does not have slim_v1 parser!")
            log("  Run --coordinator-only on Zeus first, then --worker-only")
            log("  Proceeding — ensure coordinator is deployed before rigs pull.")

    log(f"\n[FILE 2] {path}")
    backup(path)
    with open(path) as f:
        content = f.read()

    ok_count = 0

    # Check if already patched
    if "slim_v1" in content and "separators=(" in content:
        log("  already patched — skipping worker")
        log("\n  sieve_gpu_worker.py: already patched (0 patches needed)")
        return True

    # Fix 2a: compact separators in _emit
    old = ('def _emit(obj: dict):\n'
           '    """Write JSON line to stdout and flush."""\n'
           '    print(json.dumps(obj), flush=True)')
    new = ('def _emit(obj: dict):\n'
           '    """Write JSON line to stdout and flush."""\n'
           '    # [S150-slim_v1] compact separators — zero whitespace overhead\n'
           '    print(json.dumps(obj, separators=(\',\', \':\')), flush=True)')
    content, ok = replace_exact(content, old, new, "_emit: compact separators")
    ok_count += ok

    # Fix 2b: hybrid pass — dict append → tuple append
    old = (
        '                    for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):\n'
        '                        if rate >= hybrid_threshold:  # use hybrid_threshold not threshold\n'
        '                            survivors_out.append({\n'
        "                                'seed': int(seed), 'family': family_name,\n"
        "                                'match_rate': float(rate),\n"
        "                                'matches': int(rate * k), 'total': k,\n"
        "                                'strategy_id': int(sid), 'skip_sequence': ss,\n"
        '                            })'
    )
    new = (
        '                    for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):\n'
        '                        if rate >= hybrid_threshold:  # use hybrid_threshold not threshold\n'
        '                            # [S150-slim_v1] tuple: (seed, match_rate, strategy_id, skip_seq)\n'
        '                            survivors_out.append((int(seed), float(rate), int(sid), list(ss)))'
    )
    content, ok = replace_exact(content, old, new,
        "hybrid pass: dict append → slim_v1 tuple")
    ok_count += ok

    # Fix 2c: constant-skip pass — dict append → tuple append
    old = (
        '                for seed, rate, skip in zip(s_arr, r_arr, k_arr):\n'
        '                    if rate >= threshold:\n'
        '                        survivors_out.append({\n'
        "                            'seed': int(seed), 'family': family_name,\n"
        "                            'match_rate': float(rate),\n"
        "                            'matches': int(rate * k), 'total': k,\n"
        "                            'best_skip': int(skip)\n"
        '                        })'
    )
    new = (
        '                for seed, rate, skip in zip(s_arr, r_arr, k_arr):\n'
        '                    if rate >= threshold:\n'
        '                        # [S150-slim_v1] tuple: (seed, match_rate, None=no_strategy, [best_skip])\n'
        '                        survivors_out.append((int(seed), float(rate), None, [int(skip)]))'
    )
    content, ok = replace_exact(content, old, new,
        "constant pass: dict append → slim_v1 tuple")
    ok_count += ok

    # Fix 2d: return flat slim_v1 dict (NOT nested under 'result' key)
    # run_worker() wraps as {"status":"ok","result": <this>}
    # coordinator reads inner = result.get("result",{}) then inner.get("format")
    # So "format" must be at TOP LEVEL of this return dict
    old = (
        '    return {\n'
        "        'job_id': job_id,\n"
        "        'success': True,\n"
        "        'survivors': all_survivors,\n"
        "        'seed_range': {'start': seed_start, 'end': seed_end},\n"
        "        'stats': {\n"
        "            'total_seeds_tested': total_tested,\n"
        "            'total_survivors': len(all_survivors),\n"
        "            'duration_ms': round(total_duration, 2),\n"
        "            'avg_seeds_per_sec': int(total_tested / (total_duration / 1000)) if total_duration > 0 else 0\n"
        "        },\n"
        "        'per_family': {f['family']: f for f in per_family}\n"
        '    }'
    )
    new = (
        '    # [S150-slim_v1] Build flat parallel-array result dict\n'
        '    # IMPORTANT: run_worker() wraps this as {"status":"ok","result":<this>}\n'
        '    # Coordinator reads inner=result.get("result",{}) then inner.get("format")\n'
        '    # So "format" and array fields MUST be at top level here (not nested under "result")\n'
        '    _seeds     = [t[0] for t in all_survivors]\n'
        '    _rates     = [t[1] for t in all_survivors]\n'
        '    # TB ruling: drive hybrid from job context, not survivor content\n'
        '    # Zero-survivor hybrid chunks must still emit strategy_ids/skip_sequences\n'
        '    _is_hybrid = (\n'
        '        "hybrid" in str(job.get("prng_type", "")).lower()\n'
        '        or str(job.get("skip_mode", "")).lower() == "hybrid"\n'
        '    )\n'
        '    _ret = {\n'
        "        'job_id':      job_id,\n"
        "        'success':     True,\n"
        "        'format':      'slim_v1',\n"
        "        'seeds':       _seeds,\n"
        "        'match_rates': _rates,\n"
        "        'seed_range':  {'start': seed_start, 'end': seed_end},\n"
        "        'stats': {\n"
        "            'total_seeds_tested': total_tested,\n"
        "            'total_survivors':    len(_seeds),\n"
        "            'duration_ms':        round(total_duration, 2),\n"
        "            'avg_seeds_per_sec':  int(total_tested / (total_duration / 1000)) if total_duration > 0 else 0\n"
        "        },\n"
        "        'per_family': {f['family']: f for f in per_family}\n"
        '    }\n'
        '    if _is_hybrid:\n'
        "        _ret['strategy_ids']   = [t[2] for t in all_survivors]\n"
        "        _ret['skip_sequences'] = [t[3] for t in all_survivors]\n"
        '    return _ret'
    )
    content, ok = replace_exact(content, old, new,
        "run_sieve_job return: flat slim_v1 dict (no nested 'result' key)")
    ok_count += ok

    log(f"\n  sieve_gpu_worker.py: {ok_count}/4 patches applied")
    if not DRY_RUN:
        with open(path, "w") as f: f.write(content)
        log(f"  wrote {path}")
    return ok_count == 4


def verify(base_dir, mode):
    log("\n[VERIFY]")

    coord_checks = []
    worker_checks = []

    if mode in ("coordinator", "both"):
        coord = open(os.path.join(base_dir, "persistent_worker_coordinator.py")).read()
        coord_checks = [
            ("coordinator: slim_v1 fast path",        'inner.get("format") == "slim_v1"' in coord),
            ("coordinator: legacy fallback present",  'Legacy path — list of dicts' in coord),
            ("coordinator: length assertion present", 'slim_v1 length mismatch' in coord),
            ("coordinator: hybrid enforcement present", 'slim_v1 hybrid payload missing' in coord),
        ]

    if mode in ("worker", "both"):
        worker = open(os.path.join(base_dir, "sieve_gpu_worker.py")).read()
        worker_checks = [
            ("worker: compact separators",            "separators=(',', ':')" in worker),
            ("worker: hybrid tuple append",           "survivors_out.append((int(seed), float(rate), int(sid), list(ss)))" in worker),
            ("worker: constant tuple append",         "survivors_out.append((int(seed), float(rate), None, [int(skip)]))" in worker),
            ("worker: flat slim_v1 format field",     "'format':      'slim_v1'" in worker),
            ("worker: hybrid arrays from job context", "_is_hybrid = (" in worker and "prng_type" in worker),
            ("worker: no nested result key",          "'result':  _slim_result" not in worker),
            ("worker: old hybrid dict removed",       "'strategy_id': int(sid), 'skip_sequence': ss," not in worker),
            ("worker: old constant dict removed",     "'best_skip': int(skip)" not in worker),
        ]

    all_checks = coord_checks + worker_checks
    errors = []
    for label, ok in all_checks:
        log(f"  {'✓' if ok else '✗'}  {label}")
        if not ok: errors.append(label)

    if errors:
        log(f"\n  FAIL — {len(errors)} error(s)")
        return False
    log(f"\n  PASS — all {len(all_checks)} checks green")
    return True


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--coordinator-only", action="store_true")
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    do_coord  = not args.worker_only
    do_worker = not args.coordinator_only
    if args.coordinator_only:   mode = "coordinator"
    elif args.worker_only:      mode = "worker"
    else:                       mode = "both"

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S150 slim_v1 IPC Serialization Fix")
    print(f"Mode: {mode} | TB approved Option A")
    print("=" * 60)

    coord_path  = os.path.join(args.base_dir, "persistent_worker_coordinator.py")
    worker_path = os.path.join(args.base_dir, "sieve_gpu_worker.py")

    r_coord  = patch_coordinator(coord_path) if do_coord  else True
    r_worker = patch_worker(worker_path)     if do_worker else True

    if not DRY_RUN:
        passed = verify(args.base_dir, mode)
    else:
        log("\n[VERIFY] skipped in dry-run")
        passed = r_coord and r_worker

    print("\n" + "=" * 60)
    if r_coord and r_worker and passed:
        print(f"✓ slim_v1 {mode} patch COMPLETE")
        if mode == "coordinator":
            print("\nStep 1 done. Deploy to Zeus and verify old workers still work.")
            print("Then run: python3 apply_s150_slim_v1_ipc.py --worker-only")
        elif mode == "worker":
            print("\nStep 2 done. Commit and git pull on all rigs.")
            print("  git add persistent_worker_coordinator.py sieve_gpu_worker.py apply_s150_slim_v1_ipc.py")
            print("  git commit -m 'fix(s150): slim_v1 IPC — 15x payload reduction on high-survivor passes'")
            print("  git push origin main && git push public main")
            print("  ssh rrig6600  'cd ~/distributed_prng_analysis && git pull'")
            print("  ssh rrig6600b 'cd ~/distributed_prng_analysis && git pull'")
            print("  ssh rrig6600c 'cd ~/distributed_prng_analysis && git pull'")
        else:
            print("\nBoth patches done. Commit and deploy.")
        return True
    else:
        print("✗ INCOMPLETE — review errors above")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
