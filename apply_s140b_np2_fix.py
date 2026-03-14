#!/usr/bin/env python3
"""
S140b n_parallel>1 Trial History & Dynamic Warm-Start Fix
==========================================================
Fixes the production gap where trial history writes and dynamic warm-start
were silently skipped because n_parallel=2 uses a separate multiprocessing
path that bypassed all S140b patches.

Changes (window_optimizer_integration_final.py only):
  A. Build warm_start_params at n_parallel>1 block start
  B. Add warm_start_params to _partition_worker signature
  C. Add trial history write inside _worker_obj (child-local DB connection)
  D. Replace hardcoded W8_O43 warm-start with dynamic + fallback
  E. Append warm_start_params to _mp.Process args tuple

Team Beta approved. Guardrails:
  - Fork model unchanged
  - _worker_obj stays inside _partition_worker
  - Trial history write is non-fatal
  - Fallback to W8_O43 when no prior history

Usage (from ~/distributed_prng_analysis/):
  python3 apply_s140b_np2_fix.py [--dry-run]
"""

import sys, shutil, subprocess
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv

def backup(path):
    bak = path + '.bak_s140b_np2'
    if not DRY_RUN:
        shutil.copy2(path, bak)
    print(f"  BAK  {bak}")

def write(path, content):
    if not DRY_RUN:
        with open(path, 'w') as f:
            f.write(content)
    print(f"  {'DRY' if DRY_RUN else 'WRT'} {path}")

def check(condition, msg):
    if not condition:
        print(f"  ABORT: {msg}")
        sys.exit(1)


def patch():
    print("\n[1/1] window_optimizer_integration_final.py — n_parallel>1 trial history + warm-start")
    path = 'window_optimizer_integration_final.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'S140b-NP2' in content:
        print("  SKIP already patched")
        r = subprocess.run(['python3', '-m', 'py_compile', path], capture_output=True)
        print(f"  {'✅' if r.returncode == 0 else '❌'} syntax: {path}")
        return

    # A — build warm_start_params at block start
    old_a = ('        if n_parallel > 1:\n'
             '            import multiprocessing as _mp\n'
             '            import glob as _mpglob\n'
             '            import time as _mptime')
    new_a = ('        if n_parallel > 1:\n'
             '            import multiprocessing as _mp\n'
             '            import glob as _mpglob\n'
             '            import time as _mptime\n'
             '\n'
             '            # [S140b-NP2] Build warm_start_params from DB\n'
             '            # Read directly — trial_history_context not in optimize_window scope\n'
             '            _warm_start_params = None\n'
             '            try:\n'
             '                from database_system import DistributedPRNGDatabase as _DBNP2\n'
             '                _db_np2 = _DBNP2()\n'
             '                _best_np2 = _db_np2.get_best_step1_params(prng_base, limit=1)\n'
             '                if _best_np2:\n'
             '                    _bp_np2 = _best_np2[0]\n'
             "                    if all(_bp_np2.get(k) is not None for k in\n"
             "                           ['window_size','offset','skip_min','skip_max',\n"
             "                            'forward_threshold','reverse_threshold']):\n"
             '                        _warm_start_params = {\n'
             "                            'window_size':       int(_bp_np2['window_size']),\n"
             "                            'offset':            int(_bp_np2['offset']),\n"
             "                            'skip_min':          int(_bp_np2['skip_min']),\n"
             "                            'skip_max':          int(_bp_np2['skip_max']),\n"
             "                            'forward_threshold': float(_bp_np2['forward_threshold']),\n"
             "                            'reverse_threshold': float(_bp_np2['reverse_threshold']),\n"
             '                        }\n'
             '            except Exception as _e_np2:\n'
             "                print(f'   [n_parallel] warm_start DB lookup failed: {_e_np2}')")
    check(old_a in content, "anchor A not found")
    content = content.replace(old_a, new_a)
    print("  A OK — warm_start_params built")

    # B — add to _partition_worker signature
    old_b = ('            def _partition_worker(partition_idx, allowlist, config_file_w,\n'
             '                                   dataset_path_w, seed_start_w, seed_count_w,\n'
             '                                   prng_base_w, test_both_modes_w,\n'
             '                                   storage_url, study_name_w, trials_for_worker,\n'
             '                                   result_queue, temp_file):')
    new_b = ('            def _partition_worker(partition_idx, allowlist, config_file_w,\n'
             '                                   dataset_path_w, seed_start_w, seed_count_w,\n'
             '                                   prng_base_w, test_both_modes_w,\n'
             '                                   storage_url, study_name_w, trials_for_worker,\n'
             '                                   result_queue, temp_file,\n'
             '                                   warm_start_params=None):  # [S140b-NP2]')
    check(old_b in content, "anchor B not found")
    content = content.replace(old_b, new_b)
    print("  B OK — warm_start_params in _partition_worker signature")

    # C — trial history write inside _worker_obj
    old_c = ('                        trial.set_user_attr("result_dict", result.to_dict())\n'
             '                        print(f"   [P{partition_idx}] Trial {trial.number}: "\n'
             '                              f"{cfg.description()} score={score:.0f}")\n'
             '                        return score')
    new_c = ('                        trial.set_user_attr("result_dict", result.to_dict())\n'
             '                        print(f"   [P{partition_idx}] Trial {trial.number}: "\n'
             '                              f"{cfg.description()} score={score:.0f}")\n'
             '                        # [S140b-NP2] Trial history — child-local DB connection\n'
             '                        try:\n'
             '                            from database_system import DistributedPRNGDatabase as _DBTH\n'
             '                            _db_th = _DBTH()\n'
             '                            _sess = (",".join(cfg.sessions)\n'
             '                                     if isinstance(cfg.sessions, (list, tuple))\n'
             '                                     else str(cfg.sessions))\n'
             '                            _db_th.write_step1_trial(\n'
             '                                run_id=f"step1_{prng_base_w}_{int(seed_start_w)}",\n'
             '                                study_name=study_name_w,\n'
             '                                trial_number=int(trial.number),\n'
             '                                prng_type=str(prng_base_w),\n'
             '                                seed_range_start=int(seed_start_w),\n'
             '                                seed_range_end=int(seed_start_w + seed_count_w - 1),\n'
             '                                params={\n'
             "                                    'window_size': cfg.window_size,\n"
             "                                    'offset': cfg.offset,\n"
             "                                    'skip_min': cfg.skip_min,\n"
             "                                    'skip_max': cfg.skip_max,\n"
             "                                    'time_of_day': _sess,\n"
             "                                    'forward_threshold': cfg.forward_threshold,\n"
             "                                    'reverse_threshold': cfg.reverse_threshold,\n"
             '                                },\n'
             '                                trial_score=float(score),\n'
             '                                forward_survivors=int(\n'
             '                                    getattr(result, "forward_count", 0)),\n'
             '                                reverse_survivors=int(\n'
             '                                    getattr(result, "reverse_count", 0)),\n'
             '                                bidirectional_survivors=int(\n'
             '                                    getattr(result, "bidirectional_count", 0)),\n'
             '                                pruned=False\n'
             '                            )\n'
             '                        except Exception as _th_e:\n'
             '                            print(f"   [P{partition_idx}] trial-history write "\n'
             '                                  f"failed (non-fatal): {_th_e}")\n'
             '                        return score')
    check(old_c in content, "anchor C not found")
    content = content.replace(old_c, new_c)
    print("  C OK — trial history write in _worker_obj")

    # D — dynamic warm-start enqueue
    old_d = ('                if len(_setup_study.trials) == 0:\n'
             '                    _setup_study.enqueue_trial({\n'
             "                        'window_size': 8, 'offset': 43,\n"
             "                        'skip_min': 5, 'skip_max': 56,\n"
             "                        'forward_threshold': 0.49, 'reverse_threshold': 0.49\n"
             '                    })\n'
             '                    print("   [n_parallel] Warm-start enqueued (W8_O43_S5-56)")')
    new_d = ('                if len(_setup_study.trials) == 0:\n'
             '                    # [S140b-NP2] Dynamic warm-start with fallback\n'
             '                    _ws_trial = dict(_warm_start_params) if _warm_start_params else {}\n'
             '                    if _ws_trial:\n'
             '                        _setup_study.enqueue_trial(_ws_trial)\n'
             '                        print(\n'
             '                            f"   [n_parallel] Warm-start enqueued "\n'
             "                            f\"(W{_ws_trial.get('window_size')}_\"\n"
             "                            f\"O{_ws_trial.get('offset')}_\"\n"
             "                            f\"S{_ws_trial.get('skip_min')}-\"\n"
             "                            f\"{_ws_trial.get('skip_max')})\"  # [S140b-NP2]\n"
             '                        )\n'
             '                    else:\n'
             '                        _setup_study.enqueue_trial({\n'
             "                            'window_size': 8, 'offset': 43,\n"
             "                            'skip_min': 5, 'skip_max': 56,\n"
             "                            'forward_threshold': 0.49, 'reverse_threshold': 0.49\n"
             '                        })\n'
             '                        print("   [n_parallel] Warm-start fallback enqueued (W8_O43_S5-56)")')
    check(old_d in content, "anchor D not found")
    content = content.replace(old_d, new_d)
    print("  D OK — dynamic warm-start with fallback")

    # E — append warm_start_params to _mp.Process args
    old_e = ('                    args=(\n'
             '                        _pi,\n'
             '                        _PARALLEL_PARTITIONS[_pi],\n'
             '                        getattr(self, \'config_file\', \'distributed_config.json\'),\n'
             '                        dataset_path, seed_start, seed_count,\n'
             '                        prng_base, test_both_modes,\n'
             '                        _mp_storage_url, _mp_study_name,\n'
             '                        _trials_per_worker[_pi],\n'
             '                        _rq,\n'
             '                        f\'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json\',\n'
             '                    ),')
    new_e = ('                    args=(\n'
             '                        _pi,\n'
             '                        _PARALLEL_PARTITIONS[_pi],\n'
             '                        getattr(self, \'config_file\', \'distributed_config.json\'),\n'
             '                        dataset_path, seed_start, seed_count,\n'
             '                        prng_base, test_both_modes,\n'
             '                        _mp_storage_url, _mp_study_name,\n'
             '                        _trials_per_worker[_pi],\n'
             '                        _rq,\n'
             '                        f\'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json\',\n'
             '                        _warm_start_params,  # [S140b-NP2]\n'
             '                    ),')
    check(old_e in content, "anchor E not found")
    content = content.replace(old_e, new_e)
    print("  E OK — warm_start_params in _mp.Process args")

    write(path, content)
    r = subprocess.run(['python3', '-m', 'py_compile', path], capture_output=True)
    print(f"  {'✅' if r.returncode == 0 else '❌'} syntax: {path}")
    if r.returncode != 0:
        print(f"  {r.stderr.decode()}")
        sys.exit(1)
    print("  OK")


if __name__ == '__main__':
    if DRY_RUN:
        print("DRY RUN — no files modified\n")
    if not Path('window_optimizer.py').exists():
        print("ERROR: Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    patch()

    print("\n" + "="*55)
    print("✅ S140b n_parallel>1 fix applied")
    print("\nNext:")
    print("  git add window_optimizer_integration_final.py")
    print("  git commit -m 'S140b-NP2: trial history + warm-start for n_parallel>1 path'")
    print("  git push origin main && git push public main")
