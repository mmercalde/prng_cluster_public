#!/usr/bin/env python3
"""
S140 Seed Coverage Tracker — Deploy Script
==========================================
Patches 4 files to prevent re-searching the same seed range on every Step 1 run.

Files patched:
  1. database_system.py         — add get_next_seed_start()
  2. window_optimizer.py        — add seed_start param + CLI arg + write-back
  3. agents/watcher_agent.py    — add coverage preflight block
  4. agent_manifests/window_optimizer.json — add seed_start to manifest

Usage (from ~/distributed_prng_analysis/):
  python3 apply_s140_seed_coverage_tracker.py [--dry-run]

Run from project root. Creates .bak_s140 backups before patching.
"""

import sys, os, json, shutil, argparse
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv

def backup(path):
    bak = path + '.bak_s140'
    if not DRY_RUN:
        shutil.copy2(path, bak)
    print(f"  BAK  {bak}")

def write(path, content):
    if not DRY_RUN:
        with open(path, 'w') as f:
            f.write(content)
    print(f"  {'DRY' if DRY_RUN else 'WRT'} {path}")

def verify_lines(path, expected_count, label):
    with open(path) as f:
        lines = f.readlines()
    actual = len(lines)
    status = "OK" if actual == expected_count else f"WARN (expected {expected_count}, got {actual})"
    print(f"  LNC  {label}: {actual} lines — {status}")

def patch_database_system():
    print("\n[1/4] database_system.py — add get_next_seed_start()")
    path = 'database_system.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'get_next_seed_start' in content:
        print("  SKIP already patched")
        return

    anchor = '''    def get_exhaustive_progress(self, search_id: str) -> List[Dict]:
        """Get progress for exhaustive search"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(\'\'\'
                SELECT * FROM exhaustive_progress WHERE search_id=?
                ORDER BY seed_range_start
            \'\'\', (search_id,))
            
            return [dict(row) for row in cursor.fetchall()]'''

    new_method = '''

    def get_next_seed_start(self, prng_type: str, chunk_size: int) -> int:
        """
        [S140] Seed Coverage Tracker — returns the next uncovered seed_start
        for a given prng_type across ALL prior runs.

        Queries MAX(seed_range_end) from exhaustive_progress for this prng_type.
        If no prior coverage exists, returns 0 (start from beginning).

        Args:
            prng_type:  PRNG identifier e.g. \'java_lcg\', \'mt19937\'
            chunk_size: Size of each search chunk (logged for context only)

        Returns:
            int: Next seed_start to use (0 if no prior coverage recorded)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    \'SELECT MAX(seed_range_end) FROM exhaustive_progress WHERE prng_type = ?\',
                    (prng_type,)
                ).fetchone()
                if result and result[0] is not None:
                    next_start = int(result[0])
                    import logging
                    logging.getLogger(__name__).info(
                        f"[COVERAGE] {prng_type}: prior coverage up to {next_start:,} — "
                        f"next seed_start={next_start:,}"
                    )
                    return next_start
                else:
                    import logging
                    logging.getLogger(__name__).info(
                        f"[COVERAGE] {prng_type}: no prior coverage — starting at seed 0"
                    )
                    return 0
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"[COVERAGE] get_next_seed_start failed: {e} — defaulting to seed_start=0"
            )
            return 0'''

    assert anchor in content, "ABORT: anchor not found in database_system.py"
    new_content = content.replace(anchor, anchor + new_method)
    write(path, new_content)
    verify_lines(path, len(new_content.splitlines()), 'database_system.py')
    print("  OK")


def patch_window_optimizer():
    print("\n[2/4] window_optimizer.py — seed_start param + CLI + write-back")
    path = 'window_optimizer.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if '[S140]' in content:
        print("  SKIP already patched")
        return

    # 2a — add seed_start to function signature
    old_sig = ('def run_bayesian_optimization(\n'
               '    lottery_file: str,\n'
               '    trials: int,\n'
               '    output_config: str,\n'
               '    seed_count: int = 10_000_000,\n'
               "    prng_type: str = 'java_lcg',")
    new_sig = ('def run_bayesian_optimization(\n'
               '    lottery_file: str,\n'
               '    trials: int,\n'
               '    output_config: str,\n'
               '    seed_count: int = 10_000_000,\n'
               '    seed_start: int = 0,                   # [S140] coverage tracker base\n'
               "    prng_type: str = 'java_lcg',")
    assert old_sig in content, "ABORT: signature anchor not found"
    content = content.replace(old_sig, new_sig)

    # 2b — pass seed_start into optimize_window
    old_call = ('    results = coordinator.optimize_window(\n'
                '        dataset_path=lottery_file,\n'
                '        seed_start=0,\n'
                '        seed_count=seed_count,')
    new_call = ('    results = coordinator.optimize_window(\n'
                '        dataset_path=lottery_file,\n'
                '        seed_start=seed_start,             # [S140] from coverage tracker\n'
                '        seed_count=seed_count,')
    assert old_call in content, "ABORT: optimize_window call anchor not found"
    content = content.replace(old_call, new_call)

    # 2c — add --seed-start CLI arg
    old_cli = ("    parser.add_argument('--seed-cap-amd', type=int, default=2_000_000,\n"
               "                       help='[S137] Max seeds per job chunk for AMD GPUs (default: 2000000).')\n"
               "\n"
               "    args = parser.parse_args()")
    new_cli = ("    parser.add_argument('--seed-cap-amd', type=int, default=2_000_000,\n"
               "                       help='[S137] Max seeds per job chunk for AMD GPUs (default: 2000000).')\n"
               "    parser.add_argument('--seed-start', type=int, default=0,\n"
               "                       help='[S140] Starting seed for search range. Set automatically by '\n"
               "                            'WATCHER coverage tracker to advance into unexplored seed space. '\n"
               "                            'Default 0.')\n"
               "\n"
               "    args = parser.parse_args()")
    assert old_cli in content, "ABORT: CLI anchor not found"
    content = content.replace(old_cli, new_cli)

    # 2d — pass seed_start through CLI bayesian call site
    old_bc = ("        results = run_bayesian_optimization(\n"
              "            lottery_file=args.lottery_file,\n"
              "            trials=args.trials,\n"
              "            output_config=args.output,\n"
              "            seed_count=args.max_seeds if args.max_seeds else 10_000_000,\n"
              "            prng_type=args.prng_type,\n"
              "            test_both_modes=args.test_both_modes,\n"
              "            resume_study=getattr(args, 'resume_study', False),\n"
              "            study_name=getattr(args, 'study_name', ''),\n"
              "            enable_pruning=getattr(args, 'enable_pruning', False),\n"
              "            n_parallel=getattr(args, 'n_parallel', 1),\n"
              "            trse_context_file=getattr(args, 'trse_context', 'trse_context.json'),\n"
              "            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134\n"
              "            worker_pool_size=getattr(args, 'worker_pool_size', 8),                  # S134\n"
              "            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),            # S137\n"
              "            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),                  # S137\n"
              "        )")
    new_bc = ("        results = run_bayesian_optimization(\n"
              "            lottery_file=args.lottery_file,\n"
              "            trials=args.trials,\n"
              "            output_config=args.output,\n"
              "            seed_count=args.max_seeds if args.max_seeds else 10_000_000,\n"
              "            seed_start=getattr(args, 'seed_start', 0),                              # S140\n"
              "            prng_type=args.prng_type,\n"
              "            test_both_modes=args.test_both_modes,\n"
              "            resume_study=getattr(args, 'resume_study', False),\n"
              "            study_name=getattr(args, 'study_name', ''),\n"
              "            enable_pruning=getattr(args, 'enable_pruning', False),\n"
              "            n_parallel=getattr(args, 'n_parallel', 1),\n"
              "            trse_context_file=getattr(args, 'trse_context', 'trse_context.json'),\n"
              "            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134\n"
              "            worker_pool_size=getattr(args, 'worker_pool_size', 8),                  # S134\n"
              "            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),            # S137\n"
              "            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),                  # S137\n"
              "        )")
    assert old_bc in content, "ABORT: bayesian CLI call site anchor not found"
    content = content.replace(old_bc, new_bc)

    # 2e — coverage write-back after optimize_window completes
    old_wb = "    # Save optimal config for downstream use\n    best_config = results['best_config']"
    new_wb = (
        "    # [S140] SEED COVERAGE WRITE-BACK — log this run's range to exhaustive_progress\n"
        "    # Runs once per Step 1 completion. Enables WATCHER to advance seed_start next run.\n"
        "    try:\n"
        "        from database_system import DistributedPRNGDatabase as _DBM\n"
        "        _db = _DBM()\n"
        "        _best_result = results.get('best_result', {})\n"
        "        _survivors = _best_result.get('bidirectional_survivors', [])\n"
        "        _best_seed = None\n"
        "        if _survivors and isinstance(_survivors[0], dict):\n"
        "            _best_seed = _survivors[0].get('seed', None)\n"
        "        elif _survivors and isinstance(_survivors[0], int):\n"
        "            _best_seed = _survivors[0]\n"
        "        _db.update_exhaustive_progress(\n"
        "            search_id=f'step1_{prng_type}_{int(seed_start)}',\n"
        "            prng_type=prng_type,\n"
        "            mapping_type='bidirectional',\n"
        "            seed_range_start=seed_start,\n"
        "            seed_range_end=seed_start + seed_count,\n"
        "            seeds_completed=seed_count,\n"
        "            best_score=results.get('best_score'),\n"
        "            best_seed=_best_seed\n"
        "        )\n"
        "        print(f'   [COVERAGE] Logged range {seed_start:,} → {seed_start + seed_count:,} '\n"
        "              f'for {prng_type} (best_seed={_best_seed})')\n"
        "    except Exception as _e:\n"
        "        print(f'   [COVERAGE] Write-back failed (non-fatal): {_e}')\n"
        "\n"
        "    # Save optimal config for downstream use\n"
        "    best_config = results['best_config']"
    )
    assert old_wb in content, "ABORT: write-back anchor not found"
    content = content.replace(old_wb, new_wb)

    write(path, content)
    verify_lines(path, len(content.splitlines()), 'window_optimizer.py')
    print("  OK")


def patch_watcher_agent():
    print("\n[3/4] agents/watcher_agent.py — add coverage preflight block")
    path = 'agents/watcher_agent.py'
    backup(path)
    with open(path) as f:
        content = f.read()

    if 'SEED COVERAGE TRACKER' in content:
        print("  SKIP already patched")
        return

    old_anchor = ('        # Remove output_file if present (use script default)\n'
                  '        final_params.pop("output_file", None)\n'
                  '\n'
                  '        # Build command')

    new_anchor = (
        '        # Remove output_file if present (use script default)\n'
        '        final_params.pop("output_file", None)\n'
        '\n'
        '        # [S140] SEED COVERAGE TRACKER — Step 1 only\n'
        '        # Reads MAX(seed_range_end) for this prng_type from exhaustive_progress.\n'
        '        # Advances seed_start between pipeline runs so we never re-search covered ranges.\n'
        '        # If DB lookup fails for any reason, defaults to 0 — run proceeds normally.\n'
        '        # Invariant: new seed range forces fresh study (resume_study=False, study_name=\'\')\n'
        '        if step == 1 and \'seed_start\' in final_params:\n'
        '            try:\n'
        '                import sys as _sys\n'
        '                _sys.path.insert(0, str(Path(__file__).parent.parent))\n'
        '                from database_system import DistributedPRNGDatabase as _DBM\n'
        '                _db = _DBM()\n'
        '                _prng_type = final_params.get(\'prng_type\', \'java_lcg\')\n'
        '                _chunk_size = final_params.get(\'max_seeds\', 5_000_000)\n'
        '                _next_start = _db.get_next_seed_start(_prng_type, _chunk_size)\n'
        '                if _next_start > 0:\n'
        '                    final_params[\'seed_start\'] = _next_start\n'
        '                    final_params[\'resume_study\'] = False   # INVARIANT: new range = fresh study\n'
        '                    final_params[\'study_name\'] = \'\'        # force fresh study name\n'
        '                    logger.info(\n'
        '                        f"[COVERAGE] Step 1: advancing seed_start to {_next_start:,} "\n'
        '                        f"for {_prng_type} — forcing fresh study"\n'
        '                    )\n'
        '                else:\n'
        '                    logger.info(\n'
        '                        f"[COVERAGE] Step 1: no prior coverage for "\n'
        '                        f"{_prng_type} — using seed_start=0"\n'
        '                    )\n'
        '            except Exception as _e:\n'
        '                logger.warning(\n'
        '                    f"[COVERAGE] Seed coverage lookup failed: {_e} — using seed_start=0"\n'
        '                )\n'
        '\n'
        '        # Build command'
    )

    assert old_anchor in content, "ABORT: anchor not found in watcher_agent.py"
    new_content = content.replace(old_anchor, new_anchor)
    write(path, new_content)
    verify_lines(path, len(new_content.splitlines()), 'watcher_agent.py')
    print("  OK")


def patch_manifest():
    print("\n[4/4] agent_manifests/window_optimizer.json — add seed_start")
    path = 'agent_manifests/window_optimizer.json'
    backup(path)
    with open(path) as f:
        manifest = json.load(f)

    changed = False

    if 'seed_start' not in manifest.get('default_params', {}):
        manifest['default_params']['seed_start'] = 0
        changed = True
        print("  ADD default_params.seed_start = 0")

    if 'seed-start' not in manifest.get('args_map', {}):
        manifest.setdefault('args_map', {})['seed-start'] = 'seed_start'
        changed = True
        print("  ADD args_map.seed-start = seed_start")

    for i, action in enumerate(manifest.get('actions', [])):
        if 'seed-start' not in action.get('args_map', {}):
            action.setdefault('args_map', {})['seed-start'] = 'seed_start'
            changed = True
            print(f"  ADD actions[{i}].args_map.seed-start = seed_start")

    if 'parameter_bounds' in manifest and 'seed_start' not in manifest['parameter_bounds']:
        manifest['parameter_bounds']['seed_start'] = {
            "type": "int",
            "min": 0,
            "max": 4294967295,
            "default": 0,
            "description": "[S140] Starting seed for search range. Set automatically by WATCHER coverage tracker to advance into unexplored seed space. Manual override allowed.",
            "optimized_by": "WATCHER"
        }
        changed = True
        print("  ADD parameter_bounds.seed_start")

    if changed:
        new_content = json.dumps(manifest, indent=2)
        write(path, new_content)
        print("  OK")
    else:
        print("  SKIP already patched")


def verify_syntax():
    print("\n=== Syntax verification ===")
    import subprocess
    files = [
        ('python3 -m py_compile database_system.py',      'database_system.py'),
        ('python3 -m py_compile window_optimizer.py',     'window_optimizer.py'),
        ('python3 -m py_compile agents/watcher_agent.py', 'watcher_agent.py'),
    ]
    all_ok = True
    for cmd, label in files:
        r = subprocess.run(cmd.split(), capture_output=True)
        status = "✅" if r.returncode == 0 else "❌"
        print(f"  {status} {label}")
        if r.returncode != 0:
            print(f"     {r.stderr.decode()}")
            all_ok = False
    # JSON
    try:
        with open('agent_manifests/window_optimizer.json') as f:
            json.load(f)
        print("  ✅ window_optimizer.json")
    except Exception as e:
        print(f"  ❌ window_optimizer.json: {e}")
        all_ok = False
    return all_ok


if __name__ == '__main__':
    if DRY_RUN:
        print("DRY RUN — no files will be modified\n")

    # Confirm we're in the right directory
    if not Path('window_optimizer.py').exists():
        print("ERROR: Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    patch_database_system()
    patch_window_optimizer()
    patch_watcher_agent()
    patch_manifest()

    ok = verify_syntax()

    print("\n" + ("="*50))
    if ok:
        print("✅ S140 Seed Coverage Tracker — all patches applied")
        print("\nNext step: commit and dual-push")
        print("  git add database_system.py window_optimizer.py agents/watcher_agent.py agent_manifests/window_optimizer.json")
        print("  git commit -m 'S140: seed coverage tracker — prevent Step 1 seed range re-search'")
        print("  git push origin main && git push public main")
    else:
        print("❌ One or more patches failed — check output above")
        sys.exit(1)
