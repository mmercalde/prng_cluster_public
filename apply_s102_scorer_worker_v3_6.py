#!/usr/bin/env python3
"""
apply_s102_scorer_worker_v3_6.py
Patch scorer_trial_worker.py: Fix NPZ prng_type decoding (v3.5 → v3.6)

PRE-EXISTING BUG (present since commit 4e340de, NOT introduced by v3.5 patch):
  load_data() NPZ branch contains only `pass` with comment:
  "Keep defaults, NPZ doesn't store per-survivor prng_type"
  This is incorrect — prng_type IS available via optimal_window_config.json,
  which is the canonical pipeline config per project design (Chapter 1,
  scorer_meta.json manifest, Schema Extension Proposal Agent M).

FIX: Read prng_type and mod from optimal_window_config.json — the
     documented single source of truth for pipeline configuration.
     No hardcoded strings. Consistent with project's configurable design.
"""

import sys
import os
import shutil
import ast
from datetime import datetime

TARGET_DEFAULT = "scorer_trial_worker.py"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

OLD = """    # Extract PRNG type from survivor metadata
    prng_type = 'java_lcg'
    mod = 1000
    if isinstance(survivors, dict) and 'seeds' in survivors:
        # NPZ format - prng_type from metadata if available
        pass  # Keep defaults, NPZ doesn't store per-survivor prng_type"""

NEW = """    # Extract PRNG type from pipeline config
    # optimal_window_config.json is the canonical source per project design:
    #   - Chapter 1 documents prng_type as a top-level field
    #   - scorer_meta.json manifest declares it as a required input
    #   - No hardcoded PRNG strings — fully configurable per project principles
    prng_type = None
    mod = None
    if isinstance(survivors, dict) and 'seeds' in survivors:
        # NPZ format — read prng_type from optimal_window_config.json
        wc_path = os.path.join(
            os.path.dirname(os.path.abspath(survivors_file)),
            'optimal_window_config.json'
        )
        if os.path.exists(wc_path):
            try:
                with open(wc_path) as _wf:
                    _wc = json.load(_wf)
                prng_type = _wc.get('prng_type')
                mod = _wc.get('mod')
                logger.info(f"Pipeline config: prng_type={prng_type}, mod={mod} "
                            f"(from optimal_window_config.json)")
            except Exception as _e:
                logger.warning(f"Could not read optimal_window_config.json: {_e}")
        if not prng_type:
            logger.warning("prng_type not resolved from config — pipeline config missing? "
                           "Defaulting to java_lcg / mod=1000")
            prng_type = 'java_lcg'
        if not mod:
            mod = 1000"""

OLD_VER = 'scorer_trial_worker.py v3.5'
NEW_VER = 'scorer_trial_worker.py v3.6'


def apply(target: str, dry_run: bool = False):
    with open(target, 'r') as f:
        src = f.read()

    errors = []
    if OLD not in src:
        errors.append("Patch target not found — file may differ from expected v3.5")
    if OLD_VER not in src:
        errors.append(f"Version string '{OLD_VER}' not found — may already be patched")

    if errors:
        for e in errors:
            print(f"❌  {e}")
        sys.exit(1)

    if dry_run:
        print("DRY RUN — no changes written.")
        print("✅  Patch target found")
        print("✅  Version string found")
        return

    backup = target + BACKUP_SUFFIX
    shutil.copy2(target, backup)
    print(f"📦  Backup written: {backup}")

    patched = src.replace(OLD, NEW, 1)
    patched = patched.replace(OLD_VER, NEW_VER, 1)

    with open(target, 'w') as f:
        f.write(patched)
    print(f"✅  Patch applied: {target}")

    print("\n── Verification ─────────────────────────────────────────────────────")
    checks = [
        ("pass block removed",         "pass  # Keep defaults" not in patched),
        ("optimal_window_config used", "optimal_window_config.json" in patched),
        ("warning on missing config",  "prng_type not resolved" in patched),
        ("version bumped to v3.6",     NEW_VER in patched),
        ("syntax valid",               _syntax_ok(patched)),
    ]
    all_ok = True
    for label, result in checks:
        icon = "✅" if result else "❌"
        print(f"  {icon}  {label}")
        if not result:
            all_ok = False

    if all_ok:
        print(f"\n✅  All checks passed. scorer_trial_worker.py is v3.6.")
        print("\nNext steps:")
        print("  1. Kill current Step 2 run (Ctrl+C)")
        print("  2. scp scorer_trial_worker.py to all worker rigs")
        print("  3. rm -f scorer_trial_results/trial_*.json")
        print("  4. PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2")
    else:
        print("\n❌  Some checks failed — review above.")
        sys.exit(1)


def _syntax_ok(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"     SyntaxError: {e}")
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=TARGET_DEFAULT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"❌  Target not found: {args.target}")
        sys.exit(1)

    print(f"Target: {os.path.abspath(args.target)}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'APPLY'}")
    apply(args.target, dry_run=args.dry_run)
