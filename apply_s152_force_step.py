#!/usr/bin/env python3
"""
apply_s152_force_step.py
========================
Patch: Add --force-step N flag to watcher_agent.py

PROBLEM
-------
check_output_freshness() skips a step if its output file already exists
and is newer than all inputs. This means every resume or restart of Step 1
gets silently skipped unless you manually run:
    rm -f optimal_window_config.json

Workaround is fragile and easy to forget, causing silent no-ops on restart.

FIX
---
Add --force-step N (repeatable) to watcher_agent.py's argparse block.

When --force-step 1 is passed:
  - The freshness check for step 1 is bypassed (is_fresh treated as False)
  - Hard freshness failures (missing input) still block — force-step only
    overrides "output is already fresh, skipping" logic
  - A clear log message is printed: [FORCE-STEP] Bypassing freshness for step N

The forced step list is passed via WatcherConfig (new field: force_steps: set)
and consumed inside run_step().

sweep_run1.sh --resume automatically adds --force-step 1 so you never need
to remember to rm the config file again.

Files patched
-------------
  agents/watcher_agent.py
  sweep_run1.sh

Backups
-------
  agents/watcher_agent.py.bak_s152_force_step
  sweep_run1.sh.bak_s152_force_step
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

WATCHER  = Path("agents/watcher_agent.py")
SWEEP_R1 = Path("sweep_run1.sh")

WATCHER_BACKUP  = Path("agents/watcher_agent.py.bak_s152_force_step")
SWEEP_BACKUP    = Path("sweep_run1.sh.bak_s152_force_step")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 of 4 — add force_steps field to WatcherConfig
# We look for the class WatcherConfig definition and add the field.
# ─────────────────────────────────────────────────────────────────────────────
OLD_WATCHER_CONFIG_HALT = '''\
    halt_file: str = "/tmp/agent_halt"'''

NEW_WATCHER_CONFIG_HALT = '''\
    halt_file: str = "/tmp/agent_halt"
    force_steps: set = None  # [S152] steps to bypass freshness check'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 of 4 — bypass freshness inside run_step() when step is force-stepped
# Insert immediately before the freshness check block.
# ─────────────────────────────────────────────────────────────────────────────
OLD_FRESHNESS_BLOCK = '''\
        # FRESHNESS CHECK (Phase 1 Patch v1.1.2)
        is_fresh, freshness_msg, is_hard_freshness = check_output_freshness(step)'''

NEW_FRESHNESS_BLOCK = '''\
        # FRESHNESS CHECK (Phase 1 Patch v1.1.2)
        # [S152] --force-step bypass: treat output as stale for forced steps
        _force_steps = getattr(self.config, 'force_steps', None) or set()
        if step in _force_steps:
            print(f"[FORCE-STEP] Bypassing freshness check for step {step} (--force-step)")
            is_fresh, freshness_msg, is_hard_freshness = False, f"FORCE-STEP: step {step} forced", False
        else:
            is_fresh, freshness_msg, is_hard_freshness = check_output_freshness(step)'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 of 4 — add --force-step to argparse
# Insert after the --params argument block.
# ─────────────────────────────────────────────────────────────────────────────
OLD_PARAMS_ARG = '''\
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON string of params to override manifest defaults"
    )


    # Phase 7 Part B: Dispatch commands'''

NEW_PARAMS_ARG = '''\
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON string of params to override manifest defaults"
    )
    parser.add_argument(
        "--force-step",
        type=int,
        action="append",
        dest="force_steps",
        metavar="N",
        default=[],
        help="Force step N to re-run even if output is fresh (repeatable: --force-step 1 --force-step 2)"
    )

    # Phase 7 Part B: Dispatch commands'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 of 4 — wire args.force_steps into WatcherConfig
# Find where config is constructed after parse_args and add force_steps.
# ─────────────────────────────────────────────────────────────────────────────
OLD_CONFIG_BUILD = '''\
    config = WatcherConfig(
        auto_proceed_threshold=args.threshold,
        use_llm=not args.no_llm,
        use_grammar=not args.no_grammar,'''

NEW_CONFIG_BUILD = '''\
    config = WatcherConfig(
        auto_proceed_threshold=args.threshold,
        use_llm=not args.no_llm,
        use_grammar=not args.no_grammar,
        force_steps=set(args.force_steps) if args.force_steps else set(),  # [S152]'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH for sweep_run1.sh --resume: auto-add --force-step 1
# ─────────────────────────────────────────────────────────────────────────────
OLD_RESUME_LAUNCH = '''\
    PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt 2>/dev/null || true

    # Re-launch
    nohup bash -c "PYTHONPATH=. python3 agents/watcher_agent.py \\
        --run-pipeline --start-step 1 --end-step 1 \\
        >> $LOG 2>&1" &'''

NEW_RESUME_LAUNCH = '''\
    PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt 2>/dev/null || true

    # Re-launch — [S152] --force-step 1 bypasses freshness check automatically on resume
    nohup bash -c "PYTHONPATH=. python3 agents/watcher_agent.py \\
        --run-pipeline --start-step 1 --end-step 1 --force-step 1 \\
        >> $LOG 2>&1" &'''


def apply():
    w_src = WATCHER.read_text()
    s_src = SWEEP_R1.read_text()

    # ── Idempotency checks ──
    if "[S152]" in w_src and "force_steps" in w_src and "--force-step" in w_src:
        print("⚠️  [S152] force-step markers already present in watcher_agent.py — aborting.")
        return

    # ── Validate all anchors ──
    missing = []
    for label, anchor in [
        ("WatcherConfig halt_field", OLD_WATCHER_CONFIG_HALT),
        ("freshness block",          OLD_FRESHNESS_BLOCK),
        ("--params argparse block",  OLD_PARAMS_ARG),
        ("WatcherConfig build",      OLD_CONFIG_BUILD),
    ]:
        if anchor not in w_src:
            missing.append(f"watcher_agent.py: {label}")

    if OLD_RESUME_LAUNCH not in s_src:
        missing.append("sweep_run1.sh: resume launch block")

    if missing:
        print("❌ Anchors NOT found:")
        for m in missing:
            print(f"   {m}")
        print("Aborting — check for prior partial patch or code changes.")
        return

    # ── Apply patches ──
    w_patched = w_src
    w_patched = w_patched.replace(OLD_WATCHER_CONFIG_HALT, NEW_WATCHER_CONFIG_HALT, 1)
    w_patched = w_patched.replace(OLD_FRESHNESS_BLOCK,     NEW_FRESHNESS_BLOCK,     1)
    w_patched = w_patched.replace(OLD_PARAMS_ARG,          NEW_PARAMS_ARG,          1)
    w_patched = w_patched.replace(OLD_CONFIG_BUILD,        NEW_CONFIG_BUILD,        1)

    s_patched = s_src.replace(OLD_RESUME_LAUNCH, NEW_RESUME_LAUNCH, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  watcher_agent.py lines: {len(w_src.splitlines())} → {len(w_patched.splitlines())}")
        print(f"  [S152] in watcher:     {'[S152]' in w_patched}")
        print(f"  --force-step in watcher: {'--force-step' in w_patched}")
        print(f"  force_steps in config:   {'force_steps=set' in w_patched}")
        print(f"  FORCE-STEP bypass:       {'FORCE-STEP' in w_patched}")
        print(f"  sweep_run1.sh patched:   {'--force-step 1' in s_patched}")
        return

    shutil.copy2(WATCHER,  WATCHER_BACKUP)
    shutil.copy2(SWEEP_R1, SWEEP_BACKUP)
    print(f"✅ Backups: {WATCHER_BACKUP}, {SWEEP_BACKUP}")

    WATCHER.write_text(w_patched)
    SWEEP_R1.write_text(s_patched)

    print(f"✅ Patched: {WATCHER}")
    print(f"✅ Patched: {SWEEP_R1}")
    print()
    print("Verification:")
    print(f"  [S152] marker in watcher:   {'[S152]' in w_patched}")
    print(f"  --force-step arg present:   {'--force-step' in w_patched}")
    print(f"  force_steps in WatcherConfig: {'force_steps: set' in w_patched}")
    print(f"  FORCE-STEP bypass in run_step: {'FORCE-STEP' in w_patched}")
    print(f"  sweep_run1.sh --resume patched: {'--force-step 1' in s_patched}")
    print()
    print("Usage:")
    print("  # Single step forced:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 --force-step 1")
    print()
    print("  # Multiple steps forced:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 3 --force-step 1 --force-step 2")
    print()
    print("  # sweep_run1.sh --resume now auto-adds --force-step 1")
    print("  bash sweep_run1.sh --resume")


if __name__ == "__main__":
    apply()
