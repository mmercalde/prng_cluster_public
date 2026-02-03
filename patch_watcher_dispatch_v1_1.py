#!/usr/bin/env python3
"""
patch_watcher_dispatch_v1_1.py — Fixed auto-patcher
=====================================================
Version: 1.1.0 (fixes v1.0 import placement bug)

v1.0 BUG: Import finder matched `from ... import` inside function bodies.
v1.1 FIX: Only matches TOP-LEVEL imports (zero indentation).
v1.1 FIX: Dispatch handling uses explicit marker insertion instead of
          fragile elif-chain detection.

Usage:
    cd ~/distributed_prng_analysis
    cp agents/watcher_agent.py.bak.20260202_175949 agents/watcher_agent.py  # restore first
    python3 patch_watcher_dispatch_v1_1.py
"""

import os
import re
import sys
import shutil
from datetime import datetime

WATCHER_PATH = os.path.join("agents", "watcher_agent.py")
PATCH_MARKER = "# Phase 7 Part B: Dispatch wiring (Session 58)"


def main():
    if not os.path.exists(WATCHER_PATH):
        print(f"ERROR: {WATCHER_PATH} not found. Run from project root.")
        sys.exit(1)

    with open(WATCHER_PATH) as f:
        lines = f.readlines()

    if any(PATCH_MARKER in line for line in lines):
        print("✅ Already patched — nothing to do.")
        sys.exit(0)

    original_count = len(lines)
    print(f"Original: {original_count} lines")

    # ──────────────────────────────────────────────────────────
    # PATCH 1: Import (after last TOP-LEVEL import/from line)
    # Key fix: only match lines with ZERO indentation
    # ──────────────────────────────────────────────────────────
    last_top_import = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Must start at column 0 (no leading whitespace)
        if line[0:1] not in (' ', '\t', '\n', '\r', '#'):
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_top_import = i

    if last_top_import == -1:
        print("❌ Could not find any top-level imports")
        sys.exit(1)

    import_block = [
        f"\n",
        f"{PATCH_MARKER}\n",
        f"from agents.watcher_dispatch import bind_to_watcher\n",
    ]
    insert_at = last_top_import + 1
    for j, new_line in enumerate(import_block):
        lines.insert(insert_at + j, new_line)
    print(f"[1/5] ✅ Import added after line {last_top_import + 1}")

    # ──────────────────────────────────────────────────────────
    # PATCH 2: bind_to_watcher() call before if __name__
    # ──────────────────────────────────────────────────────────
    main_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("if __name__"):
            main_idx = i
            break

    if main_idx is None:
        print("❌ Could not find if __name__ block")
        sys.exit(1)

    bind_block = [
        "\n",
        "# ── Phase 7 Part B: Bind dispatch methods to WatcherAgent ──\n",
        "bind_to_watcher(WatcherAgent)\n",
        "\n",
    ]
    for j, new_line in enumerate(bind_block):
        lines.insert(main_idx + j, new_line)
    print(f"[2/5] ✅ bind_to_watcher() added before __main__")

    # Update main_idx after insertion
    main_idx += len(bind_block)

    # ──────────────────────────────────────────────────────────
    # PATCH 3: CLI arguments (after last parser.add_argument)
    # ──────────────────────────────────────────────────────────
    last_add_arg = -1
    parse_args_line = -1
    for i in range(main_idx, len(lines)):
        if 'add_argument' in lines[i]:
            last_add_arg = i
        if 'parse_args' in lines[i] and parse_args_line == -1:
            parse_args_line = i

    if last_add_arg == -1:
        print("⚠️  Could not find parser.add_argument — skipping CLI args")
    else:
        cli_block = [
            "\n",
            "    # Phase 7 Part B: Dispatch commands\n",
            "    parser.add_argument('--dispatch-selfplay', action='store_true',\n",
            "                        help='Dispatch selfplay orchestrator')\n",
            "    parser.add_argument('--dispatch-learning-loop', type=str, nargs='?',\n",
            "                        const='steps_3_5_6', metavar='SCOPE',\n",
            "                        help='Dispatch learning loop (default: steps_3_5_6)')\n",
            "    parser.add_argument('--process-requests', action='store_true',\n",
            "                        help='Process pending watcher_requests/*.json')\n",
            "    parser.add_argument('--dry-run', action='store_true',\n",
            "                        help='Dry run (log actions without executing)')\n",
        ]
        insert_at = last_add_arg + 1
        # If the add_argument spans multiple lines (continuation), skip them
        while insert_at < len(lines) and lines[insert_at].strip().startswith("help="):
            insert_at += 1
        for j, new_line in enumerate(cli_block):
            lines.insert(insert_at + j, new_line)
        print(f"[3/5] ✅ CLI arguments added after line {last_add_arg + 1}")

    # ──────────────────────────────────────────────────────────
    # PATCH 4: Dispatch handling in main()
    # Strategy: find 'watcher = WatcherAgent' then find the
    # first 'else:' or end of the elif chain after it
    # ──────────────────────────────────────────────────────────
    dispatch_block = [
        "\n",
        "    # Phase 7 Part B: Dispatch handling\n",
        "    if hasattr(args, 'dispatch_selfplay') and args.dispatch_selfplay:\n",
        "        request = {'episodes': getattr(args, 'episodes', 5)}\n",
        "        dry = getattr(args, 'dry_run', False)\n",
        "        ok = watcher.dispatch_selfplay(request, dry_run=dry)\n",
        "        sys.exit(0 if ok else 1)\n",
        "    elif hasattr(args, 'dispatch_learning_loop') and args.dispatch_learning_loop:\n",
        "        dry = getattr(args, 'dry_run', False)\n",
        "        ok = watcher.dispatch_learning_loop(\n",
        "            scope=args.dispatch_learning_loop, dry_run=dry)\n",
        "        sys.exit(0 if ok else 1)\n",
        "    elif hasattr(args, 'process_requests') and args.process_requests:\n",
        "        dry = getattr(args, 'dry_run', False)\n",
        "        count = watcher._scan_watcher_requests(dry_run=dry)\n",
        "        print(f'Processed {count} request(s)')\n",
        "        sys.exit(0)\n",
    ]

    # Find the main() function definition
    main_func_idx = None
    for i in range(main_idx, len(lines)):
        if lines[i].strip().startswith('def main('):
            main_func_idx = i
            break

    # Find last elif in main() — look for 'elif args.' pattern with 4-space indent
    inserted_dispatch = False
    if main_func_idx:
        # Find 'else:' at 4-space indent level in main() function
        for i in range(main_func_idx + 1, len(lines)):
            stripped = lines[i].strip()
            # Find the standalone 'else:' that ends the arg processing chain
            if stripped == 'else:' and lines[i].startswith('    else:'):
                # Insert our dispatch block BEFORE this else:
                for j, new_line in enumerate(dispatch_block):
                    lines.insert(i + j, new_line)
                inserted_dispatch = True
                print(f"[4/5] ✅ Dispatch handling added before else: at line {i + 1}")
                break

    if not inserted_dispatch:
        # Fallback: find last line of main() and add before it
        # Or find 'sys.exit' pattern
        for i in range(len(lines) - 1, main_idx, -1):
            if 'sys.exit' in lines[i] or lines[i].strip() == '':
                for j, new_line in enumerate(dispatch_block):
                    lines.insert(i + j, new_line)
                inserted_dispatch = True
                print(f"[4/5] ✅ Dispatch handling added near end of main()")
                break

    if not inserted_dispatch:
        # Last resort: append at end of file
        lines.extend(["\n"] + dispatch_block)
        print("[4/5] ⚠️  Dispatch handling appended at end of file "
              "— verify placement manually")

    # ──────────────────────────────────────────────────────────
    # PATCH 5: Daemon wiring (request scanning in run_daemon)
    # ──────────────────────────────────────────────────────────
    daemon_patch = [
        "            # Phase 7 Part B: Scan for Chapter 13 requests\n",
        "            try:\n",
        "                self._scan_watcher_requests()\n",
        "            except Exception as _req_err:\n",
        "                logger.warning(f'Request scan error: {_req_err}')\n",
        "\n",
    ]

    # Find run_daemon method, then find time.sleep in it
    daemon_start = None
    for i, line in enumerate(lines):
        if 'def run_daemon' in line:
            daemon_start = i
            break

    daemon_patched = False
    if daemon_start:
        for i in range(daemon_start, min(daemon_start + 100, len(lines))):
            if 'time.sleep' in lines[i]:
                # Insert before the sleep line
                for j, new_line in enumerate(daemon_patch):
                    lines.insert(i + j, new_line)
                daemon_patched = True
                print(f"[5/5] ✅ Daemon wiring added before time.sleep at line {i + 1}")
                break

    if not daemon_patched:
        print("[5/5] ⚠️  Could not find time.sleep in run_daemon() — "
              "add request scanning manually")

    # ──────────────────────────────────────────────────────────
    # Write result
    # ──────────────────────────────────────────────────────────
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{WATCHER_PATH}.bak.{timestamp}"
    shutil.copy2(WATCHER_PATH, backup_path)
    print(f"\nBackup: {backup_path}")

    with open(WATCHER_PATH, 'w') as f:
        f.writelines(lines)

    new_count = len(lines)
    print(f"Patched: {new_count} lines (+{new_count - original_count})")
    print(f"✅ Patch applied to {WATCHER_PATH}")

    # Quick syntax check
    print("\nSyntax check... ", end="")
    import py_compile
    try:
        py_compile.compile(WATCHER_PATH, doraise=True)
        print("✅ OK")
    except py_compile.PyCompileError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        print(f"\nRestore with: cp {backup_path} {WATCHER_PATH}")
        sys.exit(1)

    print(f"\nNext:")
    print(f"  PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-selfplay --dry-run")
    print(f"  PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6 --dry-run")


if __name__ == "__main__":
    main()
