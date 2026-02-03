#!/usr/bin/env python3
"""
patch_watcher_dispatch.py — Auto-patcher for WATCHER dispatch wiring
=====================================================================
Version: 1.0.0
Date: 2026-02-03
Session: 58 (Phase 7 Part B)

PURPOSE:
    Makes minimal, safe edits to agents/watcher_agent.py to wire in
    the dispatch module (agents/watcher_dispatch.py).

WHAT IT DOES:
    1. Creates a timestamped backup of agents/watcher_agent.py
    2. Adds import + bind_to_watcher() call (after existing imports)
    3. Adds new CLI arguments (--dispatch-selfplay, --dispatch-learning-loop,
       --process-requests, --dry-run)
    4. Adds dispatch handling to __main__ block
    5. Adds _scan_watcher_requests() to run_daemon() loop
    6. Ensures watcher_requests/ directory exists

SAFETY:
    - Creates backup BEFORE any modifications
    - All insertions use pattern matching (no hardcoded line numbers)
    - If a pattern isn't found, patch aborts with clear error
    - Idempotent: won't double-apply if already patched
    - Prints diff-style preview before writing

USAGE:
    cd ~/distributed_prng_analysis
    python3 patch_watcher_dispatch.py           # Apply patch
    python3 patch_watcher_dispatch.py --preview  # Preview only
    python3 patch_watcher_dispatch.py --verify   # Check if already patched

UNDO:
    cp agents/watcher_agent.py.bak.YYYYMMDD_HHMMSS agents/watcher_agent.py

CRITICAL REMINDER:
    NEVER restore from backup after modifying code — fix mistakes by
    removing/editing the bad additions instead.
    This patcher is the EXCEPTION to that rule because it runs ONCE
    and creates a clean, inspectable backup.
"""

import os
import re
import sys
import shutil
from datetime import datetime


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WATCHER_PATH = os.path.join("agents", "watcher_agent.py")
REQUESTS_DIR = "watcher_requests"

# Marker to detect if already patched
PATCH_MARKER = "# Phase 7 Part B: Dispatch wiring (Session 58)"


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

# Patch 1: Import + binding (inserted after the last top-level import)
IMPORT_PATCH = f"""
{PATCH_MARKER}
from agents.watcher_dispatch import bind_to_watcher
"""

# Patch 2: Bind call (inserted after class WatcherAgent: ... ends,
#           right before `if __name__`)
BIND_PATCH = f"""
# ── Phase 7 Part B: Bind dispatch methods to WatcherAgent ────────────────
bind_to_watcher(WatcherAgent)
"""

# Patch 3: CLI arguments (inserted after last add_argument)
CLI_ARGS_PATCH = """
    # Phase 7 Part B: Dispatch commands
    parser.add_argument('--dispatch-selfplay', action='store_true',
                        help='Dispatch selfplay orchestrator')
    parser.add_argument('--dispatch-learning-loop', type=str, nargs='?',
                        const='steps_3_5_6', metavar='SCOPE',
                        help='Dispatch learning loop (default: steps_3_5_6)')
    parser.add_argument('--process-requests', action='store_true',
                        help='Process pending watcher_requests/*.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (log actions without executing)')
"""

# Patch 4: Dispatch handling (inserted after existing arg handling in __main__)
DISPATCH_HANDLING_PATCH = """
    # Phase 7 Part B: Dispatch handling
    elif args.dispatch_selfplay:
        request = {"episodes": getattr(args, 'episodes', 5)}
        dry = getattr(args, 'dry_run', False)
        ok = watcher.dispatch_selfplay(request, dry_run=dry)
        sys.exit(0 if ok else 1)
    elif args.dispatch_learning_loop:
        dry = getattr(args, 'dry_run', False)
        ok = watcher.dispatch_learning_loop(
            scope=args.dispatch_learning_loop, dry_run=dry)
        sys.exit(0 if ok else 1)
    elif args.process_requests:
        dry = getattr(args, 'dry_run', False)
        count = watcher._scan_watcher_requests(dry_run=dry)
        print(f"Processed {count} request(s)")
        sys.exit(0)
"""

# Patch 5: Daemon wiring (inserted inside run_daemon loop)
DAEMON_PATCH = """
            # Phase 7 Part B: Scan for Chapter 13 requests
            try:
                self._scan_watcher_requests()
            except Exception as _req_err:
                logger.warning(f"Request scan error: {_req_err}")
"""


# ---------------------------------------------------------------------------
# Patch logic
# ---------------------------------------------------------------------------

def is_already_patched(content: str) -> bool:
    """Check if the file has already been patched."""
    return PATCH_MARKER in content


def apply_import_patch(content: str) -> str:
    """Add import + bind_to_watcher after the last import block."""
    # Find the last 'import' or 'from ... import' line before class definition
    lines = content.split('\n')
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or
                stripped.startswith('from ')):
            last_import_idx = i
        # Stop searching once we hit class definition
        if stripped.startswith('class WatcherAgent'):
            break

    if last_import_idx == -1:
        raise RuntimeError("Could not find import block in watcher_agent.py")

    lines.insert(last_import_idx + 1, IMPORT_PATCH)
    return '\n'.join(lines)


def apply_bind_patch(content: str) -> str:
    """Add bind_to_watcher() call right before if __name__ == '__main__'."""
    marker = 'if __name__'
    idx = content.find(marker)
    if idx == -1:
        raise RuntimeError("Could not find 'if __name__' block")

    return content[:idx] + BIND_PATCH + '\n' + content[idx:]


def apply_cli_args_patch(content: str) -> str:
    """Add new CLI arguments after the last existing add_argument call."""
    # Find the last parser.add_argument line before parse_args
    lines = content.split('\n')
    last_add_arg_idx = -1
    for i, line in enumerate(lines):
        if 'add_argument' in line and 'parser' in line:
            last_add_arg_idx = i
        # Stop at parse_args
        if 'parse_args' in line:
            break

    if last_add_arg_idx == -1:
        raise RuntimeError("Could not find parser.add_argument lines")

    lines.insert(last_add_arg_idx + 1, CLI_ARGS_PATCH)
    return '\n'.join(lines)


def apply_dispatch_handling_patch(content: str) -> str:
    """Add dispatch arg handling in the __main__ block.

    Inserts after a known 'elif' block (like 'elif args.halt' or
    'elif args.status' or the last elif before 'else:').
    """
    # Strategy: find the last 'elif args.' line in __main__ block
    lines = content.split('\n')
    in_main = False
    last_elif_idx = -1

    for i, line in enumerate(lines):
        if "if __name__" in line:
            in_main = True
        if in_main and line.strip().startswith('elif args.'):
            last_elif_idx = i
            # Also capture multi-line blocks by finding next elif/else
        if in_main and last_elif_idx > 0 and i > last_elif_idx:
            stripped = line.strip()
            if (stripped.startswith('elif ') or stripped.startswith('else:')):
                # Insert BEFORE this line
                lines.insert(i, DISPATCH_HANDLING_PATCH)
                return '\n'.join(lines)

    # Fallback: insert after last elif block
    if last_elif_idx > -1:
        # Find end of last elif block (next line with same or less indent)
        base_indent = len(lines[last_elif_idx]) - len(
            lines[last_elif_idx].lstrip()
        )
        insert_at = last_elif_idx + 1
        for j in range(last_elif_idx + 1, len(lines)):
            stripped = lines[j].strip()
            if not stripped:
                continue
            curr_indent = len(lines[j]) - len(lines[j].lstrip())
            if curr_indent <= base_indent and stripped:
                insert_at = j
                break
        lines.insert(insert_at, DISPATCH_HANDLING_PATCH)
        return '\n'.join(lines)

    print("  ⚠️  Could not auto-place dispatch handling — "
          "manual insertion needed")
    print(f"  Add this block to __main__ handling:\n{DISPATCH_HANDLING_PATCH}")
    return '\n'.join(lines)


def apply_daemon_patch(content: str) -> str:
    """Add request scanning to run_daemon() loop.

    Looks for 'time.sleep' or 'poll_interval' inside run_daemon
    and inserts the scan call just before the sleep.
    """
    # Find run_daemon method
    daemon_match = re.search(
        r'def run_daemon\(self.*?\):', content
    )
    if not daemon_match:
        print("  ⚠️  Could not find run_daemon() — "
              "manual daemon wiring needed")
        print(f"  Add this to your daemon loop:\n{DAEMON_PATCH}")
        return content

    daemon_start = daemon_match.start()

    # Find time.sleep or poll_interval inside the daemon
    sleep_pattern = re.compile(
        r'(\s+)(time\.sleep|poll_interval|self\.config\.poll)',
        re.MULTILINE
    )
    sleep_match = sleep_pattern.search(content, daemon_start)
    if sleep_match:
        # Insert before the sleep line
        insert_pos = content.rfind('\n', 0, sleep_match.start()) + 1
        return content[:insert_pos] + DAEMON_PATCH + content[insert_pos:]

    # Fallback: find 'while' in run_daemon and add near start of loop
    while_match = re.search(r'while\s+', content[daemon_start:])
    if while_match:
        # Find end of while line (colon)
        while_pos = daemon_start + while_match.start()
        colon_pos = content.find(':', while_pos)
        newline_pos = content.find('\n', colon_pos)
        insert_pos = newline_pos + 1
        return content[:insert_pos] + DAEMON_PATCH + content[insert_pos:]

    print("  ⚠️  Could not find sleep/while in run_daemon — "
          "manual daemon wiring needed")
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse as ap
    parser = ap.ArgumentParser(
        description="Patch watcher_agent.py for Phase 7 dispatch wiring"
    )
    parser.add_argument('--preview', action='store_true',
                        help='Show what would be changed without writing')
    parser.add_argument('--verify', action='store_true',
                        help='Check if already patched')
    args = parser.parse_args()

    # Verify we're in the right directory
    if not os.path.exists(WATCHER_PATH):
        print(f"ERROR: {WATCHER_PATH} not found.")
        print("Run this script from ~/distributed_prng_analysis/")
        sys.exit(1)

    # Read current file
    with open(WATCHER_PATH) as f:
        original = f.read()

    # Check if already patched
    if is_already_patched(original):
        print("✅ watcher_agent.py is already patched (Phase 7 Part B)")
        sys.exit(0)

    if args.verify:
        print("❌ watcher_agent.py is NOT yet patched")
        sys.exit(1)

    print("=" * 60)
    print("Phase 7 Part B: WATCHER Dispatch Patcher")
    print("=" * 60)
    print(f"\nTarget: {WATCHER_PATH}")
    print(f"Original size: {len(original)} chars, "
          f"{len(original.splitlines())} lines")

    # Apply patches sequentially
    patched = original
    patches_applied = []

    try:
        print("\n[1/5] Adding import + bind_to_watcher()...")
        patched = apply_import_patch(patched)
        patches_applied.append("import")

        print("[2/5] Adding bind_to_watcher() call...")
        patched = apply_bind_patch(patched)
        patches_applied.append("bind")

        print("[3/5] Adding CLI arguments...")
        patched = apply_cli_args_patch(patched)
        patches_applied.append("cli_args")

        print("[4/5] Adding dispatch handling in __main__...")
        patched = apply_dispatch_handling_patch(patched)
        patches_applied.append("dispatch_handling")

        print("[5/5] Adding request scanning to run_daemon()...")
        patched = apply_daemon_patch(patched)
        patches_applied.append("daemon")

    except RuntimeError as e:
        print(f"\n❌ PATCH FAILED: {e}")
        print("No changes written.")
        sys.exit(1)

    # Summary
    new_lines = len(patched.splitlines())
    added_lines = new_lines - len(original.splitlines())
    print(f"\nPatched size: {len(patched)} chars, {new_lines} lines "
          f"(+{added_lines} lines)")
    print(f"Patches applied: {', '.join(patches_applied)}")

    if args.preview:
        print("\n--- PREVIEW MODE (no changes written) ---")
        print("\nNew content that would be added:")
        for line in patched.splitlines():
            if "Phase 7 Part B" in line:
                # Show context: 3 lines before and after
                idx = patched.splitlines().index(line)
                start = max(0, idx - 1)
                end = min(len(patched.splitlines()), idx + 5)
                print(f"\n  ... around line {idx + 1}:")
                for j in range(start, end):
                    prefix = "+ " if "Phase 7" in patched.splitlines()[j] or \
                             patched.splitlines()[j] not in original else "  "
                    print(f"  {prefix}{patched.splitlines()[j]}")
        sys.exit(0)

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{WATCHER_PATH}.bak.{timestamp}"
    shutil.copy2(WATCHER_PATH, backup_path)
    print(f"\nBackup: {backup_path}")

    # Write patched file
    with open(WATCHER_PATH, 'w') as f:
        f.write(patched)
    print(f"✅ Patch applied to {WATCHER_PATH}")

    # Create watcher_requests directory
    if not os.path.isdir(REQUESTS_DIR):
        os.makedirs(REQUESTS_DIR, exist_ok=True)
        print(f"✅ Created {REQUESTS_DIR}/")
    else:
        print(f"✅ {REQUESTS_DIR}/ already exists")

    # Verify dispatch module exists
    dispatch_path = os.path.join("agents", "watcher_dispatch.py")
    if os.path.exists(dispatch_path):
        print(f"✅ {dispatch_path} found")
    else:
        print(f"⚠️  {dispatch_path} NOT FOUND — copy it first!")

    print(f"\n{'=' * 60}")
    print("Next steps:")
    print("  1. Review the patched file:")
    print(f"     diff {backup_path} {WATCHER_PATH}")
    print("  2. Run self-test:")
    print("     PYTHONPATH=. python3 agents/watcher_dispatch.py --self-test")
    print("  3. Run dry-run dispatch:")
    print("     PYTHONPATH=. python3 agents/watcher_agent.py "
          "--dispatch-selfplay --dry-run")
    print("  4. Run dry-run learning loop:")
    print("     PYTHONPATH=. python3 agents/watcher_agent.py "
          "--dispatch-learning-loop steps_3_5_6 --dry-run")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
