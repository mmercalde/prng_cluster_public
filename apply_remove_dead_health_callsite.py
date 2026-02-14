#!/usr/bin/env python3
"""
Remove dead Phase-6 training health check callsite in WatcherAgent._handle_proceed().

v1.1 -- Hardened per Team Beta review:
  - Dynamic indentation extraction (refactor-proof)
  - Scope guard: confirms Phase 6 block is inside _handle_proceed()

Why:
- Eliminates double-call to check_training_health()
- Removes misleading "param-threading not yet implemented" log spam
- Keeps training-health retry logic exclusively in the real retry-threaded path

Safe:
- Creates a timestamped backup
- Idempotent apply/revert via BEGIN/END markers
- Syntax checks watcher_agent.py after modification

Session 82 -- Team Alpha + Team Beta approved.
"""

import argparse
import datetime as _dt
import os
import re
import shutil
import sys
import py_compile

TARGET = os.path.join("agents", "watcher_agent.py") if os.path.isdir("agents") else "watcher_agent.py"

BEGIN = "# >>> S82_REMOVE_DEAD_HEALTH_CALLSITE_BEGIN <<<"
END   = "# >>> S82_REMOVE_DEAD_HEALTH_CALLSITE_END <<<"

PHASE6_START = r"^\s*#\s*CHAPTER 14 PHASE 6: Post-Step-5 Training Health Check\s*$"
PIPELINE_COMPLETE = r"^\s*#\s*Check if pipeline complete\s*$"

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def backup_path() -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{TARGET}.pre_s82_dead_health_callsite_{ts}"

def apply() -> None:
    if not os.path.isfile(TARGET):
        print(f"ERROR: {TARGET} not found. Run from project root.")
        sys.exit(1)

    content = _read(TARGET)

    if BEGIN in content:
        print("ALREADY APPLIED -- markers found. Use --revert first.")
        return

    # Locate the Phase 6 banner
    start_m = re.search(PHASE6_START, content, flags=re.MULTILINE)
    if not start_m:
        print("ERROR: Could not find Phase 6 banner in file.")
        sys.exit(1)

    # --- TB Issue 2: Scope guard -- confirm we're inside _handle_proceed() ---
    preceding = content[:start_m.start()]
    last_def_match = None
    for m in re.finditer(r"^\s*def\s+(\w+)\s*\(", preceding, flags=re.MULTILINE):
        last_def_match = m
    if last_def_match is None or last_def_match.group(1) != "_handle_proceed":
        found_fn = last_def_match.group(1) if last_def_match else "NONE"
        print(f"ERROR: Phase 6 block is not inside _handle_proceed().")
        print(f"  Nearest enclosing function: {found_fn}")
        print(f"  This is unexpected -- aborting for safety.")
        sys.exit(1)
    print(f"Scope guard: Phase 6 block confirmed inside _handle_proceed() -- OK")

    # Find the end anchor: "# Check if pipeline complete"
    end_m = re.search(PIPELINE_COMPLETE, content[start_m.start():], flags=re.MULTILINE)
    if not end_m:
        print("ERROR: Could not find '# Check if pipeline complete' after Phase 6 banner.")
        sys.exit(1)

    cut_start = start_m.start()
    cut_end = start_m.start() + end_m.start()

    # --- TB Issue 1: Dynamic indentation extraction ---
    banner_line_start = content.rfind('\n', 0, start_m.start()) + 1
    banner_line = content[banner_line_start:content.find('\n', start_m.start())]
    indent = re.match(r"^(\s*)", banner_line).group(1)
    print(f"Detected indentation: {len(indent)} spaces")

    # Include the decoration line above the banner (the line with ════)
    line_above_start = content.rfind('\n', 0, cut_start - 1) + 1
    line_above = content[line_above_start:cut_start].rstrip('\n')
    if re.match(r'^\s*#\s*[^\w]', line_above):
        cut_start = line_above_start

    # Backup
    bkp = backup_path()
    shutil.copy2(TARGET, bkp)
    print(f"Backup created: {bkp}")

    # Build replacement with dynamic indentation
    replacement = (
        f"{indent}{BEGIN}\n"
        f"{indent}# Removed dead Phase-6 training health callsite (S82 cleanup)\n"
        f"{indent}# Real health check logic lives in run_pipeline() S76 retry loop\n"
        f"{indent}# Team Alpha + Team Beta approved. See SESSION_CHANGELOG_20260213_S82.md\n"
        f"{indent}{END}\n\n"
    )

    patched = content[:cut_start] + replacement + content[cut_end:]

    _write(TARGET, patched)

    # Syntax check
    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"SYNTAX CHECK FAILED after apply: {e}")
        print("Restoring from backup...")
        shutil.copy2(bkp, TARGET)
        sys.exit(1)

    print("Syntax check: PASSED")

    # Verify the misleading log is gone
    final = _read(TARGET)
    if "param-threading not yet implemented" in final:
        print("WARNING: 'param-threading not yet implemented' still present!")
    else:
        print("Verified: 'param-threading not yet implemented' removed")

    # Verify real S76 loop untouched
    if "_handle_training_health" in final and "_build_retry_params" in final:
        print("Verified: S76 retry methods preserved")
    else:
        print("WARNING: S76 retry methods may be missing!")

    print()
    print("APPLIED SUCCESSFULLY")
    print()
    print("Next steps:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --status")

def revert() -> None:
    if not os.path.isfile(TARGET):
        print(f"ERROR: {TARGET} not found.")
        sys.exit(1)

    content = _read(TARGET)

    if BEGIN not in content:
        print("No markers found -- nothing to revert.")
        return

    target_dir = os.path.dirname(TARGET) or "."
    target_base = os.path.basename(TARGET)
    backups = sorted([p for p in os.listdir(target_dir)
                      if target_base in p and "pre_s82_dead_health_callsite_" in p])
    if not backups:
        print("ERROR: Markers found but no backup file discovered. Revert aborted.")
        sys.exit(1)

    bkp = os.path.join(target_dir, backups[-1])
    shutil.copy2(bkp, TARGET)

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"SYNTAX CHECK FAILED after revert: {e}")
        sys.exit(1)

    print(f"Reverted using backup: {bkp}")

def status() -> None:
    if not os.path.isfile(TARGET):
        print(f"{TARGET} not found.")
        return
    content = _read(TARGET)
    if BEGIN in content:
        print("S82 cleanup patch is ACTIVE (dead callsite removed).")
    else:
        print("S82 cleanup patch is NOT active.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S82: Remove dead Phase-6 health check callsite (v1.1 hardened)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="Apply the patch")
    g.add_argument("--revert", action="store_true", help="Revert from backup")
    g.add_argument("--status", action="store_true", help="Check patch status")
    args = ap.parse_args()

    if args.apply:
        apply()
    elif args.revert:
        revert()
    else:
        status()
