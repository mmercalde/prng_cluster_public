#!/usr/bin/env python3
"""
S173 Patch: pwc_worker_service.py instrumentation hooks

Applies 4 anchor-based substitutions:
  1. Add import for active_job_state at top of file
  2. Add ActiveJobState init in __init__
  3. Call write_gpu_bus_map + ActiveJobState init in _setup_env
  4. Wrap _execute_job kernel call with on_chunk_start/on_chunk_end

Run from the project root (where persistent/ is):
    python3 s173_patch_worker_service.py persistent/pwc_worker_service.py

Idempotent: re-running on an already-patched file is a no-op (anchors are
only present in the unpatched form).
"""

import argparse
import os
import sys


# Each substitution is (description, anchor_text, replacement_text).
# Anchors are uniquely identifiable strings from the unpatched file.
SUBSTITUTIONS = [
    # ---------------------------------------------------------------------
    # 1. Module-level import (after existing imports near top of file)
    # ---------------------------------------------------------------------
    (
        "Module imports — add active_job_state",
        "from typing import Any, Dict, Optional",
        """from typing import Any, Dict, Optional

# [S173] Fault attribution instrumentation
from persistent.active_job_state import ActiveJobState, write_gpu_bus_map""",
    ),

    # ---------------------------------------------------------------------
    # 2. Init ActiveJobState slot in __init__
    # ---------------------------------------------------------------------
    (
        "__init__ — add active_job_state slot",
        "        # Derived file paths (set in _setup_env after worker_id is known)\n"
        "        self._hb_json_path:  Optional[str] = None\n"
        "        self._hb_jsonl_path: Optional[str] = None",
        "        # Derived file paths (set in _setup_env after worker_id is known)\n"
        "        self._hb_json_path:  Optional[str] = None\n"
        "        self._hb_jsonl_path: Optional[str] = None\n"
        "\n"
        "        # [S173] Active-job state (set in _setup_env after gpu_id known)\n"
        "        self._active_job_state: Optional[ActiveJobState] = None",
    ),

    # ---------------------------------------------------------------------
    # 3. _setup_env — write gpu_bus_map and init ActiveJobState
    # ---------------------------------------------------------------------
    (
        "_setup_env — initialize S173 instrumentation",
        '        self._hb_json_path  = os.path.join(HEARTBEAT_DIR, f"{safe_name}.json")\n'
        '        self._hb_jsonl_path = os.path.join(HEARTBEAT_DIR, f"{safe_name}.events.jsonl")',
        '        self._hb_json_path  = os.path.join(HEARTBEAT_DIR, f"{safe_name}.json")\n'
        '        self._hb_jsonl_path = os.path.join(HEARTBEAT_DIR, f"{safe_name}.events.jsonl")\n'
        "\n"
        "        # [S173] Fault attribution instrumentation\n"
        "        try:\n"
        "            write_gpu_bus_map(gpu_id=self.gpu_id, worker_pid=os.getpid())\n"
        "            self._active_job_state = ActiveJobState(\n"
        "                worker_id=self.worker_id,\n"
        "                gpu_id=self.gpu_id,\n"
        "                host=socket.gethostname(),\n"
        "            )\n"
        "            log.info(\n"
        '                f"[{self.worker_id}] [S173] active-job instrumentation enabled "\n'
        '                f"gpu_id={self.gpu_id} pid={os.getpid()}"\n'
        "            )\n"
        "        except Exception as exc:\n"
        '            log.warning(f"[{self.worker_id}] [S173] instrumentation init failed: {exc}")\n'
        "            self._active_job_state = None",
    ),

    # ---------------------------------------------------------------------
    # 4. _execute_job — wrap kernel call
    # ---------------------------------------------------------------------
    (
        "_execute_job — wrap kernel call with on_chunk_start/on_chunk_end",
        '            self._emit_heartbeat("pre_kernel", job=job)\n'
        "\n"
        "            result = self._execute_sieve_job(sieve_job, 0)\n"
        "\n"
        '            self._emit_heartbeat("post_kernel", job=job)',
        '            self._emit_heartbeat("pre_kernel", job=job)\n'
        "\n"
        "            # [S173] write active-job JSON before kernel\n"
        "            if self._active_job_state is not None:\n"
        "                try:\n"
        "                    self._active_job_state.on_chunk_start(job)\n"
        "                except Exception:\n"
        "                    pass  # instrumentation never breaks worker\n"
        "\n"
        "            _kernel_t0 = time.time()\n"
        "            result = self._execute_sieve_job(sieve_job, 0)\n"
        "            _kernel_elapsed_ms = int((time.time() - _kernel_t0) * 1000)\n"
        "\n"
        '            self._emit_heartbeat("post_kernel", job=job)\n'
        "\n"
        "            # [S173] update active-job rolling stats after kernel\n"
        "            if self._active_job_state is not None:\n"
        "                try:\n"
        "                    self._active_job_state.on_chunk_end(\n"
        "                        job=job,\n"
        "                        elapsed_ms=_kernel_elapsed_ms,\n"
        "                        success=bool(result.get(\"success\", False)),\n"
        "                    )\n"
        "                except Exception:\n"
        "                    pass  # instrumentation never breaks worker",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply S173 instrumentation patches")
    ap.add_argument("file", help="Path to pwc_worker_service.py")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change but do not write")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip writing .s173_bak backup")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 1

    with open(args.file, "r") as f:
        original = f.read()

    text = original
    applied = []
    skipped = []

    for desc, anchor, replacement in SUBSTITUTIONS:
        if anchor not in text:
            # Already patched? Check if the replacement is present.
            if replacement in text:
                skipped.append((desc, "already applied"))
            else:
                # Anchor missing AND replacement absent — file diverged
                skipped.append((desc, "ANCHOR NOT FOUND — file diverged from expected"))
            continue

        # Count occurrences — must be unique
        if text.count(anchor) > 1:
            skipped.append((desc, f"AMBIGUOUS — anchor appears {text.count(anchor)}x"))
            continue

        text = text.replace(anchor, replacement, 1)
        applied.append(desc)

    # Report
    print("=" * 72)
    print("S173 worker service patch")
    print("=" * 72)
    for d in applied:
        print(f"  ✅ {d}")
    for d, reason in skipped:
        marker = "⚠️ " if "diverged" in reason or "AMBIGUOUS" in reason else "ℹ️ "
        print(f"  {marker} {d}  ({reason})")

    if any("diverged" in r or "AMBIGUOUS" in r for _, r in skipped):
        print("\nABORTING — fix anchor mismatches before applying.")
        return 2

    if not applied:
        print("\nNothing to do.")
        return 0

    if args.dry_run:
        print("\n[dry-run] no changes written")
        return 0

    if not args.no_backup:
        backup_path = args.file + ".s173_bak"
        with open(backup_path, "w") as f:
            f.write(original)
        print(f"\nBackup written: {backup_path}")

    with open(args.file, "w") as f:
        f.write(text)
    print(f"Patched: {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
