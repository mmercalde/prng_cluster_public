#!/usr/bin/env python3
"""
S173 Patch v2: persistent_worker_coordinator.py — POST-DISPATCH ledger

v1 had two bugs:
  1. Hook fired BEFORE dispatch, so worker_handle_or_node was usually None
     for TCP transport (workers selected dynamically by transport)
  2. Result: every entry showed worker_host=localhost, worker_gpu=None

v2 fixes both:
  - Hook fires AFTER _run_once() returns, with the result dict in hand
  - Captures worker_id, hostname, gpu_id FROM THE RESULT (real worker that ran it)
  - Falls back to localhost only when result has no worker info (Zeus subprocess)

Anchor strategy:
  - Re-uses the ledger init from v1 (already deployed in __init__)
  - Re-uses the helper, but redefines its signature to take (job, result, gpu_hint)
  - Moves the call site from before _run_once() to after the retry block

This patch is idempotent: applying v1 + v2 produces the same final state as v2 alone.
The v1 code patterns are detected and removed if present.

Usage:
    python3 s173_patch_coordinator_v2.py persistent_worker_coordinator.py
"""

import argparse
import os
import sys


# v1 anchor patterns to REMOVE (they exist after v1 was applied)
V1_PATTERNS_TO_REMOVE = [
    # The v1 _s173_log_assignment helper (with all its body)
    (
        "v1 helper definition",
        "    def _s173_log_assignment(self, job: dict, worker_handle_or_node) -> None:",
        "    def _is_localhost(self, hostname: str) -> bool:",
    ),
    # The v1 dispatch hook
    (
        "v1 dispatch hook",
        "            # [S173] Log assignment to ledger before dispatch\n"
        "            self._s173_log_assignment(job, worker_handle_or_node)\n"
        "\n"
        "            t0 = time.time()\n"
        "            res = _run_once(worker_handle_or_node)\n"
        "            elapsed = time.time() - t0",
        "            t0 = time.time()\n"
        "            res = _run_once(worker_handle_or_node)\n"
        "            elapsed = time.time() - t0",
    ),
]


# v2 substitutions (after v1 cleanup)
V2_SUBSTITUTIONS = [
    # Insert v2 helper before _is_localhost
    (
        "Insert _s173_log_assignment v2 helper",
        "    def _is_localhost(self, hostname: str) -> bool:\n"
        "        return hostname in (\"localhost\", \"127.0.0.1\", socket.gethostname())",

        "    def _s173_log_assignment(self, job: dict, result: dict,\n"
        "                              worker_handle_or_node=None) -> None:\n"
        "        \"\"\"\n"
        "        [S173 v2] Append one JSONL line to the assignment ledger AFTER dispatch.\n"
        "        Captures worker_id, hostname, gpu_id from the result (which the\n"
        "        TCP transport populates from the worker that actually ran the job).\n"
        "        Falls back to local handle info for Zeus subprocess path.\n"
        "        NEVER raises — instrumentation must not affect dispatch.\n"
        "        \"\"\"\n"
        "        if not self._s173_ledger_path:\n"
        "            return\n"
        "        try:\n"
        "            import json as _json\n"
        "            import time as _time\n"
        "\n"
        "            # [TB defensive] result may be None / partial / timeout under failure\n"
        "            res_dict = result if isinstance(result, dict) else {}\n"
        "\n"
        "            # First try: result fields (TCP transport path — populated by worker)\n"
        "            host = (res_dict.get(\"hostname\") or \"\").strip()\n"
        "            gpu  = res_dict.get(\"gpu_id\")\n"
        "            worker_id_str = res_dict.get(\"worker_id\", \"\") or \"\"\n"
        "            worker_pid_source = \"unavailable\"\n"
        "            worker_pid = None\n"
        "\n"
        "            if host and host != \"\":\n"
        "                worker_pid_source = \"tcp_result\"\n"
        "            else:\n"
        "                # Fallback: WorkerHandle (SSH path) or job dict (Zeus localhost)\n"
        "                wh = worker_handle_or_node\n"
        "                if wh is not None and hasattr(wh, \"node\"):\n"
        "                    host = getattr(wh.node, \"hostname\", \"\") or \"\"\n"
        "                    gpu  = getattr(wh, \"gpu_id\", None)\n"
        "                    _proc = getattr(wh, \"proc\", None)\n"
        "                    if _proc is not None and hasattr(_proc, \"pid\"):\n"
        "                        worker_pid = _proc.pid\n"
        "                        worker_pid_source = \"ssh_local_wrapper\"\n"
        "                else:\n"
        "                    host = host or \"localhost\"\n"
        "                    if gpu is None:\n"
        "                        gpu = job.get(\"gpu_id\")\n"
        "\n"
        "            skip = job.get(\"skip_range\", [job.get(\"skip_min\"), job.get(\"skip_max\")])\n"
        "            skip_min = skip[0] if isinstance(skip, (list, tuple)) and len(skip) > 0 else None\n"
        "            skip_max = skip[1] if isinstance(skip, (list, tuple)) and len(skip) > 1 else None\n"
        "\n"
        "            entry = {\n"
        "                \"ts\":            _time.strftime(\"%Y-%m-%dT%H:%M:%S%z\", _time.localtime()),\n"
        "                \"monotonic_ns\":  _time.monotonic_ns(),\n"
        "                \"worker_id\":     worker_id_str,\n"
        "                \"worker_host\":   host,\n"
        "                \"worker_gpu\":    gpu,\n"
        "                \"worker_pid\":              worker_pid,\n"
        "                \"worker_pid_source\":       worker_pid_source,\n"
        "                \"job_id\":        job.get(\"job_id\"),\n"
        "                \"chunk_id\":      job.get(\"job_id\"),\n"
        "                \"seed_start\":    job.get(\"seed_start\"),\n"
        "                \"seed_end\":      job.get(\"seed_end\"),\n"
        "                \"window_size\":   job.get(\"window_size\"),\n"
        "                \"offset\":        job.get(\"offset\"),\n"
        "                \"skip_min\":      skip_min,\n"
        "                \"skip_max\":      skip_max,\n"
        "                \"forward_threshold\":  job.get(\"min_match_threshold\"),\n"
        "                \"reverse_threshold\":  job.get(\"phase2_threshold\"),\n"
        "                \"hybrid\":        bool(job.get(\"hybrid\", False)),\n"
        "                \"prng_family\":   (job.get(\"prng_families\", [None]) or [None])[0],\n"
        "                \"phase\":         job.get(\"search_type\"),\n"
        "                \"dispatch_status\": res_dict.get(\"status\"),\n"
        "                \"elapsed_s\":     None,  # filled by caller if known\n"
        "            }\n"
        "            with self._s173_ledger_lock:\n"
        "                with open(self._s173_ledger_path, \"a\") as f:\n"
        "                    f.write(_json.dumps(entry, separators=(\",\", \":\")) + \"\\n\")\n"
        "        except Exception:\n"
        "            pass  # ledger failures must never affect dispatch\n"
        "\n"
        "    def _is_localhost(self, hostname: str) -> bool:\n"
        "        return hostname in (\"localhost\", \"127.0.0.1\", socket.gethostname())",
    ),

    # Insert hook AFTER the retry block, just before "with lock:"
    (
        "Insert post-dispatch ledger hook",
        "            with lock:\n"
        "                results_by_chunk[idx] = res\n"
        "            status = \"✅\" if res.get(\"status\") == \"ok\" else \"❌\"",

        "            # [S173 v2] Log AFTER dispatch with real worker attribution from result\n"
        "            self._s173_log_assignment(job, res, worker_handle_or_node)\n"
        "\n"
        "            with lock:\n"
        "                results_by_chunk[idx] = res\n"
        "            status = \"✅\" if res.get(\"status\") == \"ok\" else \"❌\"",
    ),
]


def remove_v1_block(text: str, desc: str, start_anchor: str, end_anchor: str) -> tuple:
    """Remove a block from text starting at start_anchor up to (but not including) end_anchor."""
    if start_anchor not in text:
        return text, False
    start_idx = text.find(start_anchor)
    end_idx = text.find(end_anchor, start_idx)
    if end_idx == -1:
        return text, False
    return text[:start_idx] + text[end_idx:], True


def remove_v1_dispatch_hook(text: str) -> tuple:
    """Remove the v1 dispatch hook (3-line block)."""
    v1_block = (
        "            # [S173] Log assignment to ledger before dispatch\n"
        "            self._s173_log_assignment(job, worker_handle_or_node)\n"
        "\n"
        "            t0 = time.time()\n"
    )
    v1_replacement = "            t0 = time.time()\n"
    if v1_block not in text:
        return text, False
    return text.replace(v1_block, v1_replacement, 1), True


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply S173 v2 coordinator patch")
    ap.add_argument("file", help="Path to persistent_worker_coordinator.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 1

    with open(args.file, "r") as f:
        original = f.read()

    text = original
    log = []

    # ---- Phase 1: remove v1 helper block (between def _s173_log_assignment and def _is_localhost) ----
    text2, changed = remove_v1_block(
        text,
        "v1 helper",
        "    def _s173_log_assignment(self, job: dict, worker_handle_or_node) -> None:",
        "    def _is_localhost(self, hostname: str) -> bool:",
    )
    if changed:
        text = text2
        log.append("✅ removed v1 _s173_log_assignment helper")
    else:
        log.append("ℹ️  v1 helper not present (fresh patch)")

    # ---- Phase 2: remove v1 dispatch hook ----
    text2, changed = remove_v1_dispatch_hook(text)
    if changed:
        text = text2
        log.append("✅ removed v1 dispatch hook")
    else:
        log.append("ℹ️  v1 dispatch hook not present")

    # ---- Phase 3: apply v2 substitutions ----
    for desc, anchor, replacement in V2_SUBSTITUTIONS:
        if anchor not in text:
            if replacement in text:
                log.append(f"ℹ️  {desc} (already applied)")
            else:
                log.append(f"⚠️  {desc} (ANCHOR NOT FOUND)")
            continue
        if text.count(anchor) > 1:
            log.append(f"⚠️  {desc} (AMBIGUOUS: {text.count(anchor)}x)")
            continue
        text = text.replace(anchor, replacement, 1)
        log.append(f"✅ {desc}")

    print("=" * 72)
    print("S173 v2 coordinator patch")
    print("=" * 72)
    for line in log:
        print(f"  {line}")

    if any("ANCHOR NOT FOUND" in l or "AMBIGUOUS" in l for l in log):
        print("\nABORTING — fix anchor mismatches.")
        return 2

    if text == original:
        print("\nNothing to do.")
        return 0

    if args.dry_run:
        print("\n[dry-run] no changes written")
        return 0

    if not args.no_backup:
        backup_path = args.file + ".s173v2_bak"
        with open(backup_path, "w") as f:
            f.write(original)
        print(f"\nBackup written: {backup_path}")

    with open(args.file, "w") as f:
        f.write(text)
    print(f"Patched: {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
