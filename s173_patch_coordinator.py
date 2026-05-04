#!/usr/bin/env python3
"""
S173 Patch: persistent_worker_coordinator.py — JSONL assignment ledger

Applies 3 anchor-based substitutions:
  1. Add module-level imports for json/time near existing imports
  2. Add ledger path init + helper method in __init__
  3. Hook ledger.append() inside dispatch_chunk() before _run_once() call

Each chunk dispatch writes one line to:
    logs/s173_job_assignment_ledger.jsonl

Usage:
    python3 s173_patch_coordinator.py persistent_worker_coordinator.py
"""

import argparse
import os
import sys


SUBSTITUTIONS = [
    # ---------------------------------------------------------------------
    # 1. __init__ — add ledger path + helper
    # ---------------------------------------------------------------------
    (
        "__init__ — add S173 ledger state",
        "        self.logger = logging.getLogger(\"PersistentWorkerCoordinator\")\n"
        "        if not self.logger.handlers:\n"
        "            h = logging.StreamHandler()\n"
        "            h.setFormatter(logging.Formatter(\"[PWC] %(levelname)s %(message)s\"))\n"
        "            self.logger.addHandler(h)\n"
        "        self.logger.setLevel(logging.INFO)",

        "        self.logger = logging.getLogger(\"PersistentWorkerCoordinator\")\n"
        "        if not self.logger.handlers:\n"
        "            h = logging.StreamHandler()\n"
        "            h.setFormatter(logging.Formatter(\"[PWC] %(levelname)s %(message)s\"))\n"
        "            self.logger.addHandler(h)\n"
        "        self.logger.setLevel(logging.INFO)\n"
        "\n"
        "        # [S173] Job assignment ledger — one JSONL line per dispatch\n"
        "        try:\n"
        "            os.makedirs(\"logs\", exist_ok=True)\n"
        "            self._s173_ledger_path = \"logs/s173_job_assignment_ledger.jsonl\"\n"
        "            self._s173_ledger_lock = threading.Lock()\n"
        "        except Exception as exc:\n"
        "            self.logger.warning(f\"[S173] ledger init failed: {exc}\")\n"
        "            self._s173_ledger_path = None\n"
        "            self._s173_ledger_lock = None",
    ),

    # ---------------------------------------------------------------------
    # 2. Insert _s173_log_assignment helper method right before _is_localhost
    # ---------------------------------------------------------------------
    (
        "Insert _s173_log_assignment helper",
        "    def _is_localhost(self, hostname: str) -> bool:\n"
        "        return hostname in (\"localhost\", \"127.0.0.1\", socket.gethostname())",

        "    def _s173_log_assignment(self, job: dict, worker_handle_or_node) -> None:\n"
        "        \"\"\"\n"
        "        [S173] Append one JSONL line to the assignment ledger.\n"
        "        Called from dispatch_chunk() before _run_once().\n"
        "        NEVER raises — instrumentation must not affect dispatch.\n"
        "        \"\"\"\n"
        "        if not self._s173_ledger_path:\n"
        "            return\n"
        "        try:\n"
        "            import json as _json\n"
        "            import time as _time\n"
        "            wh = worker_handle_or_node\n"
        "            host = None\n"
        "            gpu  = None\n"
        "            # [S173 — TB Blocker 3] worker_pid is best-effort:\n"
        "            #   SSH transport: WorkerHandle.proc.pid is the LOCAL ssh wrapper PID,\n"
        "            #     not the remote python PID. Logged for completeness but daemon\n"
        "            #     cannot use it to match netconsole \"in process python pid N\".\n"
        "            #   TCP transport: remote PID is in worker_active_job_state file,\n"
        "            #     not visible to coordinator at dispatch time. Logged as null.\n"
        "            #   In both cases, the daemon's authoritative PID source is\n"
        "            #   /tmp/prng_active_worker_gpu*.json on the rig.\n"
        "            worker_pid_local_ssh = None\n"
        "            worker_pid_source = \"unavailable\"\n"
        "            if wh is not None and hasattr(wh, \"node\"):\n"
        "                host = getattr(wh.node, \"hostname\", None)\n"
        "                gpu  = getattr(wh, \"gpu_id\", None)\n"
        "                _proc = getattr(wh, \"proc\", None)\n"
        "                if _proc is not None and hasattr(_proc, \"pid\"):\n"
        "                    worker_pid_local_ssh = _proc.pid\n"
        "                    worker_pid_source = \"ssh_local_wrapper\"\n"
        "                elif self._tcp_transport is not None:\n"
        "                    worker_pid_source = \"tcp_remote_unknown\"\n"
        "            elif wh is not None and hasattr(wh, \"hostname\"):\n"
        "                host = wh.hostname\n"
        "                gpu  = job.get(\"gpu_id\")\n"
        "            else:\n"
        "                host = \"localhost\"\n"
        "                gpu  = job.get(\"gpu_id\")\n"
        "\n"
        "            skip = job.get(\"skip_range\", [job.get(\"skip_min\"), job.get(\"skip_max\")])\n"
        "            skip_min = skip[0] if isinstance(skip, (list, tuple)) and len(skip) > 0 else None\n"
        "            skip_max = skip[1] if isinstance(skip, (list, tuple)) and len(skip) > 1 else None\n"
        "\n"
        "            entry = {\n"
        "                \"ts\":            _time.strftime(\"%Y-%m-%dT%H:%M:%S%z\", _time.localtime()),\n"
        "                \"monotonic_ns\":  _time.monotonic_ns(),\n"
        "                \"worker_host\":   host,\n"
        "                \"worker_gpu\":    gpu,\n"
        "                \"worker_pid\":              worker_pid_local_ssh,\n"
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

    # ---------------------------------------------------------------------
    # 3. dispatch_chunk — call _s173_log_assignment before _run_once
    # ---------------------------------------------------------------------
    (
        "dispatch_chunk — log assignment before _run_once",
        "            t0 = time.time()\n"
        "            res = _run_once(worker_handle_or_node)\n"
        "            elapsed = time.time() - t0",

        "            # [S173] Log assignment to ledger before dispatch\n"
        "            self._s173_log_assignment(job, worker_handle_or_node)\n"
        "\n"
        "            t0 = time.time()\n"
        "            res = _run_once(worker_handle_or_node)\n"
        "            elapsed = time.time() - t0",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply S173 coordinator patches")
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
    applied = []
    skipped = []

    for desc, anchor, replacement in SUBSTITUTIONS:
        if anchor not in text:
            if replacement in text:
                skipped.append((desc, "already applied"))
            else:
                skipped.append((desc, "ANCHOR NOT FOUND — file diverged"))
            continue
        if text.count(anchor) > 1:
            skipped.append((desc, f"AMBIGUOUS — anchor appears {text.count(anchor)}x"))
            continue
        text = text.replace(anchor, replacement, 1)
        applied.append(desc)

    print("=" * 72)
    print("S173 coordinator patch")
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
