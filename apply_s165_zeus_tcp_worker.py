#!/usr/bin/env python3
"""
apply_s165_zeus_tcp_worker.py
==============================
Adds Zeus local persistent TCP workers to _tcp_launch_workers().

What this does:
  - Launches pwc_worker_service.py locally on Zeus (one per GPU, CUDA mode)
  - Workers connect to 127.0.0.1:5600 — same TCP path as AMD rigs
  - Zeus chunks dispatched via _dispatch_to_tcp() instead of subprocess
  - Expected improvement: ~550ms/chunk → ~25ms/chunk on Zeus 3080Ti GPUs
  - Effective Zeus throughput: ~180 seeds/sec → ~4.4M seeds/sec

Changes to persistent_worker_coordinator.py:
  1. Kill stale Zeus local workers at TCP startup
  2. Launch 2 persistent CUDA workers on Zeus before _tcp_wait_online()

Zero changes to pwc_worker_service.py — it already supports PWC_USE_ROCM=0.

Usage:
  python3 apply_s165_zeus_tcp_worker.py [--dry-run]

Backup created at: persistent_worker_coordinator.py.bak_s165_zeus_tcp
"""

import sys
import os
import ast
import shutil
import argparse

TARGET = "persistent_worker_coordinator.py"
BACKUP = TARGET + ".bak_s165_zeus_tcp"
MARKER = "[S165-ZEUS-TCP]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        sys.exit(1)

    with open(TARGET, "r") as f:
        content = f.read()

    # ── Guard: already patched? ────────────────────────────────────────────────
    if MARKER in content:
        print(f"Already patched ({MARKER} found). Nothing to do.")
        sys.exit(0)

    # ── Patch 1: Kill stale Zeus workers at start of _tcp_launch_workers ───────
    # Insert after the AMD stale-kill block's logger line, before the per-rig loop
    ANCHOR_KILL = (
        '        total_launched = 0\n'
    )
    REPLACEMENT_KILL = (
        '        total_launched = 0\n'
        '\n'
        '        # [S165-ZEUS-TCP] Kill stale Zeus local pwc workers before launching fresh ones\n'
        '        try:\n'
        '            import subprocess as _sp_zeus_kill\n'
        '            _sp_zeus_kill.run(\n'
        '                ["pkill", "-9", "-f", "pwc_worker_service"],\n'
        '                capture_output=True, timeout=5\n'
        '            )\n'
        '            self.logger.info("[PWC-TCP] Zeus: stale local workers killed")\n'
        '        except Exception:\n'
        '            pass  # No stale workers is fine\n'
    )

    if ANCHOR_KILL not in content:
        print("ERROR: Anchor 1 (total_launched = 0) not found — file may have changed.")
        sys.exit(1)

    content = content.replace(ANCHOR_KILL, REPLACEMENT_KILL, 1)
    print("Patch 1 applied: Zeus stale worker kill at startup")

    # ── Patch 2: Launch Zeus local workers before _tcp_wait_online ────────────
    ANCHOR_LAUNCH = (
        '        self.logger.info(\n'
        '            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"\n'
        '            " — waiting for online, then init, then ready"\n'
        '        )\n'
    )

    REPLACEMENT_LAUNCH = (
        '        # [S165-ZEUS-TCP] Launch persistent CUDA workers on Zeus (localhost)\n'
        '        # Workers connect back to 127.0.0.1:port — same TCP path as AMD rigs.\n'
        '        # PWC_USE_ROCM=0 → CUDA_VISIBLE_DEVICES path in pwc_worker_service._setup_env()\n'
        '        import subprocess as _sp_zeus\n'
        '        for _zeus_node in self.nodes:\n'
        '            if not self._is_localhost(_zeus_node.hostname):\n'
        '                continue\n'
        '            _zeus_pool = min(2, _zeus_node.gpu_count)  # 2 RTX 3080Ti\n'
        '            for _zeus_gpu in range(_zeus_pool):\n'
        '                _zeus_worker_id = f"zeus_gpu{_zeus_gpu}"\n'
        '                _zeus_log = f"/tmp/pwc_tcp_worker_zeus_gpu{_zeus_gpu}.log"\n'
        '                _zeus_env = os.environ.copy()\n'
        '                _zeus_env["PWC_GPU_ID"]           = str(_zeus_gpu)\n'
        '                _zeus_env["PWC_WORKER_ID"]        = _zeus_worker_id\n'
        '                _zeus_env["PWC_HOST"]             = "127.0.0.1"\n'
        '                _zeus_env["PWC_PORT"]             = str(self.pwc_port)\n'
        '                _zeus_env["PWC_USE_ROCM"]         = "0"\n'
        '                _zeus_env["CUDA_VISIBLE_DEVICES"] = str(_zeus_gpu)\n'
        '                _zeus_env["CUPY_CACHE_DIR"]       = f"/tmp/cupy_cache_zeus_gpu{_zeus_gpu}"\n'
        '                _zeus_env["PYTHONPATH"]           = _zeus_node.script_path\n'
        '                try:\n'
        '                    with open(_zeus_log, "a") as _lf:\n'
        '                        _zeus_proc = _sp_zeus.Popen(\n'
        '                            [_zeus_node.python_env, "-m", "persistent.pwc_worker_service"],\n'
        '                            env=_zeus_env,\n'
        '                            cwd=_zeus_node.script_path,\n'
        '                            stdout=_lf, stderr=_lf,\n'
        '                            start_new_session=True,\n'
        '                        )\n'
        '                    self.logger.info(\n'
        '                        f"[PWC-TCP] Zeus GPU{_zeus_gpu} local worker launched "\n'
        '                        f"— PID={_zeus_proc.pid} log={_zeus_log}"\n'
        '                    )\n'
        '                    total_launched += 1\n'
        '                except Exception as _ze:\n'
        '                    self.logger.error(\n'
        '                        f"[PWC-TCP] Zeus GPU{_zeus_gpu} local worker launch failed: {_ze}"\n'
        '                    )\n'
        '\n'
        '        self.logger.info(\n'
        '            "[PWC-TCP] " + str(total_launched) + " workers launched across all rigs"\n'
        '            " — waiting for online, then init, then ready"\n'
        '        )\n'
    )

    if ANCHOR_LAUNCH not in content:
        print("ERROR: Anchor 2 (workers launched across all rigs) not found.")
        sys.exit(1)

    content = content.replace(ANCHOR_LAUNCH, REPLACEMENT_LAUNCH, 1)
    print("Patch 2 applied: Zeus local persistent worker launch")

    # ── AST check ─────────────────────────────────────────────────────────────
    try:
        ast.parse(content)
        print("AST check: PASS")
    except SyntaxError as e:
        print(f"AST check: FAIL — {e}")
        sys.exit(1)

    if args.dry_run:
        print("\nDRY RUN — no files written.")
        print(f"Marker: {MARKER}")
        print("Both patches validated. Run without --dry-run to apply.")
        return

    # ── Write ──────────────────────────────────────────────────────────────────
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    with open(TARGET, "w") as f:
        f.write(content)

    print(f"\nPatched: {TARGET}")
    print(f"Marker:  {MARKER}")
    print()
    print("Verify with:")
    print(f"  grep -n 'S165-ZEUS-TCP' {TARGET}")
    print()
    print("Smoke test:")
    print("  python3 -c \"import persistent_worker_coordinator; print('import OK')\"")


if __name__ == "__main__":
    main()
