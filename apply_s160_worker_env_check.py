#!/usr/bin/env python3
"""
apply_s160_worker_env_check.py
==============================
S160 hardening patch (TB-recommended):
  Adds _verify_worker_env() to ZMQSQLiteCoordinator and calls it
  45 seconds after _launch_workers() completes.

  If any required ROCm protective env var is missing from a live worker
  PID, coordinator logs an ERROR and raises RuntimeError to halt the run
  before wasting cluster time on a configuration that will crash.

Required vars verified on each AMD rig worker (gpu0):
  HSA_ENABLE_SDMA=0
  HSA_ENABLE_RUNTIME_POWER_MGMT=0
  AMDGPU_NO_POWER_PROFILE=1
  HSA_OVERRIDE_GFX_VERSION=10.3.0
  ROCR_VISIBLE_DEVICES=0   (checked as non-empty / present)

Usage:
  python3 apply_s160_worker_env_check.py [--dry-run]
"""

import ast
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("zmq_sqlite_coordinator.py")
BACKUP = Path("zmq_sqlite_coordinator.py.pre_s160_env_check")

# ---------------------------------------------------------------------------
# New method to insert after _launch_workers (entire method as a string)
# ---------------------------------------------------------------------------
_VERIFY_METHOD = '''
    # --- S160: TB-recommended env-invariant check ---
    _REQUIRED_WORKER_VARS = {
        "HSA_ENABLE_SDMA":              "0",
        "HSA_ENABLE_RUNTIME_POWER_MGMT": "0",
        "AMDGPU_NO_POWER_PROFILE":       "1",
        "HSA_OVERRIDE_GFX_VERSION":      "10.3.0",
    }

    def _verify_worker_env(self, host: str, username: str) -> bool:
        """
        SSH to host, find a live zmq_sqlite_worker PID, read /proc/<pid>/environ,
        and verify all required ROCm protective vars are set to correct values.

        Returns True if all vars pass. Logs ERROR and returns False otherwise.
        Called 45 s after _launch_workers() in _launch_and_verify().
        """
        import subprocess as _sp
        cmd = (
            "pid=$(pgrep -f zmq_sqlite_worker | head -1); "
            "[ -n \\"$pid\\" ] && cat /proc/$pid/environ | tr \\'\\\\0\\' \\'\\\\n\\'"
        )
        try:
            r = _sp.run(
                ["ssh", "-q", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
                 f"{username}@{host}", "bash", "-c", cmd],
                capture_output=True, text=True, timeout=20
            )
        except Exception as exc:
            self.logger.warning(f"[EnvCheck] SSH error on {host}: {exc}")
            return False

        if r.returncode != 0:
            self.logger.warning(
                f"[EnvCheck] No live zmq_sqlite_worker found on {host} "
                f"(pgrep returned non-zero). Worker may still be starting."
            )
            return False

        env_dict = {}
        for line in r.stdout.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env_dict[k] = v

        missing = []
        for var, expected in self._REQUIRED_WORKER_VARS.items():
            actual = env_dict.get(var)
            if actual != expected:
                missing.append(f"{var}={actual!r} (expected {expected!r})")

        # ROCR_VISIBLE_DEVICES just needs to be present (value depends on gpu_id)
        if "ROCR_VISIBLE_DEVICES" not in env_dict:
            missing.append("ROCR_VISIBLE_DEVICES=MISSING")

        if missing:
            self.logger.error(
                f"[EnvCheck] FAIL on {host}: {missing}"
            )
            return False

        self.logger.info(f"[EnvCheck] PASS on {host}: all ROCm protective vars confirmed")
        return True

    def _launch_and_verify(self):
        """
        Launch workers, wait for settle, then verify env on each AMD rig.
        Raises RuntimeError if any rig fails the env check.
        """
        self._launch_workers()

        # Give systemd-run workers time to start before checking /proc
        self.logger.info("[EnvCheck] Waiting 45 s for workers to initialize...")
        import time as _time
        _time.sleep(45)

        failed_hosts = []
        for node in self._nodes:
            host = node.get("hostname", "")
            if host in ("localhost", "127.0.0.1"):
                continue  # CUDA workers on Zeus — different env path
            username = node.get("username", "michael")
            if not self._verify_worker_env(host, username):
                failed_hosts.append(host)

        if failed_hosts:
            raise RuntimeError(
                f"[EnvCheck] Env-invariant check FAILED on: {failed_hosts}. "
                "ROCm protective vars missing. Check /etc/environment and "
                "systemd-run --setenv args in zmq_sqlite_coordinator.py. "
                "Halting to prevent GPU crash."
            )
        self.logger.info("[EnvCheck] All AMD rigs passed env-invariant check.")
'''

# ---------------------------------------------------------------------------
# Anchor: insert _verify_method BEFORE def run_sieve_pass
# ---------------------------------------------------------------------------
ANCHOR = "    def run_sieve_pass("

# ---------------------------------------------------------------------------
# Also patch the call site: replace _launch_workers() call inside run_sieve_pass
# with _launch_and_verify() — but only when workers haven't been launched yet.
# ---------------------------------------------------------------------------
OLD_CALL = "        # Launch workers (no-op after first pass — _workers_launched guard)\n        self._launch_workers()"
NEW_CALL = "        # Launch workers (no-op after first pass — _workers_launched guard)\n        # S160: first call also runs env-invariant check (45 s settle + SSH verify)\n        self._launch_and_verify()"


def apply(dry_run: bool = False):
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from distributed_prng_analysis/")
        sys.exit(1)

    text = TARGET.read_text()

    # --- Verify anchors ---
    if ANCHOR not in text:
        print(f"ERROR: Anchor not found:\n  {ANCHOR!r}")
        sys.exit(1)
    if OLD_CALL not in text:
        print(f"ERROR: Call-site anchor not found:\n  {OLD_CALL!r}")
        sys.exit(1)

    # --- Apply ---
    # 1. Insert method before run_sieve_pass
    text_new = text.replace(ANCHOR, _VERIFY_METHOD + "\n" + ANCHOR, 1)

    # 2. Replace call site
    text_new = text_new.replace(OLD_CALL, NEW_CALL, 1)

    # --- AST check ---
    try:
        ast.parse(text_new)
    except SyntaxError as e:
        print(f"ERROR: AST check failed: {e}")
        sys.exit(1)

    if dry_run:
        print("DRY RUN — no files modified. AST check passed.")
        print(f"Would backup: {TARGET} -> {BACKUP}")
        print(f"Would write: {TARGET} ({len(text_new)} chars)")
        return

    shutil.copy(TARGET, BACKUP)
    TARGET.write_text(text_new)
    print(f"Backup: {BACKUP}")
    print(f"Patched: {TARGET}")
    print("AST OK. _verify_worker_env() + _launch_and_verify() added.")
    print("Call site updated: _launch_workers() -> _launch_and_verify()")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
