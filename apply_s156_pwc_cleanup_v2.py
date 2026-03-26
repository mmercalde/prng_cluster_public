#!/usr/bin/env python3
"""
apply_s156_pwc_cleanup_v2.py

Upgrades the S156 pre-spawn cleanup from v1 (broad pkill -9) to v2 
per Team Beta requirements:
- Targeted match: sieve_gpu_worker.*--persistent (not all sieve workers)
- SIGTERM first, SIGKILL only on timeout
- Log exact processes found and reaped
- Scope by --persistent flag

Deploy:
  scp ~/Downloads/apply_s156_pwc_cleanup_v2.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s156_pwc_cleanup_v2.py'
"""

import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s156_v1")

OLD_BLOCK = '''        # [S156] Pre-spawn cleanup — kill any existing sieve_gpu_worker processes
        # on remote nodes before spawning new workers. Prevents zombie worker
        # accumulation when a rig reboots mid-run and rejoins with stale workers
        # from a previous PWC instance still running.
        for node in self.nodes:
            if self._is_localhost(node.hostname):
                continue
            if not self._is_rocm(node):
                continue
            try:
                cleanup_cmd = (
                    f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                    f"-o BatchMode=yes "
                    f"{node.username}@{node.hostname} "
                    f"'pkill -9 -f sieve_gpu_worker 2>/dev/null; sleep 1; echo cleanup_done'"
                )
                result = subprocess.run(cleanup_cmd, shell=True, capture_output=True,
                                      text=True, timeout=15)
                if "cleanup_done" in result.stdout:
                    self.logger.info(f"  [S156] {node.hostname}: pre-spawn cleanup done")
                else:
                    self.logger.warning(f"  [S156] {node.hostname}: pre-spawn cleanup uncertain")
            except Exception as e:
                self.logger.warning(f"  [S156] {node.hostname}: pre-spawn cleanup failed: {e}")
        # Small delay after cleanup to allow ROCm contexts to fully release
        import time as _time
        _time.sleep(2)'''

NEW_BLOCK = '''        # ── [S156-BANDAID v2] Pre-spawn targeted cleanup ──────────────────────
        # TEMPORARY SAFETY NET — not the root fix.
        # Root fix: session-scoped PWC (Phase B, S157).
        # TB requirements: targeted --persistent match, SIGTERM first, log exact procs.
        import subprocess as _s156_sp
        import time as _s156_time
        for _s156_node in self.nodes:
            if self._is_localhost(_s156_node.hostname):
                continue
            if not self._is_rocm(_s156_node):
                continue
            try:
                # Find matching persistent workers only
                _s156_find = (
                    f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                    f"-o BatchMode=yes "
                    f"{_s156_node.username}@{_s156_node.hostname} "
                    f"'pgrep -af \"sieve_gpu_worker.*--persistent\" 2>/dev/null "
                    f"|| echo none'"
                )
                _s156_r = _s156_sp.run(
                    _s156_find, shell=True, capture_output=True, text=True, timeout=10
                )
                _s156_found = _s156_r.stdout.strip()
                if _s156_found and _s156_found != "none":
                    self.logger.info(
                        f"  [S156] {_s156_node.hostname}: found stale workers: "
                        f"{_s156_found[:200]}"
                    )
                    # SIGTERM first, escalate to SIGKILL
                    _s156_reap = (
                        f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                        f"-o BatchMode=yes "
                        f"{_s156_node.username}@{_s156_node.hostname} "
                        f"'pkill -15 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
                        f"sleep 2; "
                        f"pkill -9 -f \"sieve_gpu_worker.*--persistent\" 2>/dev/null; "
                        f"sleep 1; "
                        f"remaining=$(pgrep -c -f \"sieve_gpu_worker.*--persistent\" "
                        f"2>/dev/null || echo 0); "
                        f"echo \"reaped:$remaining\"'"
                    )
                    _s156_r2 = _s156_sp.run(
                        _s156_reap, shell=True, capture_output=True, text=True, timeout=15
                    )
                    self.logger.info(
                        f"  [S156] {_s156_node.hostname}: "
                        f"reap result: {_s156_r2.stdout.strip()}"
                    )
                else:
                    self.logger.info(
                        f"  [S156] {_s156_node.hostname}: no stale persistent workers"
                    )
            except Exception as _s156_e:
                self.logger.warning(
                    f"  [S156] {_s156_node.hostname}: pre-spawn cleanup failed: {_s156_e}"
                )
        # Allow ROCm contexts to fully release after cleanup
        _s156_time.sleep(2)
        # ── end [S156-BANDAID v2] ───────────────────────────────────────────'''


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    if OLD_BLOCK not in content:
        if "[S156-BANDAID v2]" in content:
            print("v2 already applied — skipping")
            return True
        print("ERROR: v1 block not found — cannot upgrade")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    new_content = content.replace(OLD_BLOCK, NEW_BLOCK)
    TARGET.write_text(new_content)

    verify = TARGET.read_text()
    if "[S156-BANDAID v2]" in verify and OLD_BLOCK not in verify:
        print(f"✅ Upgraded to v2: {TARGET}")
        return True
    else:
        print("❌ Verification failed — restoring backup")
        shutil.copy2(BACKUP, TARGET)
        return False


if __name__ == "__main__":
    print("Upgrading S156 cleanup to v2 (TB-compliant)...")
    success = apply()
    if success:
        print("\nNext steps:")
        print("  git add persistent_worker_coordinator.py apply_s156_pwc_cleanup_v2.py")
        print("  git commit -m 'fix(s156-bandaid-v2): targeted SIGTERM-first cleanup per TB'")
        print("  git push origin main && git push public main")
