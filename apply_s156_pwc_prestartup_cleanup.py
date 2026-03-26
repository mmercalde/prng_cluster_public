#!/usr/bin/env python3
"""
apply_s156_pwc_prestartup_cleanup.py  (v2 — TB modifications applied)

TEMPORARY SAFETY BANDAID — not the root fix.
Root fix is session-scoped PWC (Phase B, S157).

Adds a pre-spawn cleanup step to PersistentWorkerCoordinator.startup()
that kills any existing sieve_gpu_worker processes on remote nodes
before spawning new workers.

This prevents zombie worker accumulation when a rig reboots mid-run
and rejoins — the new PWC instance was spawning 8 fresh workers while
old workers from the previous PWC were still running, causing py_procs
to reach 12+ and PageTable overflow crashes.

Root cause: Each trial creates a new PWC instance. When a rig reboots
and rejoins, the new trial's PWC spawns fresh workers without cleaning
up the previous trial's workers still running on the rig.

Fix: Before spawning workers in startup(), SSH to each remote node and
kill all existing sieve_gpu_worker processes. This ensures a clean slate.
"""

import re
import shutil
import subprocess
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s156_prestartup")

# The cleanup code to inject before the spawn loop
CLEANUP_CODE = '''        # [S156] Pre-spawn cleanup — kill any existing sieve_gpu_worker processes
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
        _time.sleep(2)

'''

def apply_patch():
    if not TARGET.exists():
        print(f"ERROR: Target file not found: {TARGET}")
        return False

    # Read current content
    content = TARGET.read_text()

    # Find the insertion point — right before the spawn loop
    # Look for the log line that precedes the spawn loop
    insert_before = '        self.logger.info("Starting persistent worker pool...")'
    
    if insert_before not in content:
        print(f"ERROR: Could not find insertion point in {TARGET}")
        print("Looking for:")
        print(f"  {insert_before}")
        return False

    if "[S156] Pre-spawn cleanup" in content:
        print("Patch already applied — skipping")
        return True

    # Create backup
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    # Apply patch
    new_content = content.replace(
        insert_before,
        CLEANUP_CODE + insert_before
    )

    TARGET.write_text(new_content)
    print(f"Patch applied to {TARGET}")

    # Verify
    verify = TARGET.read_text()
    if "[S156] Pre-spawn cleanup" in verify:
        print("✅ Patch verified successfully")
        return True
    else:
        print("❌ Patch verification failed — restoring backup")
        shutil.copy2(BACKUP, TARGET)
        return False


if __name__ == "__main__":
    print("Applying S156 PWC pre-startup cleanup patch...")
    success = apply_patch()
    if success:
        print("\nDone. Deploy with:")
        print("  scp ~/Downloads/apply_s156_pwc_prestartup_cleanup.py rzeus:~/distributed_prng_analysis/")
        print("  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s156_pwc_prestartup_cleanup.py'")
    else:
        print("\nPatch failed.")
