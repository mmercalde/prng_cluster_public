#!/usr/bin/env python3
"""
apply_s151_respawn_stagger.py
==============================
Adds per-node respawn lock with ROCM_SPAWN_STAGGER_S delay to
_ensure_worker_alive() in persistent_worker_coordinator.py.

Problem: When multiple workers on the same rig die simultaneously,
all respawns fire concurrently — N SSH connections + N HIP inits at
once. This is the SSH hammer that crashes rrig6600c.

Fix: Per-node threading.Lock serializes respawns. ROCM_SPAWN_STAGGER_S
delay between each spawn (same as initial startup stagger).

Change 1: Add _node_respawn_locks dict initialization in __init__
Change 2: Add per-node respawn lock + stagger in _ensure_worker_alive
Change 3: Initialize lock entries in _load_config alongside semaphores
"""

import re
import shutil
import sys
import os

TARGET = "persistent_worker_coordinator.py"
BACKUP = "persistent_worker_coordinator.py.bak_s151_respawn"


def apply():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found")
        sys.exit(1)

    content = open(TARGET).read()

    # ── Change 1: Add _node_respawn_locks in __init__ ─────────────────────
    OLD1 = "        # Per-node semaphores — throttle concurrent dispatch to max_per_node (S133-A lesson)\n        self._node_semaphores: Dict[str, threading.Semaphore] = {}"
    NEW1 = "        # Per-node semaphores — throttle concurrent dispatch to max_per_node (S133-A lesson)\n        self._node_semaphores: Dict[str, threading.Semaphore] = {}\n        # [S151] Per-node respawn locks — serialize respawns with stagger to prevent SSH hammer\n        self._node_respawn_locks: Dict[str, threading.Lock] = {}"

    if OLD1 not in content:
        print("ERROR: Change 1 anchor not found")
        sys.exit(1)
    content = content.replace(OLD1, NEW1, 1)
    print("✅ Change 1: _node_respawn_locks dict added to __init__")

    # ── Change 2: Initialize lock in _load_config ──────────────────────────
    OLD2 = "            # Create per-node semaphore — limits concurrent in-flight jobs to max_per_node\n            self._node_semaphores[node.hostname] = threading.Semaphore(self.max_per_node)"
    NEW2 = "            # Create per-node semaphore — limits concurrent in-flight jobs to max_per_node\n            self._node_semaphores[node.hostname] = threading.Semaphore(self.max_per_node)\n            # [S151] Per-node respawn lock — serializes respawns with stagger\n            self._node_respawn_locks[node.hostname] = threading.Lock()"

    if OLD2 not in content:
        print("ERROR: Change 2 anchor not found")
        sys.exit(1)
    content = content.replace(OLD2, NEW2, 1)
    print("✅ Change 2: _node_respawn_locks initialized in _load_config")

    # ── Change 3: Add lock + stagger in _ensure_worker_alive ──────────────
    OLD3 = """    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        \"\"\"Check worker still alive; respawn if dead.\"\"\"
        if handle.quarantined:
            return False
        if handle.proc is None or handle.proc.poll() is not None:
            self.logger.warning(f\"Worker {handle.node.hostname}:GPU{handle.gpu_id} dead — respawning\")
            handle.alive = False
            success = self._spawn_worker(handle)
            if not success:
                handle.quarantined = True
                self.logger.error(f\"Respawn failed — {handle.node.hostname}:GPU{handle.gpu_id} quarantined\")
            return success
        return True"""

    NEW3 = """    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:
        \"\"\"Check worker still alive; respawn if dead.\"\"\"
        if handle.quarantined:
            return False
        if handle.proc is None or handle.proc.poll() is not None:
            self.logger.warning(f\"Worker {handle.node.hostname}:GPU{handle.gpu_id} dead — respawning\")
            handle.alive = False
            # [S151] Per-node respawn lock + stagger — prevents SSH hammer when multiple
            # workers die simultaneously on the same rig (PCIe 1x crypto-miner constraint)
            respawn_lock = self._node_respawn_locks.get(handle.node.hostname)
            if respawn_lock:
                with respawn_lock:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
                    success = self._spawn_worker(handle)
            else:
                success = self._spawn_worker(handle)
            if not success:
                handle.quarantined = True
                self.logger.error(f\"Respawn failed — {handle.node.hostname}:GPU{handle.gpu_id} quarantined\")
            return success
        return True"""

    if OLD3 not in content:
        print("ERROR: Change 3 anchor not found")
        sys.exit(1)
    content = content.replace(OLD3, NEW3, 1)
    print("✅ Change 3: Per-node respawn lock + stagger added to _ensure_worker_alive")

    # Backup and write
    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    open(TARGET, "w").write(content)
    print(f"✅ Patch applied to {TARGET}")

    # Verify
    new_content = open(TARGET).read()
    assert "_node_respawn_locks" in new_content
    assert "Per-node respawn lock + stagger" in new_content
    assert "respawn_lock = self._node_respawn_locks.get" in new_content
    print("✅ Verification passed")


if __name__ == "__main__":
    apply()
