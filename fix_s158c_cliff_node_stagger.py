#!/usr/bin/env python3
"""
fix_s158c_cliff_node_stagger.py

S158C — ROCM_CLIFF_NODES inter-node spawn stagger

ROOT CAUSE OF CRASH:
rrig6600c crashes during worker spawn when all 3 rigs spawn simultaneously.
The PWC startup loop spawns nodes sequentially but with ZERO inter-node pause.
By the time rrig6600c starts spawning (after rrig6600 + rrig6600b finish),
16 ROCm workers are already active. Adding 8 more HIP initializations on
rrig6600c simultaneously causes a kernel panic.

This is distinct from the S158B thread dispatch fix — the crash happens
during startup(), not during run_sieve_pass().

FIX:
1. Add ROCM_CLIFF_NODES constant — list of nodes requiring extra inter-node
   stagger before their spawn begins.
2. Add ROCM_INTER_NODE_STAGGER_S constant (15s) — pause between nodes.
3. Apply stagger after each node's workers finish spawning, with extra
   stagger for CLIFF nodes.

Deploy:
  scp ~/Downloads/fix_s158c_cliff_node_stagger.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 fix_s158c_cliff_node_stagger.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s158c_cliff")

# Add constants after existing ROCM constants
CONST_OLD = "ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id"

CONST_NEW = """ROCM_SPAWN_STAGGER_S   = 4.0   # seconds between worker spawns per gpu_id
# [S158C] Inter-node stagger — pause between finishing one node's workers
# and starting the next node's spawn. Prevents combined ROCm init overload
# when multiple rigs have active workers simultaneously.
ROCM_INTER_NODE_STAGGER_S = 10.0  # seconds between nodes
# [S158C] Cliff nodes — require extra inter-node stagger before spawn.
# These nodes are sensitive to ROCm init load from other active workers.
ROCM_CLIFF_NODES = ["192.168.3.162"]  # rrig6600c"""

# Add inter-node stagger after each node's worker spawn loop
OLD_SPAWN_LOOP = """                # Stagger to prevent simultaneous HIP init (S130/S133 lesson)
                if gpu_id < pool - 1:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
        self._started = True"""

NEW_SPAWN_LOOP = """                # Stagger to prevent simultaneous HIP init (S130/S133 lesson)
                if gpu_id < pool - 1:
                    time.sleep(ROCM_SPAWN_STAGGER_S)
            # [S158C] Inter-node stagger — let workers stabilize before
            # spawning next node. Extra stagger for CLIFF nodes.
            _is_cliff = node.hostname in ROCM_CLIFF_NODES
            _inter_stagger = (ROCM_INTER_NODE_STAGGER_S * 1.5) if _is_cliff else ROCM_INTER_NODE_STAGGER_S
            self.logger.info(
                f"  [S158C] Inter-node stagger {_inter_stagger}s "
                f"after {node.hostname} "
                f"({'CLIFF node' if _is_cliff else 'standard'})"
            )
            time.sleep(_inter_stagger)
        self._started = True"""


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    if "S158C" in content:
        print("S158C already applied — skipping")
        return True

    # Verify anchors
    if CONST_OLD not in content:
        print("ERROR: constants anchor not found")
        return False

    if OLD_SPAWN_LOOP not in content:
        print("ERROR: spawn loop anchor not found")
        idx = content.find("if gpu_id < pool - 1:")
        if idx >= 0:
            print("Context:")
            print(repr(content[idx:idx+200]))
        return False

    # Apply both patches
    content = content.replace(CONST_OLD, CONST_NEW, 1)
    content = content.replace(OLD_SPAWN_LOOP, NEW_SPAWN_LOOP, 1)

    # Validate
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"ERROR: Syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(content)

    try:
        ast.parse(TARGET.read_text())
        print("✅ S158C cliff-node stagger patch applied and verified")
        print(f"   ROCM_INTER_NODE_STAGGER_S = 10.0s")
        print(f"   ROCM_CLIFF_NODES = ['192.168.3.162']")
        print(f"   Cliff node stagger = 15.0s")
        print("\nNext steps:")
        print("  git add -f persistent_worker_coordinator.py fix_s158c_cliff_node_stagger.py")
        print("  git commit -m 'fix(s158c): ROCM_CLIFF_NODES inter-node spawn stagger'")
        print("  git push origin main && git push public main")
        return True
    except SyntaxError as e:
        print(f"ERROR: Post-write syntax error: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("Restored backup")
        return False


if __name__ == "__main__":
    print("Applying S158C cliff-node stagger patch...")
    apply()
