#!/usr/bin/env python3
"""
S137 patch: Add --seed-cap-nvidia and --seed-cap-amd to window_optimizer.py

Problem: agent_manifests/window_optimizer.json (S131) added seed-cap-nvidia and
seed-cap-amd to actions[0].args_map so WATCHER would pass them through. But
window_optimizer.py argparse never got these args — causing 'unrecognized arguments'
error and immediate exit code 2.

Fix (3 sites in window_optimizer.py):
  1. run_bayesian_optimization() signature — add seed_cap_nvidia, seed_cap_amd params
  2. coordinator attribute wiring — set on coordinator object (same pattern as S134)
  3. argparse — add --seed-cap-nvidia and --seed-cap-amd
  4. main() bayesian call — pass through via getattr
"""

import re
import shutil
from pathlib import Path

TARGET = Path('window_optimizer.py')
BACKUP = Path('window_optimizer.py.bak_s137_pre')

assert TARGET.exists(), f"ERROR: {TARGET} not found — run from ~/distributed_prng_analysis/"

# Backup
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

# ── PATCH 1: run_bayesian_optimization() signature ──────────────────────────
OLD1 = (
    "    use_persistent_workers: bool = False,   # S134\n"
    "    worker_pool_size: int = 8,             # S134\n"
    ") -> Dict[str, Any]:"
)
NEW1 = (
    "    use_persistent_workers: bool = False,   # S134\n"
    "    worker_pool_size: int = 8,             # S134\n"
    "    seed_cap_nvidia: int = 5_000_000,      # S137\n"
    "    seed_cap_amd: int = 2_000_000,         # S137\n"
    ") -> Dict[str, Any]:"
)
assert content.count(OLD1) == 1, f"PATCH 1 anchor not found exactly once"
content = content.replace(OLD1, NEW1)
print("✅ Patch 1: run_bayesian_optimization() signature — seed_cap params added")

# ── PATCH 2: coordinator attribute wiring ────────────────────────────────────
OLD2 = (
    "    # S134: wire persistent worker flags onto coordinator so integration gate can read them\n"
    "    coordinator.use_persistent_workers = use_persistent_workers\n"
    "    coordinator.worker_pool_size        = worker_pool_size\n"
    "    if use_persistent_workers:\n"
    "        print(f\"   [S134] Persistent worker mode ENABLED (pool_size={worker_pool_size} per rig)\")"
)
NEW2 = (
    "    # S134: wire persistent worker flags onto coordinator so integration gate can read them\n"
    "    coordinator.use_persistent_workers = use_persistent_workers\n"
    "    coordinator.worker_pool_size        = worker_pool_size\n"
    "    if use_persistent_workers:\n"
    "        print(f\"   [S134] Persistent worker mode ENABLED (pool_size={worker_pool_size} per rig)\")\n"
    "    # S137: wire seed cap flags onto coordinator so integration final can read them\n"
    "    coordinator.seed_cap_nvidia = seed_cap_nvidia\n"
    "    coordinator.seed_cap_amd    = seed_cap_amd"
)
assert content.count(OLD2) == 1, f"PATCH 2 anchor not found exactly once"
content = content.replace(OLD2, NEW2)
print("✅ Patch 2: coordinator attribute wiring — seed_cap_nvidia/amd set on coordinator")

# ── PATCH 3: argparse ─────────────────────────────────────────────────────────
OLD3 = (
    "    parser.add_argument('--use-persistent-workers', action='store_true', default=False,\n"
    "                       help='[S134] Use persistent worker engine instead of subprocess sieve. '\n"
    "                            'Workers stay alive across all 4 sieve passes per trial.')\n"
    "    parser.add_argument('--worker-pool-size', type=int, default=8,\n"
    "                       help='[S134] Number of persistent workers to spawn per rig (default: 8).')"
)
NEW3 = (
    "    parser.add_argument('--use-persistent-workers', action='store_true', default=False,\n"
    "                       help='[S134] Use persistent worker engine instead of subprocess sieve. '\n"
    "                            'Workers stay alive across all 4 sieve passes per trial.')\n"
    "    parser.add_argument('--worker-pool-size', type=int, default=8,\n"
    "                       help='[S134] Number of persistent workers to spawn per rig (default: 8).')\n"
    "    parser.add_argument('--seed-cap-nvidia', type=int, default=5_000_000,\n"
    "                       help='[S137] Max seeds per job chunk for NVIDIA GPUs (default: 5000000).')\n"
    "    parser.add_argument('--seed-cap-amd', type=int, default=2_000_000,\n"
    "                       help='[S137] Max seeds per job chunk for AMD GPUs (default: 2000000).')"
)
assert content.count(OLD3) == 1, f"PATCH 3 anchor not found exactly once"
content = content.replace(OLD3, NEW3)
print("✅ Patch 3: argparse — --seed-cap-nvidia and --seed-cap-amd added")

# ── PATCH 4: main() bayesian call ────────────────────────────────────────────
OLD4 = (
    "            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134\n"
    "            worker_pool_size=getattr(args, 'worker_pool_size', 8),                  # S134\n"
    "        )"
)
NEW4 = (
    "            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134\n"
    "            worker_pool_size=getattr(args, 'worker_pool_size', 8),                  # S134\n"
    "            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),            # S137\n"
    "            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),                  # S137\n"
    "        )"
)
assert content.count(OLD4) == 1, f"PATCH 4 anchor not found exactly once"
content = content.replace(OLD4, NEW4)
print("✅ Patch 4: main() bayesian call — seed_cap_nvidia/amd wired through")

assert content != original, "ERROR: No changes made"
TARGET.write_text(content)
print(f"\n✅ All 4 patches applied to {TARGET}")

# Verify
import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    print("Restoring backup...")
    shutil.copy2(BACKUP, TARGET)
    raise

# Quick sanity check
assert '--seed-cap-nvidia' in content
assert '--seed-cap-amd' in content
assert 'seed_cap_nvidia: int = 5_000_000' in content
assert 'coordinator.seed_cap_nvidia = seed_cap_nvidia' in content
print("✅ Sanity checks passed")
print("\nDeploy with:")
print("  scp ~/Downloads/apply_s137_seed_cap_argparse.py rzeus:~/distributed_prng_analysis/")
print("  ssh rzeus 'cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && python3 apply_s137_seed_cap_argparse.py'")
