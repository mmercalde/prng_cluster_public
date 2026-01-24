#!/bin/bash
# patch_hip_cache_clear_v2.sh
# Team Beta Approved: January 23, 2026
# Version 2: All required corrections applied
#
# ISSUE: GPU2 first-wave failures due to stale HIP kernel cache
# FIX: Clear HIP cache once per node per run_id before first batch
#
# Guardrails (Team Beta mandated):
#   B1) Once per node per run_id (NOT per batch, NOT per job)
#   B2) Non-fatal / idempotent (ignore failures, log warning)
#   B3) Configurable (CLEAR_HIP_CACHE env var)
#   B4) Synchronous (wait before launching first wave)
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash patch_hip_cache_clear_v2.sh

set -e
cd ~/distributed_prng_analysis

echo "=============================================="
echo "HIP CACHE CLEAR PATCH v2"
echo "Team Beta Approved: January 23, 2026"
echo "=============================================="

# Backup
cp scripts_coordinator.py scripts_coordinator.py.bak
echo "✓ Backed up to scripts_coordinator.py.bak"

# Apply patch using Python
python3 << 'PATCH_SCRIPT'
import sys

with open('scripts_coordinator.py', 'r') as f:
    content = f.read()

errors = []

# ============================================================
# PATCH 1: Add configuration constant near other constants
# ============================================================

old_config = '''ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)'''

if old_config not in content:
    errors.append("Patch failed: ENABLE_ALLOCATOR_RESET anchor not found")
else:
    new_config = '''ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)

# HIP kernel cache clear (Team Beta ruling: Jan 23, 2026)
# Clears stale compiled kernels that cause "invalid device function" errors
# Set CLEAR_HIP_CACHE=0 to disable, CLEAR_HIP_CACHE=force to clear even localhost
CLEAR_HIP_CACHE = os.environ.get('CLEAR_HIP_CACHE', '1')  # Default ON for ROCm nodes'''

    content = content.replace(old_config, new_config)
    print("✓ PATCH 1: CLEAR_HIP_CACHE config added")

# ============================================================
# PATCH 2: Ensure required imports exist (idempotent)
# ============================================================

# Check and add os import if missing
if 'import os' not in content:
    # Add after first import line
    content = content.replace('import sys', 'import sys\nimport os', 1)
    print("✓ PATCH 2a: Added 'import os'")
else:
    print("✓ PATCH 2a: 'import os' already present")

# Check and add subprocess import if missing
if 'import subprocess' not in content:
    content = content.replace('import os', 'import os\nimport subprocess', 1)
    print("✓ PATCH 2b: Added 'import subprocess'")
else:
    print("✓ PATCH 2b: 'import subprocess' already present")

# Check List is in typing imports
if 'from typing import' in content:
    typing_line = [l for l in content.split('\n') if 'from typing import' in l][0]
    if 'List' not in typing_line:
        new_typing = typing_line.replace('from typing import', 'from typing import List, ')
        content = content.replace(typing_line, new_typing)
        print("✓ PATCH 2c: Added 'List' to typing imports")
    else:
        print("✓ PATCH 2c: 'List' already in typing imports")
else:
    print("✓ PATCH 2c: typing imports not found (may use annotations)")

# ============================================================
# PATCH 3: Add the cache clear function
# ============================================================

insert_marker = '''@dataclass
class JobResult:'''

if insert_marker not in content:
    errors.append("Patch failed: 'class JobResult' anchor not found")
else:
    hip_cache_function = '''def clear_hip_cache_on_nodes(nodes: List[NodeConfig], run_id: str) -> None:
    """
    Clear HIP kernel cache once per node per run_id.
    
    Team Beta ruling (Jan 23, 2026):
    - B1: Once per node per run_id (NOT per batch/job)
    - B2: Non-fatal/idempotent (ignore failures)
    - B3: Configurable via CLEAR_HIP_CACHE env var
    - B4: Synchronous (wait before first wave)
    
    Fixes: "HIP error: invalid device function" from stale compiled kernels
    """
    if CLEAR_HIP_CACHE == '0':
        print("[PRE-FLIGHT] HIP cache clear disabled (CLEAR_HIP_CACHE=0)")
        return
    
    print(f"[PRE-FLIGHT] HIP cache clear for run_id={run_id}")
    
    # Select nodes to clear
    nodes_to_clear = []
    for node in nodes:
        # Skip localhost unless force mode
        if node.hostname == 'localhost' and CLEAR_HIP_CACHE != 'force':
            continue
        # ROCm detection: simple and stable (Team Beta required)
        is_rocm = 'rocm' in node.python_env.lower()
        if is_rocm or CLEAR_HIP_CACHE == 'force':
            nodes_to_clear.append(node)
    
    if not nodes_to_clear:
        print("[PRE-FLIGHT] No ROCm nodes to clear HIP cache")
        return
    
    node_names = [n.hostname for n in nodes_to_clear]
    print(f"[PRE-FLIGHT] Clearing HIP cache (once per run) on: {', '.join(node_names)}")
    
    cache_clear_cmd = "rm -rf ~/.cache/hip ~/.cache/amd_comgr ~/.cache/kernels 2>/dev/null || true"
    
    for node in nodes_to_clear:
        try:
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                 f'{node.username}@{node.hostname}', cache_clear_cmd],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"[PRE-FLIGHT] {node.hostname}: cleared (ok)")
            else:
                print(f"[PRE-FLIGHT] {node.hostname}: cleared (warning: exit {result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"[PRE-FLIGHT] {node.hostname}: timeout (continuing anyway)")
        except Exception as e:
            print(f"[PRE-FLIGHT] {node.hostname}: error {e} (continuing anyway)")


@dataclass
class JobResult:'''

    content = content.replace(insert_marker, hip_cache_function)
    print("✓ PATCH 3: clear_hip_cache_on_nodes() function added")

# ============================================================
# PATCH 4: Call the function before first job dispatch
# ============================================================

old_execute_start = '''        print("Executing jobs...")
        print("-" * 60)'''

if old_execute_start not in content:
    errors.append("Patch failed: 'Executing jobs...' anchor not found")
else:
    new_execute_start = '''        # Clear HIP cache once per run (Team Beta: Jan 23, 2026)
        clear_hip_cache_on_nodes(self.nodes, self.run_id)
        
        print("Executing jobs...")
        print("-" * 60)'''

    content = content.replace(old_execute_start, new_execute_start)
    print("✓ PATCH 4: Function call added before job dispatch")

# ============================================================
# Final validation
# ============================================================

if errors:
    print("\n❌ PATCH FAILED:")
    for e in errors:
        print(f"   {e}")
    sys.exit(1)

# Write patched file
with open('scripts_coordinator.py', 'w') as f:
    f.write(content)

print("\n✓ All patches applied successfully")
PATCH_SCRIPT

echo ""
echo "Verifying patch..."

# Verification checks
grep -q "CLEAR_HIP_CACHE" scripts_coordinator.py && echo "✓ CLEAR_HIP_CACHE config present"
grep -q "def clear_hip_cache_on_nodes" scripts_coordinator.py && echo "✓ clear_hip_cache_on_nodes() function present"
grep -q "Clear HIP cache once per run" scripts_coordinator.py && echo "✓ Function call before dispatch present"
grep -q "run_id=" scripts_coordinator.py && echo "✓ run_id logging present"
grep -q "'rocm' in node.python_env.lower()" scripts_coordinator.py && echo "✓ Simplified ROCm detection present"

echo ""
echo "=============================================="
echo "PATCH COMPLETE"
echo "=============================================="
echo ""
echo "Team Beta required fixes applied:"
echo "  ✓ Replace-failure assertions added"
echo "  ✓ Import guards (os, subprocess, List)"
echo "  ✓ Simplified ROCm detection"
echo "  ✓ run_id in logging"
echo ""
echo "Behavior:"
echo "  - Default: Clears HIP cache on ROCm nodes before first batch"
echo "  - CLEAR_HIP_CACHE=0    → Disable cache clear"
echo "  - CLEAR_HIP_CACHE=force → Clear even localhost"
echo ""
echo "Expected log output:"
echo "  [PRE-FLIGHT] HIP cache clear for run_id=full_scoring_results_20260123_..."
echo "  [PRE-FLIGHT] Clearing HIP cache (once per run) on: 192.168.3.120, 192.168.3.154"
echo "  [PRE-FLIGHT] 192.168.3.120: cleared (ok)"
echo "  [PRE-FLIGHT] 192.168.3.154: cleared (ok)"
echo ""
echo "Test with:"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3"
