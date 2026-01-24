#!/bin/bash
# patch_rocm_warmup_barrier.sh
# Team Beta: Warm-up Barrier Patch (Jan 23, 2026)
#
# PURPOSE:
#   Mitigate + diagnose first-wave transient HIP init / kernel compile races by
#   warming each ROCm GPU sequentially ONCE per node per run before first dispatch.
#
# GUARDRails (Team Beta style):
#   W1) Once per node per run_id (NOT per batch/job)
#   W2) Non-fatal / idempotent
#   W3) Configurable via WARMUP_ROCM env var
#   W4) Synchronous (must complete before first wave)
#
# BEHAVIOR:
#   Default: WARMUP_ROCM=1 (enabled for ROCm nodes, skips localhost)
#   WARMUP_ROCM=0     -> disable
#   WARMUP_ROCM=force -> include localhost too
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash patch_rocm_warmup_barrier.sh

set -e
cd ~/distributed_prng_analysis

echo "=============================================="
echo "ROCm WARM-UP BARRIER PATCH"
echo "Team Beta: Jan 23, 2026"
echo "=============================================="

cp scripts_coordinator.py scripts_coordinator.py.bak_warmup
echo "✓ Backed up to scripts_coordinator.py.bak_warmup"

python3 << 'PATCH_SCRIPT'
import sys

path = "scripts_coordinator.py"
with open(path, "r") as f:
    content = f.read()

errors = []

def already_patched(txt: str) -> bool:
    return ("def warmup_rocm_on_nodes" in txt) and ("WARMUP_ROCM" in txt)

if already_patched(content):
    print("✓ Already patched (idempotent) - no changes needed")
    sys.exit(0)

# ------------------------------------------------------------
# PATCH 1: Ensure required imports exist (idempotent)
# ------------------------------------------------------------
# We need: sys (already), os, subprocess, List typing
if "import os" not in content:
    if "import sys" in content:
        content = content.replace("import sys", "import sys\nimport os", 1)
        print("✓ Added import os")
    else:
        errors.append("Patch failed: anchor 'import sys' not found for os import")

if "import subprocess" not in content:
    if "import os" in content:
        content = content.replace("import os", "import os\nimport subprocess", 1)
        print("✓ Added import subprocess")
    else:
        errors.append("Patch failed: anchor 'import os' not found for subprocess import")

# Add List to typing imports if possible
if "from typing import" in content:
    lines = content.splitlines()
    idx = next((i for i,l in enumerate(lines) if l.strip().startswith("from typing import")), None)
    if idx is not None:
        tl = lines[idx]
        if "List" not in tl:
            # insert List, keeping existing style
            tl_new = tl.replace("from typing import", "from typing import List, ", 1)
            lines[idx] = tl_new
            content = "\n".join(lines)
            print("✓ Added List to typing imports")
        else:
            print("✓ List already present in typing imports")
else:
    print("✓ typing import line not found (ok if code uses builtins/annotations)")

# ------------------------------------------------------------
# PATCH 2: Add WARMUP_ROCM config near CLEAR_HIP_CACHE block
# ------------------------------------------------------------
# Anchor: CLEAR_HIP_CACHE line inserted by your v2 patch
anchor = "CLEAR_HIP_CACHE = os.environ.get('CLEAR_HIP_CACHE', '1')"
if anchor not in content:
    errors.append("Patch failed: CLEAR_HIP_CACHE anchor not found (apply hip cache v2 patch first?)")
else:
    insert = """CLEAR_HIP_CACHE = os.environ.get('CLEAR_HIP_CACHE', '1')  # Default ON for ROCm nodes

# ROCm warm-up barrier (Team Beta: Jan 23, 2026)
# Warms each GPU sequentially ONCE per node per run_id to reduce first-wave init/compile races
# Set WARMUP_ROCM=0 to disable, WARMUP_ROCM=force to include localhost
WARMUP_ROCM = os.environ.get('WARMUP_ROCM', '1')  # Default ON for ROCm nodes"""
    content = content.replace(anchor, insert, 1)
    print("✓ PATCH 2: WARMUP_ROCM config added")

# ------------------------------------------------------------
# PATCH 3: Insert warmup_rocm_on_nodes() before JobResult dataclass
# ------------------------------------------------------------
insert_marker = "@dataclass\nclass JobResult:"
if insert_marker not in content:
    errors.append("Patch failed: JobResult anchor not found")
else:
    warmup_fn = r'''def warmup_rocm_on_nodes(nodes: List[NodeConfig], run_id: str) -> None:
    """
    Warm ROCm GPUs sequentially once per node per run_id.

    Goal: mitigate + diagnose first-wave transient failures caused by parallel HIP init
    and/or concurrent kernel compilation.

    Guardrails:
    - W1: Once per node per run_id (called once in run() pre-flight)
    - W2: Non-fatal/idempotent (ignore failures, log warning)
    - W3: Configurable via WARMUP_ROCM env var
    - W4: Synchronous barrier before first dispatch
    """
    if WARMUP_ROCM == '0':
        print("[PRE-FLIGHT] ROCm warm-up disabled (WARMUP_ROCM=0)")
        return

    # Select ROCm nodes (same stable heuristic as HIP cache clear)
    nodes_to_warm = []
    for node in nodes:
        if node.hostname == 'localhost' and WARMUP_ROCM != 'force':
            continue
        is_rocm = 'rocm' in node.python_env.lower()
        if is_rocm or WARMUP_ROCM == 'force':
            nodes_to_warm.append(node)

    if not nodes_to_warm:
        print("[PRE-FLIGHT] No ROCm nodes selected for warm-up")
        return

    node_names = [n.hostname for n in nodes_to_warm]
    print(f"[PRE-FLIGHT] ROCm warm-up barrier for run_id={run_id}")
    print(f"[PRE-FLIGHT] Warming GPUs sequentially on: {', '.join(node_names)}")

    # Tiny matmul per GPU; sequential loop prevents init storm.
    # Keep workload small but non-trivial to trigger kernel compile + synchronize.
    remote_py = r"""
import torch
n = torch.cuda.device_count()
print(f'WARMUP gpu_count={n}')
for i in range(n):
    torch.cuda.set_device(i)
    x = torch.randn(512, 512, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
print('WARMUP_OK')
"""

    for node in nodes_to_warm:
        try:
            # Run under node.python_env (venv python path from distributed_config.json)
            cmd = (
                f"{node.python_env} - << 'PY'\n"
                f"{remote_py}\n"
                f"PY"
            )
            result = subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                 f'{node.username}@{node.hostname}', cmd],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                # Print a compact success line; keep stdout trimmed for logs
                print(f"[PRE-FLIGHT] {node.hostname}: warmup ok")
            else:
                print(f"[PRE-FLIGHT] {node.hostname}: warmup warning (exit {result.returncode})")
                if result.stderr:
                    print(f"[PRE-FLIGHT] {node.hostname}: stderr: {result.stderr.strip()[:300]}")
        except subprocess.TimeoutExpired:
            print(f"[PRE-FLIGHT] {node.hostname}: warmup timeout (continuing anyway)")
        except Exception as e:
            print(f"[PRE-FLIGHT] {node.hostname}: warmup error {e} (continuing anyway)")


'''
    content = content.replace(insert_marker, warmup_fn + insert_marker, 1)
    print("✓ PATCH 3: warmup_rocm_on_nodes() added")

# ------------------------------------------------------------
# PATCH 4: Call warmup right after HIP cache clear (barrier order)
# ------------------------------------------------------------
call_anchor = "clear_hip_cache_on_nodes(self.nodes, self.run_id)"
if call_anchor not in content:
    errors.append("Patch failed: hip cache clear call anchor not found (expected from hip cache patch v2)")
else:
    # Insert immediately after the call (once per run)
    injection = (
        "clear_hip_cache_on_nodes(self.nodes, self.run_id)\n"
        "        warmup_rocm_on_nodes(self.nodes, self.run_id)\n"
    )
    content = content.replace(call_anchor, injection, 1)
    print("✓ PATCH 4: warmup call inserted after hip cache clear")

# ------------------------------------------------------------
# Finalize
# ------------------------------------------------------------
if errors:
    print("\n❌ PATCH FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

with open(path, "w") as f:
    f.write(content)

print("\n✓ Warm-up barrier patch applied successfully")
PATCH_SCRIPT

echo ""
echo "Verifying patch..."

grep -q "WARMUP_ROCM" scripts_coordinator.py && echo "✓ WARMUP_ROCM config present"
grep -q "def warmup_rocm_on_nodes" scripts_coordinator.py && echo "✓ warmup_rocm_on_nodes() present"
grep -q "warmup_rocm_on_nodes(self.nodes, self.run_id)" scripts_coordinator.py && echo "✓ warmup call present"

echo ""
echo "=============================================="
echo "PATCH COMPLETE"
echo "=============================================="
echo ""
echo "Default behavior:"
echo "  - Warm-up enabled on ROCm nodes once per run"
echo "Env controls:"
echo "  WARMUP_ROCM=0      -> disable warm-up"
echo "  WARMUP_ROCM=force  -> warm localhost too"
echo ""
echo "Expected pre-flight logs:"
echo "  [PRE-FLIGHT] ROCm warm-up barrier for run_id=..."
echo "  [PRE-FLIGHT] Warming GPUs sequentially on: 192.168.3.120, 192.168.3.154"
echo "  [PRE-FLIGHT] 192.168.3.120: warmup ok"
echo "  [PRE-FLIGHT] 192.168.3.154: warmup ok"
