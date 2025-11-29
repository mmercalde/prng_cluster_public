#!/usr/bin/env python3
"""
Python-based updater for prng_registry.py
More reliable than bash sed commands for complex multi-line insertions
"""
import sys
import os
from datetime import datetime

# Check if file exists
if not os.path.exists('prng_registry.py'):
    print("❌ ERROR: prng_registry.py not found in current directory")
    sys.exit(1)

print("=" * 70)
print("prng_registry.py PyTorch GPU Updater (Python)")
print("=" * 70)
print()

# Backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = f"prng_registry.py.backup_{timestamp}"
os.system(f"cp prng_registry.py {backup_file}")
print(f"✅ Backup created: {backup_file}")

# Read original file
with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

print(f"✅ Read {len(lines)} lines from prng_registry.py")

# Find key line numbers
import_numpy_line = None
version_23_line = None
cpu_ref_end_line = None
kernel_registry_line = None
helper_functions_line = None

for i, line in enumerate(lines):
    if 'import numpy as np' in line:
        import_numpy_line = i
    if 'Version 2.3 - October 29, 2025' in line:
        version_23_line = i
    if 'def xorshift32_cpu(' in line and cpu_ref_end_line is None:
        cpu_ref_end_line = i - 1  # Insert before first CPU function
    if 'KERNEL_REGISTRY = {' in line:
        kernel_registry_line = i
    if 'def get_kernel_info(' in line:
        helper_functions_line = i

print(f"✅ Located key positions:")
print(f"   - numpy import: line {import_numpy_line}")
print(f"   - version 2.3: line {version_23_line}")
print(f"   - CPU ref section: line {cpu_ref_end_line}")
print(f"   - KERNEL_REGISTRY: line {kernel_registry_line}")
print(f"   - helper functions: line {helper_functions_line}")

# Build new content
new_lines = []

# Part 1: Add version header update
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # After version 2.3 line, add version 2.4
    if i == version_23_line:
        new_lines.append("""Version 2.4 - November 27, 2025
- ENHANCEMENT: Added PyTorch GPU implementations for Step 2.5 ML scoring
  * PyTorch functions work on both CUDA (NVIDIA) and ROCm (AMD)
  * Backward compatible - all existing code unchanged
  * Phase 1: java_lcg, java_lcg_hybrid implemented
  * Phase 2+: Remaining 44 PRNGs (placeholders ready)

""")
    
    # After numpy import, add PyTorch import
    if i == import_numpy_line:
        new_lines.append("""
# ============================================================================
# PYTORCH IMPORT (v2.4 Addition - For Step 2.5 GPU Scoring)
# ============================================================================
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - GPU scoring disabled (CPU fallback will be used)")
    print("   To enable GPU scoring: pip install torch --break-system-packages")
""")

print("✅ Added version header and PyTorch import")

# Part 2: Add PyTorch GPU functions after CPU references
pytorch_functions = '''
# ============================================================================
# PYTORCH GPU IMPLEMENTATIONS (v2.4 - For Step 2.5 ML Scoring)
# ============================================================================

def java_lcg_pytorch_gpu(
    seeds: 'torch.Tensor',
    n: int,
    mod: int,
    device: str = 'cuda',
    skip: int = 0,
    **kwargs
) -> 'torch.Tensor':
    """PyTorch GPU implementation of Java LCG (java.util.Random)"""
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch --break-system-packages")
    
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    mask = (1 << 48) - 1
    
    seeds = seeds.to(device).long()
    N = seeds.shape[0]
    state = (seeds ^ a) & mask
    
    for _ in range(skip):
        state = (a * state + c) & mask
    
    output = torch.zeros((N, n), dtype=torch.int64, device=device)
    
    for i in range(n):
        state = (a * state + c) & mask
        output[:, i] = (state >> 16) % mod
    
    return output


def java_lcg_hybrid_pytorch_gpu(
    seeds: 'torch.Tensor',
    n: int,
    mod: int,
    device: str = 'cuda',
    skip: int = 0,
    **kwargs
) -> 'torch.Tensor':
    """PyTorch GPU for java_lcg_hybrid"""
    return java_lcg_pytorch_gpu(seeds, n, mod, device, skip, **kwargs)

'''

# Insert PyTorch functions
new_lines.insert(cpu_ref_end_line + 1, pytorch_functions)

print("✅ Added PyTorch GPU functions")

# Part 3: Add helper functions before existing helpers
helper_functions = '''
# ============================================================================
# PYTORCH GPU HELPER FUNCTIONS (v2.4 Addition)
# ============================================================================

def get_pytorch_gpu_reference(prng_family: str) -> Callable:
    """Get PyTorch GPU implementation for PRNG family."""
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch --break-system-packages")
    
    info = get_kernel_info(prng_family)
    
    if 'pytorch_gpu' not in info:
        available = list_pytorch_gpu_prngs()
        raise ValueError(
            f"PyTorch GPU not implemented for '{prng_family}'. "
            f"Available GPU PRNGs: {available}. "
            f"Use get_cpu_reference('{prng_family}') for CPU fallback."
        )
    
    return info['pytorch_gpu']


def has_pytorch_gpu(prng_family: str) -> bool:
    """Check if PRNG has PyTorch GPU implementation."""
    if not PYTORCH_AVAILABLE:
        return False
    
    try:
        info = get_kernel_info(prng_family)
        return 'pytorch_gpu' in info
    except ValueError:
        return False


def list_pytorch_gpu_prngs() -> List[str]:
    """List all PRNGs with PyTorch GPU support."""
    if not PYTORCH_AVAILABLE:
        return []
    
    return [
        name for name, info in KERNEL_REGISTRY.items()
        if 'pytorch_gpu' in info
    ]

'''

# Find where to insert (before get_kernel_info)
for i, line in enumerate(new_lines):
    if 'def get_kernel_info(' in line:
        new_lines.insert(i, helper_functions)
        break

print("✅ Added helper functions")

# Part 4: Update KERNEL_REGISTRY entries
# Find java_lcg and java_lcg_hybrid entries and add pytorch_gpu key

in_java_lcg = False
in_java_lcg_hybrid = False
updated_count = 0

for i, line in enumerate(new_lines):
    # Detect java_lcg entry
    if "'java_lcg':" in line and "'java_lcg_hybrid'" not in line:
        in_java_lcg = True
    
    # Detect java_lcg_hybrid entry
    if "'java_lcg_hybrid':" in line:
        in_java_lcg_hybrid = True
    
    # Add pytorch_gpu after cpu_reference
    if in_java_lcg and "'cpu_reference': java_lcg_cpu," in line:
        new_lines.insert(i + 1, "        'pytorch_gpu': java_lcg_pytorch_gpu,\n")
        in_java_lcg = False
        updated_count += 1
    
    if in_java_lcg_hybrid and "'cpu_reference': java_lcg_cpu," in line:
        new_lines.insert(i + 1, "        'pytorch_gpu': java_lcg_hybrid_pytorch_gpu,\n")
        in_java_lcg_hybrid = False
        updated_count += 1

print(f"✅ Updated {updated_count} KERNEL_REGISTRY entries")

# Write new file
with open('prng_registry.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Wrote {len(new_lines)} lines to prng_registry.py")

# Verify syntax
print()
print("✅ Verifying syntax...")
result = os.system("python3 -c 'import prng_registry' 2>/dev/null")

if result == 0:
    print("✅ prng_registry.py syntax OK")
    
    # Test PyTorch GPU availability
    print()
    print("✅ Testing PyTorch GPU support...")
    os.system("""python3 << 'EOF'
from prng_registry import list_pytorch_gpu_prngs, PYTORCH_AVAILABLE
print(f"PYTORCH_AVAILABLE: {PYTORCH_AVAILABLE}")
prngs = list_pytorch_gpu_prngs()
print(f"GPU PRNGs: {prngs}")
if prngs == ['java_lcg', 'java_lcg_hybrid']:
    print("✅ PyTorch GPU support correctly added!")
else:
    print(f"⚠️  Expected ['java_lcg', 'java_lcg_hybrid'], got {prngs}")
EOF""")
    
    print()
    print("=" * 70)
    print("✅ UPDATE COMPLETE!")
    print("=" * 70)
    print()
    print(f"Backup: {backup_file}")
    print(f"Modified: prng_registry.py")
    sys.exit(0)
else:
    print("❌ Syntax error detected!")
    print("   Restoring from backup...")
    os.system(f"cp {backup_file} prng_registry.py")
    print("   Restored: prng_registry.py")
    sys.exit(1)
