#!/usr/bin/env python3
"""
apply_stagger_fix.py - Automatic patch for scripts_coordinator.py
Team Beta Approved - January 16, 2026

Fixes HIP initialization collision on ROCm rigs by adding staggered GPU worker startup.

Usage:
    python3 apply_stagger_fix.py

This will:
1. Create a backup of scripts_coordinator.py
2. Verify 'import time' exists (add if missing)
3. Apply the stagger fix to lines 524-532
4. Verify syntax
"""

import os
import sys
import shutil
from datetime import datetime

SCRIPT_PATH = "scripts_coordinator.py"

# The OLD code block to find and replace (lines 524-532)
OLD_CODE = '''        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(gpu_worker, gpu_id, gpu_jobs[gpu_id])
                for gpu_id in active_gpus
            ]
            for future in as_completed(futures):
                try:
                    future.result()  # Raises if worker had exception
                except Exception as e:
                    print(f"  ✗ Worker exception on {node.hostname}: {e}")'''

# The NEW code block with staggered worker startup
# Uses node.is_localhost (VERIFIED EXISTS at line 119)
# Stagger when: NOT localhost OR multi-GPU
NEW_CODE = '''        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, gpu_id in enumerate(active_gpus):
                futures.append(executor.submit(gpu_worker, gpu_id, gpu_jobs[gpu_id]))
                
                # Stagger GPU worker startup to prevent HIP init collision
                # Stagger on: remote nodes OR multi-GPU nodes
                if i < len(active_gpus) - 1:
                    if not node.is_localhost or len(active_gpus) > 1:
                        time.sleep(node.stagger_delay)
            
            for future in as_completed(futures):
                try:
                    future.result()  # Raises if worker had exception
                except Exception as e:
                    print(f"  ✗ Worker exception on {node.hostname}: {e}")'''


def main():
    # Check file exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ ERROR: {SCRIPT_PATH} not found!")
        print(f"   Run this script from ~/distributed_prng_analysis/")
        sys.exit(1)
    
    # Read current content
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "Stagger GPU worker startup to prevent HIP init collision" in content:
        print("⚠️  Already patched! The stagger fix is already applied.")
        sys.exit(0)
    
    # Check if old code exists
    if OLD_CODE not in content:
        print("❌ ERROR: Could not find the expected code block to replace!")
        print("   The code may have been modified. Manual patching required.")
        print("\n   Looking for this block:")
        print("   " + OLD_CODE[:100] + "...")
        sys.exit(1)
    
    # STEP 1: Create backup FIRST
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{SCRIPT_PATH}.backup_{timestamp}"
    shutil.copy2(SCRIPT_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # STEP 2: Verify 'import time' exists
    if 'import time' not in content:
        print("⚠️  'import time' not found - adding it...")
        # Add after 'import json' (safe location near top of imports)
        if 'import json' in content:
            content = content.replace('import json', 'import json\nimport time')
            print("✅ Added 'import time' after 'import json'")
        else:
            print("❌ ERROR: Could not find safe location to add 'import time'")
            print("   Please add 'import time' manually near the top of the file")
            sys.exit(1)
    else:
        print("✅ 'import time' already present")
    
    # STEP 3: Apply patch
    new_content = content.replace(OLD_CODE, NEW_CODE)
    
    # Write patched file
    with open(SCRIPT_PATH, 'w') as f:
        f.write(new_content)
    print(f"✅ Patch applied to {SCRIPT_PATH}")
    
    # STEP 4: Verify syntax
    print("🔍 Verifying syntax...")
    import py_compile
    try:
        py_compile.compile(SCRIPT_PATH, doraise=True)
        print("✅ Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"❌ SYNTAX ERROR: {e}")
        print(f"   Restoring backup...")
        shutil.copy2(backup_path, SCRIPT_PATH)
        print(f"   Restored from {backup_path}")
        sys.exit(1)
    
    # Show what changed
    print("\n" + "="*60)
    print("PATCH APPLIED SUCCESSFULLY")
    print("="*60)
    print("""
What changed:
  OLD: List comprehension submits ALL 12 workers simultaneously
       futures = [executor.submit(...) for gpu_id in active_gpus]

  NEW: Loop with stagger between submissions
       for i, gpu_id in enumerate(active_gpus):
           futures.append(executor.submit(...))
           if i < len(active_gpus) - 1:
               if not node.is_localhost or len(active_gpus) > 1:
                   time.sleep(node.stagger_delay)

Stagger behavior:
  - localhost (Zeus, 2 GPUs): 3.0s stagger (STAGGER_LOCALHOST)
  - remote rigs (12 GPUs each): 0.5s stagger (STAGGER_REMOTE)
  - Total ROCm startup window: ~5.5s (was 0s)

Test with:
  bash run_scorer_meta_optimizer.sh 50
""")


if __name__ == "__main__":
    main()
