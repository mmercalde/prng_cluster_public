#!/usr/bin/env python3
"""
apply_job_batching_patch.py - Apply job batching to scripts_coordinator.py

Team Beta Approved: 2026-01-20
Purpose: Prevent rig overload when dispatching large trial counts

Usage:
    python3 apply_job_batching_patch.py
    
Requires: Run from ~/distributed_prng_analysis directory
"""

import os
import sys
import shutil
from datetime import datetime

SCRIPT_PATH = 'scripts_coordinator.py'

# =============================================================================
# PATCH CONTENT
# =============================================================================

CONSTANTS_TO_ADD = '''
# =============================================================================
# JOB BATCHING (Team Beta Approved - 2026-01-20)
# =============================================================================
# Prevents rig overload when dispatching large trial counts.
# Mirrors benchmark_sample_sizes_v2.sh proven behavior.

MAX_JOBS_PER_BATCH = 20          # Validated stable limit (benchmark: 100% success)
INTER_BATCH_COOLDOWN = 5.0       # Seconds between batches
ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)
'''

RESET_METHOD = '''    def _reset_allocator_state(self):
        """
        Reset memory allocator on remote ROCm nodes.
        
        Mirrors benchmark_sample_sizes_v2.sh behavior:
        - sync filesystems
        - drop page cache (echo 3 > drop_caches)
        
        Note: localhost reset intentionally skipped (CUDA manages its own memory).
        """
        if not ENABLE_ALLOCATOR_RESET:
            return
            
        print("  Resetting allocator state on ROCm nodes...")
        for node in self.nodes:
            if not node.is_localhost:
                try:
                    subprocess.run(
                        ['ssh', f'{node.username}@{node.hostname}',
                         'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true'],
                        capture_output=True, timeout=10
                    )
                except Exception as e:
                    print(f"    Warning: Failed to reset {node.hostname}: {e}")
        time.sleep(2)  # Brief pause for stability
        print("  Allocator reset complete")

'''

OLD_EXECUTION_BLOCK = '''        print("-" * 60)
        print("Executing jobs...")
        print("-" * 60)
        
        # Launch node executor threads
        threads = []
        for node in self.nodes:
            node_jobs = assignments[node.hostname]
            if node_jobs:
                t = threading.Thread(
                    target=self._node_executor,
                    args=(node, node_jobs),
                    name=f"executor-{node.hostname}"
                )
                threads.append(t)
                t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()'''

NEW_EXECUTION_BLOCK = '''        print("-" * 60)
        print("Executing jobs...")
        print("-" * 60)
        
        # Check if batching needed
        total_jobs = len(self.jobs)
        if total_jobs > MAX_JOBS_PER_BATCH:
            # Batched execution
            num_batches = (total_jobs + MAX_JOBS_PER_BATCH - 1) // MAX_JOBS_PER_BATCH
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches of ≤{MAX_JOBS_PER_BATCH}")
            
            # Flatten all jobs with node assignment for batching
            all_assigned_jobs = []
            for node in self.nodes:
                for job in assignments[node.hostname]:
                    all_assigned_jobs.append((node, job))
            
            # Process in batches
            for batch_num in range(num_batches):
                batch_start = batch_num * MAX_JOBS_PER_BATCH
                batch_end = min(batch_start + MAX_JOBS_PER_BATCH, total_jobs)
                batch_jobs = all_assigned_jobs[batch_start:batch_end]
                
                print(f"\\n  {'='*50}")
                print(f"  [BATCH {batch_num + 1}/{num_batches}] Jobs {batch_start + 1}-{batch_end}")
                print(f"  {'='*50}")
                
                # Reset allocator before batch (except first)
                if batch_num > 0:
                    self._reset_allocator_state()
                    print(f"  Cooling down {INTER_BATCH_COOLDOWN}s...")
                    time.sleep(INTER_BATCH_COOLDOWN)
                
                # Group batch jobs by node
                batch_by_node = {}
                for node, job in batch_jobs:
                    if node.hostname not in batch_by_node:
                        batch_by_node[node.hostname] = []
                    batch_by_node[node.hostname].append(job)
                
                # Launch node executor threads for this batch
                threads = []
                for node in self.nodes:
                    node_jobs = batch_by_node.get(node.hostname, [])
                    if node_jobs:
                        t = threading.Thread(
                            target=self._node_executor,
                            args=(node, node_jobs),
                            name=f"executor-{node.hostname}-batch{batch_num}"
                        )
                        threads.append(t)
                        t.start()
                
                # Wait for batch to complete
                for t in threads:
                    t.join()
                
                # Batch summary
                batch_successful = sum(1 for r in self.results[batch_start:] if r.success)
                print(f"  [BATCH {batch_num + 1}] Complete: {batch_successful}/{len(batch_jobs)} successful")
        else:
            # Original non-batched execution for small job counts
            threads = []
            for node in self.nodes:
                node_jobs = assignments[node.hostname]
                if node_jobs:
                    t = threading.Thread(
                        target=self._node_executor,
                        args=(node, node_jobs),
                        name=f"executor-{node.hostname}"
                    )
                    threads.append(t)
                    t.start()
            
            # Wait for all threads
            for t in threads:
                t.join()'''

# =============================================================================
# PATCH APPLICATION
# =============================================================================

def main():
    print("=" * 60)
    print("JOB BATCHING PATCH - scripts_coordinator.py")
    print("Team Beta Approved: 2026-01-20")
    print("=" * 60)
    
    # Check file exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ ERROR: {SCRIPT_PATH} not found")
        print("   Run this script from ~/distributed_prng_analysis")
        sys.exit(1)
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{SCRIPT_PATH}.backup_{timestamp}"
    shutil.copy(SCRIPT_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Read current content
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'MAX_JOBS_PER_BATCH' in content:
        print("⚠️  WARNING: MAX_JOBS_PER_BATCH already exists in file")
        print("   Patch may have been applied already")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Aborted.")
            sys.exit(0)
    
    # Apply patches
    modified = content
    patches_applied = 0
    
    # Patch 1: Add constants after ROCM_HOSTNAMES
    marker = "ROCM_HOSTNAMES = ['192.168.3.120', '192.168.3.154', 'rig-6600', 'rig-6600b']"
    if marker in modified:
        modified = modified.replace(marker, marker + CONSTANTS_TO_ADD)
        patches_applied += 1
        print("✅ Patch 1: Added batching constants")
    else:
        print("❌ Patch 1 FAILED: Could not find ROCM_HOSTNAMES marker")
    
    # Patch 2: Add _reset_allocator_state method before run()
    run_marker = "    def run(self) -> Dict[str, Any]:"
    if run_marker in modified:
        modified = modified.replace(run_marker, RESET_METHOD + run_marker)
        patches_applied += 1
        print("✅ Patch 2: Added _reset_allocator_state() method")
    else:
        print("❌ Patch 2 FAILED: Could not find run() method marker")
    
    # Patch 3: Replace execution block with batched version
    if OLD_EXECUTION_BLOCK in modified:
        modified = modified.replace(OLD_EXECUTION_BLOCK, NEW_EXECUTION_BLOCK)
        patches_applied += 1
        print("✅ Patch 3: Added batched execution logic")
    else:
        print("❌ Patch 3 FAILED: Could not find execution block")
        print("   This may be due to whitespace differences")
        print("   Manual edit required (see IMPLEMENTATION_job_batching_scripts_coordinator.md)")
    
    # Write modified content
    if patches_applied >= 2:  # At least constants and method added
        with open(SCRIPT_PATH, 'w') as f:
            f.write(modified)
        print(f"\n✅ Wrote modified {SCRIPT_PATH}")
    else:
        print(f"\n❌ Not enough patches applied ({patches_applied}/3)")
        print(f"   Restoring from backup...")
        shutil.copy(backup_path, SCRIPT_PATH)
        sys.exit(1)
    
    # Verify syntax
    print("\nVerifying Python syntax...")
    result = os.system(f"python3 -m py_compile {SCRIPT_PATH}")
    if result == 0:
        print("✅ Syntax verification passed")
    else:
        print("❌ Syntax error detected!")
        print(f"   Restoring from backup: {backup_path}")
        shutil.copy(backup_path, SCRIPT_PATH)
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("PATCH SUMMARY")
    print("=" * 60)
    print(f"  Patches applied: {patches_applied}/3")
    print(f"  Backup: {backup_path}")
    print(f"  Modified: {SCRIPT_PATH}")
    
    if patches_applied == 3:
        print("\n✅ ALL PATCHES APPLIED SUCCESSFULLY")
    else:
        print(f"\n⚠️  {3 - patches_applied} patch(es) need manual application")
        print("   See: IMPLEMENTATION_job_batching_scripts_coordinator.md")
    
    print("\nNext steps:")
    print("  1. Test with 20 trials: ./run_scorer_meta_optimizer.sh 20")
    print("  2. Test with 50 trials: ./run_scorer_meta_optimizer.sh 50")
    print("  3. Test WATCHER pipeline: PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 --params '{\"trials\": 50}'")

if __name__ == "__main__":
    main()
