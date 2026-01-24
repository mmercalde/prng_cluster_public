#!/usr/bin/env python3
"""
Step-Aware Batching Patch for scripts_coordinator.py
=====================================================
Team Beta Decision - January 22, 2026

Implements:
  - Step 3: MAX_JOBS_PER_NODE_PER_BATCH = 2, INTER_BATCH_COOLDOWN = 10.0
  - Step 2.5 and others: Keep MAX_JOBS_PER_NODE_PER_BATCH = 6, INTER_BATCH_COOLDOWN = 5.0
  - Clear logging of effective limits
  - Job type detection via job file name or content

Usage:
  python3 patch_step_aware_batching.py [--dry-run] [--backup]
  
Options:
  --dry-run   Show what would change without modifying files
  --backup    Create timestamped backup before patching (default: True)
"""

import sys
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================================
# PATCH DEFINITIONS
# ============================================================================

# Old constants block to find
OLD_CONSTANTS = '''# JOB BATCHING (Team Beta Approved - 2026-01-20)
# =============================================================================
# Prevents rig overload when dispatching large trial counts.
# Mirrors benchmark_sample_sizes_v2.sh proven behavior.

MAX_JOBS_PER_BATCH = 20          # Validated stable limit (benchmark: 100% success)
MAX_JOBS_PER_NODE_PER_BATCH = 6   # ROCm nodes crash with >6 simultaneous HIP inits
INTER_BATCH_COOLDOWN = 5.0       # Seconds between batches
ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)'''

# New constants block with step-aware settings
NEW_CONSTANTS = '''# JOB BATCHING (Team Beta Approved - 2026-01-22)
# =============================================================================
# Prevents rig overload when dispatching large trial counts.
# Step-aware batching: Step 3 is 6-7x heavier than Step 2.5.
#
# TEAM BETA RULING (2026-01-22):
#   Step 3 exceeded validated envelope with cap=6 (GPU unknown states, crashes)
#   Step 3: cap=2, cooldown=10s
#   Step 2.5: cap=6, cooldown=5s (unchanged)

MAX_JOBS_PER_BATCH = 20          # Validated stable limit (benchmark: 100% success)
ENABLE_ALLOCATOR_RESET = True    # Reset memory between batches (drop_caches)

# Step 2.5 and other steps (default)
DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH = 6   # ROCm nodes crash with >6 simultaneous HIP inits
DEFAULT_INTER_BATCH_COOLDOWN = 5.0        # Seconds between batches

# Step 3 (Full Scoring) - heavier workload requires conservative limits
STEP3_MAX_JOBS_PER_NODE_PER_BATCH = 2     # Team Beta ruling: cap=2 for Step 3
STEP3_INTER_BATCH_COOLDOWN = 10.0         # Team Beta ruling: cooldown=10s for Step 3

# Legacy aliases (for backward compatibility with any direct references)
MAX_JOBS_PER_NODE_PER_BATCH = DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH
INTER_BATCH_COOLDOWN = DEFAULT_INTER_BATCH_COOLDOWN'''

# Job type detection function to insert after constants
JOB_TYPE_DETECTION = '''

# =============================================================================
# STEP DETECTION (Team Beta - 2026-01-22)
# =============================================================================

def detect_job_step(jobs_file: str, jobs: list = None) -> str:
    """
    Detect which pipeline step these jobs belong to.
    
    Returns:
        'step3' for Full Scoring jobs
        'step2.5' for Scorer Meta-Optimizer jobs
        'unknown' for other job types
    
    Detection methods (in priority order):
    1. Explicit 'job_type' field in job spec
    2. Jobs file name pattern
    3. Script name in job command
    """
    jobs_file_lower = jobs_file.lower() if jobs_file else ''
    
    # Method 1: Check job specs for explicit job_type
    if jobs:
        for job in jobs[:5]:  # Check first 5 jobs
            job_type = job.get('job_type', '')
            if job_type == 'full_scoring':
                return 'step3'
            elif job_type == 'scorer_trial':
                return 'step2.5'
    
    # Method 2: Jobs file name pattern
    if 'scoring_jobs' in jobs_file_lower and 'scorer' not in jobs_file_lower:
        return 'step3'
    if 'full_scoring' in jobs_file_lower:
        return 'step3'
    if 'scorer_jobs' in jobs_file_lower or 'scorer_trial' in jobs_file_lower:
        return 'step2.5'
    
    # Method 3: Check script names in jobs
    if jobs:
        for job in jobs[:5]:
            cmd = job.get('command', '') or job.get('script', '')
            if 'full_scoring_worker' in cmd:
                return 'step3'
            if 'scorer_trial_worker' in cmd:
                return 'step2.5'
    
    return 'unknown'


def get_step_aware_limits(step: str) -> tuple:
    """
    Get batching limits for the detected step.
    
    Returns:
        (max_jobs_per_node, cooldown_seconds, step_name)
    """
    if step == 'step3':
        return (STEP3_MAX_JOBS_PER_NODE_PER_BATCH, 
                STEP3_INTER_BATCH_COOLDOWN,
                'Step 3 (Full Scoring)')
    else:
        return (DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH,
                DEFAULT_INTER_BATCH_COOLDOWN,
                f'Step 2.5/Other ({step})')
'''

# Pattern to find the execution method's batch mode section
# We need to modify where it uses MAX_JOBS_PER_NODE_PER_BATCH and INTER_BATCH_COOLDOWN

OLD_BATCH_CHECK = '''        # Check if batching needed
        total_jobs = len(self.jobs)
        if total_jobs > MAX_JOBS_PER_BATCH:
            # Batched execution
            num_batches = (total_jobs + MAX_JOBS_PER_BATCH - 1) // MAX_JOBS_PER_BATCH
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches of ≤{MAX_JOBS_PER_BATCH}")
            
            # Build job queues per node
            node_queues = {}
            for node in self.nodes:
                node_queues[node.hostname] = list(assignments[node.hostname])
            
            # Form batches respecting per-node limits
            all_batches = []
            while any(node_queues.values()):
                batch = []
                for node in self.nodes:
                    queue = node_queues[node.hostname]
                    take = min(len(queue), MAX_JOBS_PER_NODE_PER_BATCH)
                    for _ in range(take):
                        if len(batch) < MAX_JOBS_PER_BATCH:
                            batch.append((node, queue.pop(0)))
                if batch:
                    all_batches.append(batch)
            
            num_batches = len(all_batches)
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches (max {MAX_JOBS_PER_NODE_PER_BATCH}/node)")'''

NEW_BATCH_CHECK = '''        # Check if batching needed
        total_jobs = len(self.jobs)
        if total_jobs > MAX_JOBS_PER_BATCH:
            # Step-aware batching (Team Beta 2026-01-22)
            detected_step = detect_job_step(self.jobs_file, self.jobs)
            effective_max_per_node, effective_cooldown, step_name = get_step_aware_limits(detected_step)
            
            # Batched execution
            num_batches = (total_jobs + MAX_JOBS_PER_BATCH - 1) // MAX_JOBS_PER_BATCH
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches of ≤{MAX_JOBS_PER_BATCH}")
            print(f"  [STEP-AWARE] Detected: {step_name}")
            print(f"  [STEP-AWARE] Limits: max_per_node={effective_max_per_node}, cooldown={effective_cooldown}s")
            
            # Build job queues per node
            node_queues = {}
            for node in self.nodes:
                node_queues[node.hostname] = list(assignments[node.hostname])
            
            # Form batches respecting per-node limits (using step-aware value)
            all_batches = []
            while any(node_queues.values()):
                batch = []
                for node in self.nodes:
                    queue = node_queues[node.hostname]
                    take = min(len(queue), effective_max_per_node)
                    for _ in range(take):
                        if len(batch) < MAX_JOBS_PER_BATCH:
                            batch.append((node, queue.pop(0)))
                if batch:
                    all_batches.append(batch)
            
            num_batches = len(all_batches)
            print(f"  [BATCH MODE] {total_jobs} jobs → {num_batches} batches (max {effective_max_per_node}/node)")'''

# Also need to update the cooldown reference
OLD_COOLDOWN = '''                if batch_num > 0:
                    self._reset_allocator_state()
                    print(f"  Cooling down {INTER_BATCH_COOLDOWN}s...")
                    time.sleep(INTER_BATCH_COOLDOWN)'''

NEW_COOLDOWN = '''                if batch_num > 0:
                    self._reset_allocator_state()
                    print(f"  Cooling down {effective_cooldown}s...")
                    time.sleep(effective_cooldown)'''


# ============================================================================
# PATCH APPLICATION
# ============================================================================

def create_backup(filepath: Path) -> Path:
    """Create timestamped backup of file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = filepath.parent / f"{filepath.name}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def check_already_patched(content: str) -> bool:
    """Check if patch has already been applied."""
    markers = [
        'STEP3_MAX_JOBS_PER_NODE_PER_BATCH',
        'detect_job_step',
        'get_step_aware_limits',
        'Team Beta 2026-01-22'
    ]
    return all(marker in content for marker in markers)


def apply_patch(filepath: Path, dry_run: bool = False) -> tuple:
    """
    Apply step-aware batching patch to scripts_coordinator.py.
    
    Returns:
        (success: bool, message: str, changes: list)
    """
    if not filepath.exists():
        return (False, f"File not found: {filepath}", [])
    
    content = filepath.read_text()
    changes = []
    
    # Check if already patched
    if check_already_patched(content):
        return (True, "Already patched (idempotent - no changes needed)", [])
    
    # Apply patches in sequence
    new_content = content
    
    # Patch 1: Update constants block
    if OLD_CONSTANTS in new_content:
        new_content = new_content.replace(OLD_CONSTANTS, NEW_CONSTANTS)
        changes.append("Updated constants block with step-aware settings")
    else:
        # Try to find partial match and warn
        if 'MAX_JOBS_PER_BATCH = 20' in new_content and 'STEP3_MAX_JOBS_PER_NODE_PER_BATCH' not in new_content:
            return (False, "Constants block format changed - manual review needed", [])
    
    # Patch 2: Insert job type detection functions after constants
    # Find insertion point (after INTER_BATCH_COOLDOWN line in new constants)
    if JOB_TYPE_DETECTION.strip() not in new_content:
        # Insert after the constants block
        insert_marker = 'INTER_BATCH_COOLDOWN = DEFAULT_INTER_BATCH_COOLDOWN'
        if insert_marker in new_content:
            new_content = new_content.replace(
                insert_marker,
                insert_marker + JOB_TYPE_DETECTION
            )
            changes.append("Added detect_job_step() and get_step_aware_limits() functions")
    
    # Patch 3: Update batch check section
    if OLD_BATCH_CHECK in new_content:
        new_content = new_content.replace(OLD_BATCH_CHECK, NEW_BATCH_CHECK)
        changes.append("Updated batch formation to use step-aware limits")
    elif 'detect_job_step' not in new_content:
        # The exact text didn't match but we need to patch
        return (False, "Batch check section format changed - manual review needed", [])
    
    # Patch 4: Update cooldown reference
    if OLD_COOLDOWN in new_content:
        new_content = new_content.replace(OLD_COOLDOWN, NEW_COOLDOWN)
        changes.append("Updated cooldown to use step-aware value")
    
    if not changes:
        return (False, "No patches could be applied - file format may have changed", [])
    
    if dry_run:
        return (True, "Dry run - no changes written", changes)
    
    # Write patched content
    filepath.write_text(new_content)
    return (True, "Patch applied successfully", changes)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply step-aware batching patch to scripts_coordinator.py'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without modifying files')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating backup (not recommended)')
    parser.add_argument('--file', type=str, 
                        default='scripts_coordinator.py',
                        help='Path to scripts_coordinator.py')
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    if not filepath.is_absolute():
        # Try current directory first, then common locations
        candidates = [
            Path.cwd() / args.file,
            Path.home() / 'distributed_prng_analysis' / args.file,
            Path('/home/michael/distributed_prng_analysis') / args.file,
        ]
        for candidate in candidates:
            if candidate.exists():
                filepath = candidate
                break
    
    print("=" * 70)
    print("Step-Aware Batching Patch (Team Beta - 2026-01-22)")
    print("=" * 70)
    print(f"\nTarget: {filepath}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print()
    
    if not filepath.exists():
        print(f"❌ ERROR: File not found: {filepath}")
        print("\nTry specifying the full path:")
        print("  python3 patch_step_aware_batching.py --file /path/to/scripts_coordinator.py")
        sys.exit(1)
    
    # Create backup unless disabled or dry run
    if not args.dry_run and not args.no_backup:
        backup_path = create_backup(filepath)
        print(f"✅ Backup created: {backup_path}")
    
    # Apply patch
    success, message, changes = apply_patch(filepath, dry_run=args.dry_run)
    
    print(f"\nResult: {message}")
    
    if changes:
        print("\nChanges:")
        for i, change in enumerate(changes, 1):
            print(f"  {i}. {change}")
    
    print()
    if success:
        print("✅ Patch complete")
        if not args.dry_run:
            print("\nNew effective limits:")
            print("  Step 3 (Full Scoring): max_per_node=2, cooldown=10s")
            print("  Step 2.5 (Scorer Meta): max_per_node=6, cooldown=5s")
            print("\nTo verify, run:")
            print(f"  grep -A 5 'STEP3_MAX_JOBS' {filepath}")
    else:
        print("❌ Patch failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
