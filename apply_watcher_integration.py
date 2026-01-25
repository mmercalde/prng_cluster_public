#!/usr/bin/env python3
"""
WATCHER Integration Patcher - Preflight Check + GPU Cleanup
============================================================
Version: 1.0.0

This script patches agents/watcher_agent.py to integrate:
  - preflight_check.py (Team Beta Item A)
  - gpu_cleanup.py (Team Beta Item C)

Usage:
    python3 apply_watcher_integration.py
    python3 apply_watcher_integration.py --dry-run  # Preview changes
"""

import os
import sys
import shutil
from datetime import datetime

WATCHER_PATH = "agents/watcher_agent.py"

# ============================================================================
# PATCH CONTENT
# ============================================================================

IMPORT_BLOCK = '''# Preflight Check Integration (Team Beta Item A)
try:
    from preflight_check import PreflightChecker
    PREFLIGHT_AVAILABLE = True
except ImportError:
    PREFLIGHT_AVAILABLE = False
    PreflightChecker = None

# GPU Cleanup Integration (Team Beta Item C)
try:
    from gpu_cleanup import post_batch_cleanup, cleanup_all_nodes
    GPU_CLEANUP_AVAILABLE = True
except ImportError:
    GPU_CLEANUP_AVAILABLE = False

# Steps that use distributed GPU cluster (need cleanup)
DISTRIBUTED_STEPS = {1, 2, 3}

'''

HELPER_METHODS = '''
    # ════════════════════════════════════════════════════════════════════════
    # PREFLIGHT AND CLEANUP HELPERS (Team Beta Integration)
    # ════════════════════════════════════════════════════════════════════════

    def _run_preflight_check(self, step: int) -> Tuple[bool, str]:
        """
        Run preflight checks before executing a step.
        
        Returns:
            (passed, message) - passed=False blocks step execution
        """
        if not PREFLIGHT_AVAILABLE:
            logger.debug("Preflight check not available - skipping")
            return True, "Preflight skipped (module not available)"
        
        try:
            checker = PreflightChecker()
            result = checker.check_all(step)
            
            # Categorize: SSH/ramdisk/input failures are hard blocks
            # GPU count mismatches are warnings only
            hard_fail_keywords = ["ssh", "unreachable", "ramdisk", "input", "not found"]
            hard_failures = [
                f for f in result.failures 
                if any(kw in f.lower() for kw in hard_fail_keywords)
            ]
            
            if hard_failures:
                msg = f"Preflight BLOCKED: {'; '.join(hard_failures)}"
                logger.error(msg)
                return False, msg
            
            # Warnings only - proceed
            if result.failures:
                logger.warning(f"Preflight warnings: {result.failures}")
            
            return True, result.summary() if hasattr(result, "summary") else "OK"
            
        except Exception as e:
            logger.warning(f"Preflight check error (non-blocking): {e}")
            return True, f"Preflight error (proceeding anyway): {e}"

    def _run_post_step_cleanup(self, step: int) -> None:
        """
        Run GPU cleanup after distributed steps.
        This is best-effort only - failures never block the pipeline.
        """
        if step not in DISTRIBUTED_STEPS:
            return
        
        if not GPU_CLEANUP_AVAILABLE:
            logger.debug("GPU cleanup not available - skipping")
            return
        
        try:
            logger.info(f"[CLEANUP] Running post-step cleanup for Step {step}")
            result = cleanup_all_nodes()
            nodes_cleaned = result.get("nodes_cleaned", 0) if isinstance(result, dict) else 0
            logger.info(f"[CLEANUP] Complete: {nodes_cleaned} nodes cleaned")
        except Exception as e:
            # Never block on cleanup failure
            logger.warning(f"[CLEANUP] Warning (non-blocking): {e}")

'''

PREFLIGHT_CALL = '''
        # PREFLIGHT CHECK (Team Beta Item A)
        preflight_passed, preflight_msg = self._run_preflight_check(step)
        if not preflight_passed:
            return {
                "success": False,
                "error": preflight_msg,
                "blocked_by": "preflight_check"
            }
'''

CLEANUP_CALL = '''            # POST-STEP CLEANUP (Team Beta Item C)
            self._run_post_step_cleanup(step)
'''


def patch_watcher(dry_run=False):
    """Apply all patches to watcher_agent.py"""
    
    if not os.path.exists(WATCHER_PATH):
        print(f"❌ Error: {WATCHER_PATH} not found")
        print("   Run from ~/distributed_prng_analysis/")
        return False
    
    # Read current content
    with open(WATCHER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "PREFLIGHT_AVAILABLE" in content:
        print("⚠️  Already patched (PREFLIGHT_AVAILABLE found)")
        return True
    
    original_content = content
    
    # ========================================================================
    # PATCH 1: Add imports before "from agents.safety import"
    # ========================================================================
    marker1 = "from agents.safety import"
    if marker1 not in content:
        print(f"❌ Cannot find marker: {marker1}")
        return False
    
    content = content.replace(marker1, IMPORT_BLOCK + marker1)
    print("  ✓ Patch 1: Imports added")
    
    # ========================================================================
    # PATCH 2: Add helper methods before "# STEP EXECUTION" comment
    # ========================================================================
    marker2 = "    # STEP EXECUTION"
    if marker2 not in content:
        print(f"❌ Cannot find marker: {marker2}")
        return False
    
    content = content.replace(marker2, HELPER_METHODS + marker2)
    print("  ✓ Patch 2: Helper methods added")
    
    # ========================================================================
    # PATCH 3: Add preflight call after "Running Step {step}" log line
    # ========================================================================
    marker3 = 'logger.info(f"Running Step {step}: {script}")'
    if marker3 not in content:
        print(f"❌ Cannot find marker: {marker3}")
        return False
    
    content = content.replace(marker3, marker3 + PREFLIGHT_CALL)
    print("  ✓ Patch 3: Preflight call added")
    
    # ========================================================================
    # PATCH 4: Add cleanup call before "# Try to find and load results"
    # ========================================================================
    marker4 = "            # Try to find and load results file"
    if marker4 not in content:
        print(f"❌ Cannot find marker: {marker4}")
        return False
    
    content = content.replace(marker4, CLEANUP_CALL + "\n" + marker4)
    print("  ✓ Patch 4: Cleanup call added")
    
    # ========================================================================
    # Write output
    # ========================================================================
    if dry_run:
        print("\n[DRY RUN] Would write patched content")
        print(f"  Original: {len(original_content)} bytes")
        print(f"  Patched:  {len(content)} bytes")
        print(f"  Added:    {len(content) - len(original_content)} bytes")
        return True
    
    # Create backup
    backup_path = f"{WATCHER_PATH}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(WATCHER_PATH, backup_path)
    print(f"\n  Backup: {backup_path}")
    
    # Write patched file
    with open(WATCHER_PATH, 'w') as f:
        f.write(content)
    
    print(f"  Written: {WATCHER_PATH}")
    
    # Verify
    print("\nVerifying...")
    try:
        import py_compile
        py_compile.compile(WATCHER_PATH, doraise=True)
        print("  ✓ Syntax check passed")
    except py_compile.PyCompileError as e:
        print(f"  ❌ Syntax error: {e}")
        print("  Restoring backup...")
        shutil.copy(backup_path, WATCHER_PATH)
        return False
    
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 60)
    print("WATCHER Integration Patcher")
    print("=" * 60)
    if dry_run:
        print("[DRY RUN MODE - No changes will be written]")
    print()
    
    success = patch_watcher(dry_run)
    
    print()
    print("=" * 60)
    if success:
        print("✅ WATCHER Integration Complete")
        print("=" * 60)
        print()
        print("Test with:")
        print('  PYTHONPATH=. python3 -c "from agents.watcher_agent import WatcherAgent; print(\'OK\')"')
        print()
        print("Run pipeline:")
        print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 4 --end-step 6")
    else:
        print("❌ WATCHER Integration FAILED")
        print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
