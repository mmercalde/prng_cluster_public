#!/usr/bin/env python3
"""
Apply Chapter 14 Phase 6 patches to watcher_agent.py

TEAM BETA REVIEWED — v1.1.0 (fixes applied)

Fixes from review:
1. RETRY semantics clarified — explicit "not yet implemented" behavior
2. _health_check_retries REMOVED — was half-state, worse than none
3. Import anchor hardened — fallback pattern added
4. Added schema_version to incident records

Run from ~/distributed_prng_analysis:
    python3 apply_watcher_health_check_patch.py

Creates backup before patching.
"""

import os
import sys
import shutil
from datetime import datetime

WATCHER_PATH = "agents/watcher_agent.py"

# =============================================================================
# PATCH 1: Import block to add (with hardened anchor detection)
# =============================================================================

# Primary anchor
IMPORT_ANCHOR_PRIMARY = '''except ImportError:
    GPU_CLEANUP_AVAILABLE = False

# Steps that use distributed GPU cluster (need cleanup)'''

# Fallback anchor (just the key line)
IMPORT_ANCHOR_FALLBACK = "GPU_CLEANUP_AVAILABLE = False"

IMPORT_BLOCK = '''
# Training Health Check Integration (Chapter 14 Phase 6)
try:
    from training_health_check import (
        check_training_health,
        reset_skip_registry,
        get_retry_params_suggestions,
    )
    TRAINING_HEALTH_CHECK_AVAILABLE = True
except ImportError:
    TRAINING_HEALTH_CHECK_AVAILABLE = False
    check_training_health = None
    reset_skip_registry = None
    get_retry_params_suggestions = None

'''

# =============================================================================
# PATCH 2: _handle_proceed method replacement
# Team Beta v1.1.0: Removed _health_check_retries, clarified RETRY semantics
# =============================================================================

OLD_HANDLE_PROCEED = '''    def _handle_proceed(self, step: int, context: FullAgentContext) -> bool:
        """Handle proceed action - advance to next step."""
        logger.info(f"✅ Step {step} PASSED - proceeding to next step")

        # Reset retry count for this step
        self.retry_counts[step] = 0

        # Check if pipeline complete
        if step >= 6:
            logger.info("🎉 PIPELINE COMPLETE - all 6 steps finished!")
            self._notify_complete(context)
            return False

        # Trigger next step
        next_step = step + 1
        self.current_step = next_step

        logger.info(f"Triggering Step {next_step}: {STEP_NAMES.get(next_step, 'Unknown')}")
        return True'''

NEW_HANDLE_PROCEED = '''    def _handle_proceed(self, step: int, context: FullAgentContext) -> bool:
        """Handle proceed action - advance to next step."""
        logger.info(f"✅ Step {step} PASSED - proceeding to next step")

        # Reset retry count for this step
        self.retry_counts[step] = 0

        # ════════════════════════════════════════════════════════════════════
        # CHAPTER 14 PHASE 6: Post-Step-5 Training Health Check
        # Design: Best-effort, observational only — never blocks pipeline
        # ════════════════════════════════════════════════════════════════════
        if step == 5 and TRAINING_HEALTH_CHECK_AVAILABLE:
            health = check_training_health()
            
            if health['action'] == 'PROCEED':
                # Training healthy — reset skip registry and continue
                reset_skip_registry(health['model_type'])
                logger.info(f"🏥 Training health OK ({health['model_type']}) — proceeding to Step 6")
            
            elif health['action'] == 'PROCEED_WITH_NOTE':
                # Minor issues — log for Strategy Advisor but continue
                reset_skip_registry(health['model_type'])
                logger.warning(
                    f"🏥 Training health WARNING ({health['model_type']}): "
                    f"{'; '.join(health['issues'][:3])}"
                )
                self._record_training_incident(health)
            
            elif health['action'] == 'RETRY':
                # Critical issues detected
                # NOTE: Automatic retry with param threading NOT YET IMPLEMENTED
                # Current behavior: Log suggestions, record incident, proceed anyway
                # Future: Wire suggestions into run_pipeline() param override
                logger.warning(
                    f"🏥 Training health CRITICAL ({health['model_type']}): "
                    f"{'; '.join(health['issues'][:3])}"
                )
                suggestions = get_retry_params_suggestions(health)
                logger.warning(
                    f"🏥 Retry requested but param-threading not yet implemented. "
                    f"Suggestions for manual review: {suggestions}"
                )
                self._record_training_incident(health)
                # Proceed to Step 6 — diagnostics are observational, not gating
            
            elif health['action'] == 'SKIP_MODEL':
                # Model type repeatedly failing — log and proceed
                # Step 6 continues with remaining models (--compare-models mode)
                logger.error(
                    f"🏥 Model {health['model_type']} SKIPPED — "
                    f"{health.get('consecutive_critical', 0)} consecutive critical failures"
                )
                self._record_training_incident(health)
        
        elif step == 5 and not TRAINING_HEALTH_CHECK_AVAILABLE:
            logger.debug("Training health check not available — skipping")
        # ════════════════════════════════════════════════════════════════════

        # Check if pipeline complete
        if step >= 6:
            logger.info("🎉 PIPELINE COMPLETE - all 6 steps finished!")
            self._notify_complete(context)
            return False

        # Trigger next step
        next_step = step + 1
        self.current_step = next_step

        logger.info(f"Triggering Step {next_step}: {STEP_NAMES.get(next_step, 'Unknown')}")
        return True

    def _record_training_incident(self, health: dict) -> None:
        """
        Record training health incident for Strategy Advisor consumption.
        Best-effort — failures don't block pipeline.
        
        Schema v1.0.0 — incidents written to watcher_training_incidents.jsonl
        """
        try:
            incident = {
                'schema_version': '1.0.0',
                'type': 'training_health',
                'model_type': health.get('model_type', 'unknown'),
                'severity': health.get('severity', 'unknown'),
                'action': health.get('action', 'unknown'),
                'issues': health.get('issues', [])[:5],
                'confidence': health.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat(),
            }
            
            incidents_path = "watcher_training_incidents.jsonl"
            with open(incidents_path, 'a') as f:
                f.write(json.dumps(incident) + '\\n')
            
            logger.debug(f"Recorded training incident: {incident['severity']}")
        except Exception as e:
            logger.warning(f"Failed to record training incident (non-fatal): {e}")'''


def apply_import_patch(content: str) -> tuple:
    """
    Apply import patch with hardened anchor detection.
    Returns (patched_content, success_bool, message)
    """
    if "TRAINING_HEALTH_CHECK_AVAILABLE" in content:
        return content, True, "⚠️  Patch 1 (imports) already applied — skipping"
    
    # Try primary anchor first
    if IMPORT_ANCHOR_PRIMARY in content:
        # Insert after the anchor
        new_content = content.replace(
            IMPORT_ANCHOR_PRIMARY,
            IMPORT_ANCHOR_PRIMARY.replace(
                "# Steps that use distributed GPU cluster",
                IMPORT_BLOCK + "# Steps that use distributed GPU cluster"
            )
        )
        return new_content, True, "✅ Applied Patch 1: Training health check imports (primary anchor)"
    
    # Fallback: find GPU_CLEANUP_AVAILABLE = False and insert after the except block
    if IMPORT_ANCHOR_FALLBACK in content:
        # Find the line and insert after the next blank line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if IMPORT_ANCHOR_FALLBACK in line:
                # Find next blank line or comment
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() == '' or lines[j].startswith('#'):
                        # Insert here
                        lines.insert(j, IMPORT_BLOCK.strip())
                        return '\n'.join(lines), True, "✅ Applied Patch 1: Training health check imports (fallback anchor)"
        
        return content, False, "❌ ERROR: Found fallback anchor but couldn't determine insertion point"
    
    return content, False, "❌ ERROR: Could not find import anchor (neither primary nor fallback)"


def main():
    if not os.path.exists(WATCHER_PATH):
        print(f"ERROR: {WATCHER_PATH} not found")
        print("Run this script from ~/distributed_prng_analysis")
        sys.exit(1)
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{WATCHER_PATH}.pre_health_check_{timestamp}"
    shutil.copy(WATCHER_PATH, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read current content
    with open(WATCHER_PATH, 'r') as f:
        content = f.read()
    
    # Apply Patch 1: Imports (with hardened detection)
    content, success, message = apply_import_patch(content)
    print(message)
    if not success:
        print("   Manual patching required for imports")
        sys.exit(1)
    
    # Apply Patch 2: _handle_proceed method
    if "_record_training_incident" in content:
        print("⚠️  Patch 2 (_handle_proceed) already applied — skipping")
    elif OLD_HANDLE_PROCEED not in content:
        print("❌ ERROR: Could not find _handle_proceed method")
        print("   The method signature may have changed")
        print("   Manual patching required")
        sys.exit(1)
    else:
        content = content.replace(OLD_HANDLE_PROCEED, NEW_HANDLE_PROCEED)
        print("✅ Applied Patch 2: _handle_proceed with health check (Team Beta v1.1.0)")
    
    # Write patched content
    with open(WATCHER_PATH, 'w') as f:
        f.write(content)
    
    print()
    print("=" * 60)
    print("PATCHING COMPLETE (Team Beta Reviewed v1.1.0)")
    print("=" * 60)
    print()
    print("Changes from Team Beta review:")
    print("  • RETRY: Explicit 'not yet implemented' messaging")
    print("  • Removed unreliable _health_check_retries state")
    print("  • Added schema_version to incident records")
    print("  • Hardened import anchor detection")
    print()
    print("Verify with:")
    print("  python3 -c \"from agents.watcher_agent import WatcherAgent; print('Import OK')\"")
    print()
    print("Test health check integration:")
    print("  python3 training_health_check.py --check")
    print()
    print(f"To revert: cp {backup_path} {WATCHER_PATH}")


if __name__ == "__main__":
    main()
